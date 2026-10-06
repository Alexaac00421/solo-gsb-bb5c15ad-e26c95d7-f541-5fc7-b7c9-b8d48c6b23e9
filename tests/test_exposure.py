"""累计温度暴露评估端到端测试。

覆盖：
- 时间窗左闭右开、窗前最近读数覆盖窗头、读数温度沿用到下一条读数或最大
  沿用间隔届满（先到为准）、已观测 / 未知分钟划分；
- 同一采样时刻以事件号最大者为准；撤销读数不参与；
- 温度上限按时间点适用最近生效策略，策略切换点拆段，无策略沿用全局
  上限；逐段 max(温度-上限,0)×分钟，未舍入值与预算比较（超过才不合格）；
- 结论 fail / insufficient_evidence / pass 的优先级与判定；
- 已知设备无有效读数全窗未知；未知设备 404；
- 非法时间窗、非有限数值、非正间隔、负预算 422；
- 评估只读：不改读数 / 区间 / 策略 / 批次 / 凭证 / 争议。
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app

DEVICE = "truck-001"
W0 = "2026-10-01T08:00:00Z"
W1 = "2026-10-01T09:00:00Z"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("TEMP_LIMIT", "8.0")
    with TestClient(app) as c:
        yield c


def post(client, event_no, ts, temp, device=DEVICE):
    return client.post(
        f"/devices/{device}/readings",
        json={"event_no": event_no, "sampled_at": ts, "temperature": temp},
    )


def put_policy(client, policy_no, effective_from, temp_limit, device=DEVICE):
    return client.post(
        f"/devices/{device}/limit-policies",
        json={
            "policy_no": policy_no,
            "effective_from": effective_from,
            "temp_limit": temp_limit,
        },
    )


def assess(client, start, end, staleness, budget, device=DEVICE):
    return client.post(
        f"/devices/{device}/exposure-assessments",
        json={
            "window_start": start,
            "window_end": end,
            "max_staleness_minutes": staleness,
            "exposure_budget": budget,
        },
    )


# ---------------------------------------------------------------- 已观测 / 未知

def test_window_head_covered_by_latest_pre_window_reading(client):
    # 窗前最近读数 07:55, 10.0，沿用 15 分钟 → 覆盖窗头 [08:00, 08:10)
    post(client, "E1", "2026-10-01T07:55:00Z", 10.0)
    r = assess(client, W0, W1, 15, 1000)
    assert r.status_code == 200
    b = r.json()
    assert b["window_minutes"] == 60.0
    assert b["observed_minutes"] == 10.0
    assert b["unknown_minutes"] == 50.0
    # (10-8) * 10
    assert b["cumulative_exposure"] == pytest.approx(20.0)
    assert b["conclusion"] == "insufficient_evidence"


def test_pre_window_reading_superseded_by_in_window_reading(client):
    # 窗前读数本可盖到 08:10，但 08:05 有新读数 → 其温度只保持到 08:05
    post(client, "E1", "2026-10-01T07:55:00Z", 10.0)
    post(client, "E2", "2026-10-01T08:05:00Z", 5.0)
    b = assess(client, W0, W1, 15, 1000).json()
    assert b["observed_minutes"] == 20.0  # E1 盖 5 分钟 + E2 盖 15 分钟
    assert b["unknown_minutes"] == 40.0
    # 仅 E1 的 5 分钟超限：(10-8)*5 = 10；E2=5.0 不超限
    assert b["cumulative_exposure"] == pytest.approx(10.0)


def test_older_pre_window_reading_does_not_cover_window(client):
    # 窗前较早读数在窗前即届满；窗前最近读数才是窗头的依据
    post(client, "E0", "2026-10-01T07:00:00Z", 10.0)
    post(client, "E1", "2026-10-01T07:55:00Z", 4.0)
    b = assess(client, W0, W1, 15, 1000).json()
    # E0 在 07:15 届满（窗外），与窗内无关；E1 盖窗头 10 分钟
    assert b["observed_minutes"] == 10.0
    assert b["cumulative_exposure"] == pytest.approx(0.0)


def test_staleness_expires_before_next_reading_creates_unknown_gap(client):
    # 08:00 读数沿用 10 分钟 → 08:10 起未知，直至 08:20 下一条读数
    post(client, "E1", "2026-10-01T08:00:00Z", 10.0)
    post(client, "E2", "2026-10-01T08:20:00Z", 5.0)
    b = assess(client, W0, W1, 10, 1000).json()
    assert b["observed_minutes"] == 20.0  # [08:00,08:10) 与 [08:20,08:30)
    assert b["unknown_minutes"] == 40.0   # [08:10,08:20) 与 [08:30,09:00)
    assert b["cumulative_exposure"] == pytest.approx(20.0)  # (10-8)*10


def test_next_reading_before_staleness_caps_the_hold(client):
    # 间隔 15 分钟但下一条读数 5 分钟后到达：温度只保持 5 分钟
    post(client, "E1", "2026-10-01T08:00:00Z", 10.0)
    post(client, "E2", "2026-10-01T08:05:00Z", 10.0)
    b = assess(client, W0, W1, 15, 1000).json()
    # E1 盖 5 分钟，E2 盖 15 分钟（08:20 届满）
    assert b["observed_minutes"] == 20.0
    assert b["unknown_minutes"] == 40.0
    assert b["cumulative_exposure"] == pytest.approx(40.0)  # (10-8)*20


def test_last_reading_hold_extends_to_window_end(client):
    post(client, "E1", "2026-10-01T08:30:00Z", 9.0)
    b = assess(client, W0, W1, 60, 1000).json()
    # 窗前 30 分钟未知；[08:30,09:00) 已观测
    assert b["observed_minutes"] == 30.0
    assert b["unknown_minutes"] == 30.0
    assert b["cumulative_exposure"] == pytest.approx(30.0)  # (9-8)*30


def test_reading_at_window_end_does_not_cover(client):
    # 恰在右端点的读数不在窗内，其沿用段也只从右端点开始
    post(client, "E1", W1, 10.0)
    b = assess(client, W0, W1, 30, 1000).json()
    assert b["observed_minutes"] == 0.0
    assert b["unknown_minutes"] == 60.0
    assert b["cumulative_exposure"] == pytest.approx(0.0)
    assert b["conclusion"] == "insufficient_evidence"


def test_reading_at_window_start_is_included(client):
    post(client, "E1", W0, 10.0)
    b = assess(client, W0, W1, 60, 1000).json()
    assert b["observed_minutes"] == 60.0
    assert b["unknown_minutes"] == 0.0


def test_fully_known_window_with_no_exposure_passes(client):
    post(client, "E1", W0, 5.0)
    b = assess(client, W0, W1, 60, 0).json()
    assert b["observed_minutes"] == 60.0
    assert b["unknown_minutes"] == 0.0
    assert b["cumulative_exposure"] == pytest.approx(0.0)
    assert b["conclusion"] == "pass"


# ---------------------------------------------------------------- 结论判定

def test_exposure_over_budget_fails_even_with_unknown_time(client):
    # fail 优先于证据不足：即便窗内有未知时间，超预算即不合格
    post(client, "E1", W0, 10.0)
    b = assess(client, W0, W1, 30, 50).json()  # 暴露 (10-8)*30=60 > 50
    assert b["observed_minutes"] == 30.0
    assert b["unknown_minutes"] == 30.0
    assert b["cumulative_exposure"] == pytest.approx(60.0)
    assert b["conclusion"] == "fail"


def test_exposure_equal_to_budget_passes_when_fully_known(client):
    # 严格超过才算：等于预算 → 合格（全窗已知时）
    post(client, "E1", W0, 9.0)
    b = assess(client, W0, W1, 60, 60).json()  # (9-8)*60 = 60
    assert b["cumulative_exposure"] == pytest.approx(60.0)
    assert b["conclusion"] == "pass"


def test_under_budget_with_unknown_time_is_insufficient_evidence(client):
    post(client, "E1", W0, 9.0)
    b = assess(client, W0, W1, 30, 1000).json()
    assert b["cumulative_exposure"] == pytest.approx(30.0)
    assert b["unknown_minutes"] == 30.0
    assert b["conclusion"] == "insufficient_evidence"


def test_zero_budget_any_exposure_fails(client):
    post(client, "E1", W0, 8.1)
    b = assess(client, W0, W1, 60, 0).json()
    assert b["cumulative_exposure"] == pytest.approx(6.0)
    assert b["conclusion"] == "fail"


# ---------------------------------------------------------------- 读数规则

def test_latest_event_at_same_sampled_at_wins(client):
    # 同一采样时刻两条有效读数：事件号按字符串比较取最大
    # （"E9" > "E10"，因为第二位 '9' > '1'），E9 的 4.0 胜出
    post(client, "E10", "2026-10-01T08:00:00Z", 10.0)
    post(client, "E9", "2026-10-01T08:00:00Z", 4.0)
    b = assess(client, W0, W1, 60, 0).json()
    assert b["cumulative_exposure"] == pytest.approx(0.0)
    assert b["conclusion"] == "pass"


def test_latest_event_at_same_sampled_at_explicit_max(client):
    post(client, "EA", "2026-10-01T08:00:00Z", 4.0)
    post(client, "EB", "2026-10-01T08:00:00Z", 10.0)
    b = assess(client, W0, W1, 60, 0).json()
    assert b["cumulative_exposure"] == pytest.approx(120.0)  # EB 胜：(10-8)*60
    assert b["conclusion"] == "fail"


def test_revoked_readings_ignored(client):
    post(client, "E1", W0, 10.0)
    post(client, "E2", "2026-10-01T08:20:00Z", 10.0)
    assert client.delete(f"/devices/{DEVICE}/readings/E1").status_code == 200
    assert client.delete(f"/devices/{DEVICE}/readings/E2").status_code == 200
    b = assess(client, W0, W1, 60, 0).json()
    assert b["observed_minutes"] == 0.0
    assert b["unknown_minutes"] == 60.0
    assert b["cumulative_exposure"] == pytest.approx(0.0)
    assert b["conclusion"] == "insufficient_evidence"


def test_known_device_without_valid_readings_is_fully_unknown(client):
    # 已知设备（有过读数）但全部撤销 → 全窗未知，证据不足
    post(client, "E1", W0, 10.0)
    client.delete(f"/devices/{DEVICE}/readings/E1")
    b = assess(client, W0, W1, 60, 0).json()
    assert b["observed_minutes"] == 0.0
    assert b["unknown_minutes"] == 60.0
    assert b["conclusion"] == "insufficient_evidence"


# ---------------------------------------------------------------- 策略拆段

def test_policy_switches_split_segments(client):
    # 全窗温度恒定 10.0；全局 8.0 / P1=5.0 自 08:10 / P2=10.0 自 08:20
    post(client, "E1", W0, 10.0)
    assert put_policy(client, "P1", "2026-10-01T08:10:00Z", 5.0).status_code == 201
    assert put_policy(client, "P2", "2026-10-01T08:20:00Z", 10.0).status_code == 201
    b = assess(client, W0, "2026-10-01T08:30:00Z", 60, 0).json()
    # [08:00,08:10) (10-8)*10=20；[08:10,08:20) (10-5)*10=50；[08:20,08:30) 不超限
    assert b["observed_minutes"] == 30.0
    assert b["unknown_minutes"] == 0.0
    assert b["cumulative_exposure"] == pytest.approx(70.0)
    assert b["conclusion"] == "fail"


def test_policy_effective_exactly_at_segment_start_applies(client):
    # 策略生效时刻恰为评估窗 / 温度段起点：该点起适用新上限
    post(client, "E1", W0, 6.0)
    put_policy(client, "P1", W0, 5.0)  # 全局 8.0 下 6.0 不超限；P1 下超限 1.0
    b = assess(client, W0, W1, 60, 0).json()
    assert b["cumulative_exposure"] == pytest.approx(60.0)
    assert b["conclusion"] == "fail"


def test_pre_window_policy_applies_to_window_head(client):
    post(client, "E1", "2026-10-01T07:50:00Z", 9.0)
    put_policy(client, "P1", "2026-10-01T07:00:00Z", 6.0)  # 窗前生效
    b = assess(client, W0, "2026-10-01T08:10:00Z", 30, 0).json()
    # E1 覆盖 [08:00,08:10)：(9-6)*10 = 30
    assert b["cumulative_exposure"] == pytest.approx(30.0)


def test_policy_switch_inside_pre_window_hold(client):
    # 窗前读数覆盖窗头，窗内策略切换：按时间点适用，照样拆段
    post(client, "E1", "2026-10-01T07:55:00Z", 10.0)  # 盖到 08:10
    put_policy(client, "P1", "2026-10-01T08:05:00Z", 9.0)
    b = assess(client, W0, "2026-10-01T08:10:00Z", 15, 0).json()
    # [08:00,08:05) 全局 8：2*5=10；[08:05,08:10) P1 9：1*5=5
    assert b["cumulative_exposure"] == pytest.approx(15.0)


# ---------------------------------------------------------------- 设备 / 校验

def test_unknown_device_returns_404(client):
    r = assess(client, W0, W1, 15, 0, device="ghost")
    assert r.status_code == 404
    assert "ghost" in r.json()["detail"]


def test_unknown_device_checked_after_body_validation(client):
    # 请求体本身非法仍是 422（请求解析先于设备判定）
    r = client.post(
        "/devices/ghost/exposure-assessments",
        json={
            "window_start": W0,
            "window_end": W1,
            "max_staleness_minutes": 0,
            "exposure_budget": 0,
        },
    )
    assert r.status_code == 422


def test_non_positive_staleness_rejected(client):
    for bad in (0, -1, -0.5):
        r = assess(client, W0, W1, bad, 0)
        assert r.status_code == 422


def test_negative_budget_rejected(client):
    r = assess(client, W0, W1, 15, -0.01)
    assert r.status_code == 422


def test_non_finite_values_rejected(client):
    payload = (
        '{"window_start":"%s","window_end":"%s",'
        '"max_staleness_minutes":%s,"exposure_budget":%s}'
    )
    for stale, budget in (("NaN", "1"), ("Infinity", "1"), ("5", "NaN"),
                          ("5", "-Infinity"), ("-Infinity", "1")):
        r = client.post(
            f"/devices/{DEVICE}/exposure-assessments",
            content=payload % (W0, W1, stale, budget),
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 422, stale


def test_invalid_window_rejected(client):
    assert assess(client, "not-a-time", W1, 15, 0).status_code == 422
    assert assess(client, W0, "not-a-time", 15, 0).status_code == 422
    # 结束不晚于开始（含相等、跨时区相等与结束更早）
    assert assess(client, W0, W0, 15, 0).status_code == 422
    assert assess(
        client, "2026-10-01T16:00:00+08:00", "2026-10-01T08:00:00Z", 15, 0
    ).status_code == 422
    assert assess(
        client, "2026-10-01T09:00:00Z", "2026-10-01T08:00:00Z", 15, 0
    ).status_code == 422


def test_missing_fields_rejected(client):
    r = client.post(
        f"/devices/{DEVICE}/exposure-assessments",
        json={"window_start": W0, "window_end": W1},
    )
    assert r.status_code == 422


def test_equivalent_timezone_windows_give_identical_results(client):
    post(client, "E1", "2026-10-01T16:05:00+08:00", 10.0)
    a = assess(client, "2026-10-01T16:00:00+08:00", "2026-10-01T17:00:00+08:00",
               30, 0).json()
    b = assess(client, W0, W1, 30, 0).json()
    assert a == b
    assert a["window_start"] == "2026-10-01T08:00:00.000000Z"
    assert a["window_end"] == "2026-10-01T09:00:00.000000Z"


# ---------------------------------------------------------------- 只读 / 幂等

def test_assessment_does_not_mutate_anything(client):
    post(client, "E1", W0, 10.0)
    post(client, "E2", "2026-10-01T08:20:00Z", 5.0)
    put_policy(client, "P1", "2026-10-01T08:10:00Z", 6.0)
    # 先出一张交接凭证
    assert client.post(
        f"/devices/{DEVICE}/handover-vouchers",
        json={"handover_no": "H1", "window_start": W0, "window_end": W1},
    ).status_code == 201

    readings = client.get(f"/devices/{DEVICE}/readings").json()
    intervals = client.get(f"/devices/{DEVICE}/intervals").json()
    policies = client.get(f"/devices/{DEVICE}/limit-policies").json()
    vouchers = client.get(f"/devices/{DEVICE}/handover-vouchers").json()
    disputes = client.get(f"/devices/{DEVICE}/disputes").json()

    for _ in range(2):
        r = assess(client, W0, W1, 15, 40)
        assert r.status_code == 200

    assert client.get(f"/devices/{DEVICE}/readings").json() == readings
    assert client.get(f"/devices/{DEVICE}/intervals").json() == intervals
    assert client.get(f"/devices/{DEVICE}/limit-policies").json() == policies
    assert client.get(f"/devices/{DEVICE}/handover-vouchers").json() == vouchers
    assert client.get(f"/devices/{DEVICE}/disputes").json() == disputes


def test_assessment_reflects_latest_state(client):
    post(client, "E1", W0, 10.0)
    first = assess(client, W0, W1, 60, 0)
    assert first.json()["conclusion"] == "fail"
    # 更正为不超限后重新评估 → 合格（评估总是读最新状态，不冻结）
    client.put(
        f"/devices/{DEVICE}/readings/E1",
        json={"sampled_at": W0, "temperature": 5.0},
    )
    second = assess(client, W0, W1, 60, 0)
    assert second.json()["conclusion"] == "pass"
    assert second.json()["cumulative_exposure"] == pytest.approx(0.0)


def test_devices_are_isolated(client):
    post(client, "E1", W0, 10.0, device="dev-A")
    post(client, "E1", W0, 4.0, device="dev-B")
    a = assess(client, W0, W1, 60, 0, device="dev-A").json()
    b = assess(client, W0, W1, 60, 0, device="dev-B").json()
    assert a["conclusion"] == "fail"
    assert b["conclusion"] == "pass"


# ---------------------------------------------------------------- 纯函数：未舍入比较

def test_unrounded_cumulative_value_used_against_budget():
    """累计使用未舍入值：0.1℃ × 3 分钟在浮点下略大于 0.3，应判超预算。

    若实现先把累计暴露量（或每段结果）舍入再与预算比较，本例将得到
    0.3 == 0.3 → 不超预算，与“比较使用未舍入值”的口径相悖。
    """
    from app.exposure import assess_exposure

    readings = [
        {"event_no": "E1", "sampled_at": "2026-10-01T08:00:00.000000Z",
         "temperature": 0.1},
    ]
    result = assess_exposure(
        readings, [], 0.0,
        "2026-10-01T08:00:00.000000Z",
        "2026-10-01T08:03:00.000000Z",
        10, 0.3,
    )
    # 0.1 * 3 == 0.30000000000000004（未舍入），严格超过 0.3 → fail
    assert result.cumulative_exposure > 0.3
    assert result.conclusion == "fail"


def test_pure_function_dedupe_and_partial_coverage():
    from app.exposure import assess_exposure

    readings = [
        {"event_no": "E1", "sampled_at": "2026-10-01T07:59:00.000000Z",
         "temperature": 10.0},
        {"event_no": "E2", "sampled_at": "2026-10-01T08:10:00.000000Z",
         "temperature": 9.0},
        {"event_no": "E3", "sampled_at": "2026-10-01T08:10:00.000000Z",
         "temperature": 5.0},  # 与 E2 同时刻，事件号更大 → 胜出
    ]
    result = assess_exposure(
        readings, [], 8.0,
        "2026-10-01T08:00:00.000000Z",
        "2026-10-01T09:00:00.000000Z",
        15, 1000,
    )
    # E1 盖窗头 10 分钟（10.0，暴露 2*10=20）；E3 胜 E2，5.0 盖到 08:25
    assert result.observed_minutes == 25.0
    assert result.unknown_minutes == 35.0
    assert result.cumulative_exposure == pytest.approx(20.0)
    assert result.conclusion == "insufficient_evidence"
