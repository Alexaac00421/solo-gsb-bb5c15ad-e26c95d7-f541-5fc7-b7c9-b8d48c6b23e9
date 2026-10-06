"""累计温度暴露评估端到端测试。

覆盖：时间窗左闭右开；读数温度自采样时刻沿用至下一条有效读数或最大沿用
间隔届满（取先到者）；窗前最近读数覆盖窗头；同一采样时刻以事件号最大者
为准（仅未撤销读数参与）；上限按时间点适用最近生效策略并在策略切换时
拆段，无策略沿用全局上限；逐段累计 max(温度-上限,0)×分钟；暴露量超过
预算判 fail，未超过但有未知时间判 insufficient_evidence，其余 pass；
已知设备无有效读数全窗未知；未知设备 404；非法时间窗 / 非有限数值 /
非正间隔 / 负预算 422；评估只读最新状态，不改区间、读数、策略与凭证。
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app

DEVICE = "truck-001"
W0 = "2026-10-01T08:00:00Z"
W1 = "2026-10-01T08:30:00Z"


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


def assess(client, start, end, interval, budget, device=DEVICE):
    return client.post(
        f"/devices/{device}/exposure-assessments",
        json={
            "window_start": start,
            "window_end": end,
            "max_interval_minutes": interval,
            "exposure_budget": budget,
        },
    )


# ---------------------------------------------------------------- 基本评估与结论

def test_full_coverage_pass_with_cumulative_exposure(client):
    # E1/E2/E3 首尾相接覆盖全窗；暴露 = (9-8)*10 + (9.5-8)*10 + 0 = 25
    post(client, "E1", "2026-10-01T08:00:00Z", 9.0)
    post(client, "E2", "2026-10-01T08:10:00Z", 9.5)
    post(client, "E3", "2026-10-01T08:20:00Z", 6.0)
    r = assess(client, W0, W1, 15, 30.0)
    assert r.status_code == 200
    body = r.json()
    assert body["device_id"] == DEVICE
    assert body["window_start"] == "2026-10-01T08:00:00.000000Z"
    assert body["window_end"] == "2026-10-01T08:30:00.000000Z"
    assert body["max_interval_minutes"] == 15.0
    assert body["exposure_budget"] == 30.0
    assert body["limit"] == 8.0
    assert body["observed_minutes"] == 30.0
    assert body["unknown_minutes"] == 0.0
    assert body["exposure"] == pytest.approx(25.0)
    assert body["verdict"] == "pass"


def test_carry_forward_expiry_leaves_unknown_tail(client):
    # 沿用间隔 5 分钟：每条读数只覆盖 5 分钟，其后至下一条之间记未知
    post(client, "E1", "2026-10-01T08:00:00Z", 9.0)
    post(client, "E2", "2026-10-01T08:10:00Z", 9.5)
    post(client, "E3", "2026-10-01T08:20:00Z", 6.0)
    r = assess(client, W0, W1, 5, 30.0)
    body = r.json()
    assert body["observed_minutes"] == 15.0
    assert body["unknown_minutes"] == 15.0
    assert body["exposure"] == pytest.approx(12.5)  # (9-8)*5 + (9.5-8)*5 + 0
    assert body["verdict"] == "insufficient_evidence"


def test_gap_between_readings_beyond_carry_is_unknown(client):
    # 读数间隔大于沿用间隔：中间空洞记未知
    post(client, "E1", "2026-10-01T08:00:00Z", 9.0)
    post(client, "E2", "2026-10-01T08:30:00Z", 9.0)
    r = assess(client, W0, "2026-10-01T08:40:00Z", 10, 100.0)
    body = r.json()
    assert body["observed_minutes"] == 20.0
    assert body["unknown_minutes"] == 20.0
    assert body["exposure"] == pytest.approx(20.0)
    assert body["verdict"] == "insufficient_evidence"


def test_exposure_over_budget_fails(client):
    post(client, "E1", "2026-10-01T08:00:00Z", 20.0)
    r = assess(client, W0, "2026-10-01T08:10:00Z", 10, 5.0)
    body = r.json()
    assert body["exposure"] == pytest.approx(120.0)  # (20-8)*10
    assert body["verdict"] == "fail"


def test_fail_takes_precedence_over_unknown_time(client):
    # 既有未知时间又超预算：超预算即不合格
    post(client, "E1", "2026-10-01T08:00:00Z", 20.0)
    r = assess(client, W0, "2026-10-01T08:10:00Z", 5, 5.0)
    body = r.json()
    assert body["observed_minutes"] == 5.0
    assert body["unknown_minutes"] == 5.0
    assert body["exposure"] == pytest.approx(60.0)
    assert body["verdict"] == "fail"


def test_exposure_equal_to_budget_is_not_exceeded(client):
    # 等于预算不算超过：无未知时间 → 合格
    post(client, "E1", "2026-10-01T08:00:00Z", 9.0)
    r = assess(client, W0, "2026-10-01T08:10:00Z", 10, 10.0)
    body = r.json()
    assert body["exposure"] == pytest.approx(10.0)
    assert body["verdict"] == "pass"


def test_zero_budget_with_zero_exposure_passes(client):
    post(client, "E1", "2026-10-01T08:00:00Z", 5.0)  # 低于上限，暴露为 0
    r = assess(client, W0, "2026-10-01T08:10:00Z", 10, 0.0)
    body = r.json()
    assert body["exposure"] == 0.0
    assert body["verdict"] == "pass"


# ---------------------------------------------------------------- 覆盖规则

def test_pre_window_reading_covers_window_head(client):
    # 窗前最近读数沿用覆盖窗头 [08:00, 08:05)，窗内 E1 接续覆盖
    post(client, "E0", "2026-10-01T07:55:00Z", 9.0)
    post(client, "E1", "2026-10-01T08:05:00Z", 9.0)
    r = assess(client, W0, "2026-10-01T08:10:00Z", 20, 100.0)
    body = r.json()
    assert body["observed_minutes"] == 10.0
    assert body["unknown_minutes"] == 0.0
    assert body["exposure"] == pytest.approx(10.0)
    assert body["verdict"] == "pass"


def test_pre_window_reading_expired_before_window_is_unknown(client):
    # 窗前读数的沿用间隔在窗开始前已届满：窗头未知
    post(client, "E0", "2026-10-01T07:30:00Z", 9.0)
    r = assess(client, W0, "2026-10-01T08:10:00Z", 15, 100.0)
    body = r.json()
    assert body["observed_minutes"] == 0.0
    assert body["unknown_minutes"] == 10.0
    assert body["exposure"] == 0.0
    assert body["verdict"] == "insufficient_evidence"


def test_window_is_half_open(client):
    # 恰在窗终点的读数不覆盖窗内；恰在窗起点的读数自起点覆盖
    post(client, "E1", W0, 9.0)
    post(client, "E2", W1, 20.0)
    r = assess(client, W0, W1, 15, 100.0)
    body = r.json()
    assert body["observed_minutes"] == 15.0   # E1 沿用 15 分钟
    assert body["unknown_minutes"] == 15.0
    assert body["exposure"] == pytest.approx(15.0)  # 不含 E2 的 20.0


def test_same_timestamp_keeps_max_event_no(client):
    # 同一采样时刻以事件号最大者为准：E2(5.0) 覆盖 E1(9.0)
    post(client, "E1", W0, 9.0)
    post(client, "E2", W0, 5.0)
    r = assess(client, W0, "2026-10-01T08:10:00Z", 10, 0.0)
    body = r.json()
    assert body["observed_minutes"] == 10.0
    assert body["exposure"] == 0.0   # max(5-8, 0)*10
    assert body["verdict"] == "pass"


def test_same_timestamp_max_event_no_revoked_falls_back(client):
    # 事件号最大者被撤销后，同时刻未撤销的 E1 生效
    post(client, "E1", W0, 9.0)
    post(client, "E2", W0, 5.0)
    assert client.delete(f"/devices/{DEVICE}/readings/E2").status_code == 200
    r = assess(client, W0, "2026-10-01T08:10:00Z", 10, 0.0)
    body = r.json()
    assert body["exposure"] == pytest.approx(10.0)  # (9-8)*10
    assert body["verdict"] == "fail"


def test_fractional_minutes(client):
    # 秒级窗口：总 0.5 分钟，全部覆盖，暴露 0.5 恰好等于预算 → 合格
    post(client, "E1", W0, 9.0)
    r = assess(client, W0, "2026-10-01T08:00:30Z", 1, 0.5)
    body = r.json()
    assert body["observed_minutes"] == pytest.approx(0.5)
    assert body["unknown_minutes"] == 0.0
    assert body["exposure"] == pytest.approx(0.5)
    assert body["verdict"] == "pass"


# ---------------------------------------------------------------- 策略拆段

def test_policy_switch_splits_observed_segment(client):
    # 全局 8.0；P1 自 08:05 起上限 5.0：读数 7.0 的覆盖段在 08:05 拆段
    post(client, "E1", W0, 7.0)
    assert put_policy(client, "P1", "2026-10-01T08:05:00Z", 5.0).status_code == 201
    r = assess(client, W0, "2026-10-01T08:10:00Z", 15, 100.0)
    body = r.json()
    assert body["observed_minutes"] == 10.0
    # [08:00,08:05) 全局 8.0 → 0；[08:05,08:10) 策略 5.0 → (7-5)*5
    assert body["exposure"] == pytest.approx(10.0)
    assert body["verdict"] == "pass"


def test_multiple_policies_split_into_multiple_segments(client):
    # P1 自 08:03 起 5.0，P2 自 08:07 起 10.0；读数 8.0 覆盖全窗
    post(client, "E1", W0, 8.0)
    assert put_policy(client, "P1", "2026-10-01T08:03:00Z", 5.0).status_code == 201
    assert put_policy(client, "P2", "2026-10-01T08:07:00Z", 10.0).status_code == 201
    r = assess(client, W0, "2026-10-01T08:10:00Z", 15, 100.0)
    body = r.json()
    # [08:00,08:03) 全局 8.0 → 0；[08:03,08:07) 5.0 → 12；[08:07,08:10) 10.0 → 0
    assert body["exposure"] == pytest.approx(12.0)


def test_policy_effective_before_window_applies_at_window_head(client):
    # 窗前已生效的策略覆盖整个窗口
    post(client, "E1", W0, 7.0)
    assert put_policy(client, "P1", "2026-10-01T07:00:00Z", 6.0).status_code == 201
    r = assess(client, W0, "2026-10-01T08:10:00Z", 15, 100.0)
    body = r.json()
    assert body["exposure"] == pytest.approx(10.0)  # (7-6)*10


def test_no_policy_uses_global_limit(client):
    post(client, "E1", W0, 9.0)
    r = assess(client, W0, "2026-10-01T08:10:00Z", 10, 100.0)
    assert r.json()["exposure"] == pytest.approx(10.0)  # (9-8)*10，全局上限 8.0


# ---------------------------------------------------------------- 设备与读数状态

def test_known_device_without_valid_readings_is_all_unknown(client):
    # 已知设备（有读数记录）但全部撤销：全窗未知
    post(client, "E1", W0, 9.0)
    assert client.delete(f"/devices/{DEVICE}/readings/E1").status_code == 200
    r = assess(client, W0, W1, 15, 100.0)
    assert r.status_code == 200
    body = r.json()
    assert body["observed_minutes"] == 0.0
    assert body["unknown_minutes"] == 30.0
    assert body["exposure"] == 0.0
    assert body["verdict"] == "insufficient_evidence"


def test_unknown_device_returns_404(client):
    r = assess(client, W0, W1, 15, 100.0, device="ghost")
    assert r.status_code == 404
    assert "ghost" in r.json()["detail"]


def test_assessments_isolated_per_device(client):
    post(client, "E1", W0, 9.0, device="dev-A")
    post(client, "E1", W0, 1.0, device="dev-B")
    a = assess(client, W0, "2026-10-01T08:10:00Z", 10, 100.0, device="dev-A")
    b = assess(client, W0, "2026-10-01T08:10:00Z", 10, 100.0, device="dev-B")
    assert a.json()["exposure"] == pytest.approx(10.0)
    assert b.json()["exposure"] == 0.0


# ---------------------------------------------------------------- 输入校验

def test_invalid_window_rejected(client):
    post(client, "E1", W0, 9.0)
    # 结束等于开始 / 结束早于开始（含等价时区）
    assert assess(client, W0, W0, 15, 100.0).status_code == 422
    assert assess(client, W1, W0, 15, 100.0).status_code == 422
    assert (
        assess(client, "2026-10-01T16:00:00+08:00", W0, 15, 100.0).status_code
        == 422
    )
    # 非法时间格式
    assert assess(client, "not-a-time", W1, 15, 100.0).status_code == 422


def test_non_positive_interval_rejected(client):
    post(client, "E1", W0, 9.0)
    assert assess(client, W0, W1, 0, 100.0).status_code == 422
    assert assess(client, W0, W1, -5, 100.0).status_code == 422


def test_negative_budget_rejected(client):
    post(client, "E1", W0, 9.0)
    assert assess(client, W0, W1, 15, -0.5).status_code == 422


def test_non_finite_values_rejected(client):
    post(client, "E1", W0, 9.0)

    def raw(body: str):
        # httpx 的 json= 序列化器拒绝 NaN/Infinity，直接发送原始 JSON 文本
        return client.post(
            f"/devices/{DEVICE}/exposure-assessments",
            content=body,
            headers={"Content-Type": "application/json"},
        )

    base = '"window_start":"%s","window_end":"%s"' % (W0, W1)
    assert raw('{%s,"max_interval_minutes":NaN,"exposure_budget":100.0}' % base).status_code == 422
    assert raw('{%s,"max_interval_minutes":Infinity,"exposure_budget":100.0}' % base).status_code == 422
    assert raw('{%s,"max_interval_minutes":15,"exposure_budget":NaN}' % base).status_code == 422
    assert raw('{%s,"max_interval_minutes":15,"exposure_budget":-Infinity}' % base).status_code == 422


def test_missing_fields_rejected(client):
    post(client, "E1", W0, 9.0)
    r = client.post(
        f"/devices/{DEVICE}/exposure-assessments",
        json={"window_start": W0, "window_end": W1},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------- 时区与最新状态

def test_equivalent_timezone_window_normalized(client):
    post(client, "E1", W0, 9.0)
    r = assess(
        client,
        "2026-10-01T16:00:00+08:00",
        "2026-10-01T16:10:00+08:00",
        10,
        100.0,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["window_start"] == "2026-10-01T08:00:00.000000Z"
    assert body["window_end"] == "2026-10-01T08:10:00.000000Z"
    assert body["exposure"] == pytest.approx(10.0)


def test_assessment_reads_latest_state(client):
    post(client, "E1", W0, 9.0)
    first = assess(client, W0, W1, 15, 100.0).json()
    assert first["unknown_minutes"] == 15.0

    # 补报后续读数与策略后，同一评估请求按最新状态给出不同结果
    post(client, "E2", "2026-10-01T08:15:00Z", 9.0)
    assert put_policy(client, "P1", W0, 8.5).status_code == 201
    second = assess(client, W0, W1, 15, 100.0).json()
    assert second["unknown_minutes"] == 0.0
    assert second["exposure"] == pytest.approx(15.0)  # (9-8.5)*15 * 2


def test_assessment_does_not_change_existing_data(client):
    post(client, "E1", "2026-10-01T08:00:00Z", 9.0)
    post(client, "E2", "2026-10-01T08:10:00Z", 9.5)
    post(client, "E3", "2026-10-01T08:20:00Z", 6.0)
    assert put_policy(client, "P1", "2026-10-01T08:05:00Z", 5.0).status_code == 201
    voucher = client.post(
        f"/devices/{DEVICE}/handover-vouchers",
        json={"handover_no": "H1", "window_start": W0, "window_end": W1},
    )
    assert voucher.status_code == 201

    intervals_before = client.get(f"/devices/{DEVICE}/intervals").json()
    readings_before = client.get(f"/devices/{DEVICE}/readings").json()
    policies_before = client.get(f"/devices/{DEVICE}/limit-policies").json()

    r = assess(client, W0, W1, 15, 100.0)
    assert r.status_code == 200

    # 评估为只读：区间、读数、策略与已存凭证均保持不变
    assert client.get(f"/devices/{DEVICE}/intervals").json() == intervals_before
    assert client.get(f"/devices/{DEVICE}/readings").json() == readings_before
    assert client.get(f"/devices/{DEVICE}/limit-policies").json() == policies_before
    assert (
        client.get(f"/devices/{DEVICE}/handover-vouchers/H1").json()
        == voucher.json()
    )
