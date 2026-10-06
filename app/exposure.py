"""按设备与时间窗评估累计温度暴露（纯函数，便于单测）。

规则：
- 输入为该设备全部未撤销读数（按 (采样时间, 事件号) 升序查询得到）与全部
  温度上限策略（按生效时间升序）；
- 同一采样时刻存在多条有效读数时，仅保留**事件号最大**者（事件号按字符串
  比较，即“事件号最大”的口径与提交 / 凭证接口的排序口径一致）；
- 每条读数的温度自采样时刻起保持，至下一条有效读数的采样时刻或自身采样
  时刻 + 最大沿用间隔届满为止（先到为准），形成左闭右开的已知温度段；
- 评估时间窗为左闭右开 [window_start, window_end)：窗前最近一条读数的
  已知段同样可覆盖窗头，超出时间窗的部分裁掉；窗内未被任何已知温度段
  覆盖的时间记为未知；
- 温度上限按时间点适用“生效时间不晚于该时刻的最近一条策略”，无策略
  覆盖时沿用全局上限；策略切换点把已知温度段拆成若干小段，逐段判定；
- 逐小段累计 max(温度 - 上限, 0) × 分钟数，全程使用未舍入的浮点值；
- 结论：累计暴露量**严格超过**预算 → ``fail``；未超过但存在未知分钟 →
  ``insufficient_evidence``；其余（未超过且全窗已知）→ ``pass``。
"""
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Sequence

# 内部统一用整数微秒记账，避免浮点拼接时长出现负零 / 一微秒误差
_MICROSECOND = timedelta(microseconds=1)
_US_PER_MINUTE = 60 * 1_000_000


def parse_utc(value: str) -> datetime:
    """把 to_utc_iso 产出的固定宽度 UTC ISO 字符串解析为 aware datetime。"""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _as_utc(value) -> datetime:
    """接受 aware datetime（直接使用）或 UTC ISO 字符串。"""
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return parse_utc(value)


def dedupe_latest_events(readings: Sequence[dict]) -> list[dict]:
    """同一采样时刻仅保留事件号最大者，并按采样时间升序返回。

    输入读数为 dict（至少含 event_no / sampled_at / temperature）；
    事件号按字符串比较取最大（“同一采样时刻以事件号最大者为准”）。
    """
    latest: dict[datetime, dict] = {}
    for r in readings:
        ts = _as_utc(r["sampled_at"])
        current = latest.get(ts)
        if current is None or r["event_no"] > current["event_no"]:
            latest[ts] = {
                "event_no": r["event_no"],
                "sampled_at": ts,
                "temperature": r["temperature"],
            }
    return [latest[ts] for ts in sorted(latest)]


@dataclass
class ExposureResult:
    """累计温度暴露评估结果（分钟数与暴露量均为未舍入浮点值）。"""

    window_minutes: float
    observed_minutes: float
    unknown_minutes: float
    cumulative_exposure: float  # ℃·分钟
    conclusion: str  # pass | insufficient_evidence | fail


def assess_exposure(
    readings: Sequence[dict],
    policies: Sequence[tuple],
    default_limit: float,
    window_start,
    window_end,
    max_staleness_minutes: float,
    exposure_budget: float,
) -> ExposureResult:
    """评估 [window_start, window_end) 内的累计温度暴露。

    readings 为该设备全部未撤销读数（dict：event_no / sampled_at / temperature，
    sampled_at 为 UTC ISO 字符串或 aware datetime）；
    policies 为 (生效时间, 上限[, 策略号]) 三元组列表，按生效时间升序；
    max_staleness_minutes 为正数，exposure_budget 为非负数。
    """
    window_start = _as_utc(window_start)
    window_end = _as_utc(window_end)
    window_us = (window_end - window_start) // _MICROSECOND

    events = dedupe_latest_events(readings)
    staleness = timedelta(minutes=max_staleness_minutes)

    # 已知温度段（左闭右开，按起点升序、互不重叠）：
    # 温度从采样时刻保持到“下一条有效读数”与“沿用间隔届满”先到者。
    segments: list[tuple[datetime, datetime, float]] = []
    for i, ev in enumerate(events):
        seg_start = ev["sampled_at"]
        stale_end = seg_start + staleness
        if i + 1 < len(events):
            seg_end = min(events[i + 1]["sampled_at"], stale_end)
        else:
            seg_end = stale_end
        if seg_end <= seg_start:
            continue
        # 裁到时间窗 [window_start, window_end)，窗外部分不参与评估；
        # 窗前最近读数的段在此自然覆盖窗头。
        lo = max(seg_start, window_start)
        hi = min(seg_end, window_end)
        if hi > lo:
            segments.append((lo, hi, ev["temperature"]))

    # 策略切换点（aware datetime）与对应上限；无策略时全部沿用全局上限。
    policy_times = [_as_utc(p[0]) for p in policies]
    policy_limits = [p[1] for p in policies]

    def limit_at(ts: datetime) -> float:
        idx = bisect_right(policy_times, ts) - 1
        return policy_limits[idx] if idx >= 0 else default_limit

    exposure = 0.0
    known_us = 0
    for seg_lo, seg_hi, temperature in segments:
        known_us += (seg_hi - seg_lo) // _MICROSECOND
        # 按段内的策略生效边界拆小段：每个边界 b（lo < b < hi）切一刀，
        # 各小段内上限恒定。
        start_idx = bisect_right(policy_times, seg_lo)
        end_idx = bisect_right(policy_times, seg_hi)
        cuts = [seg_lo]
        for j in range(start_idx, end_idx):
            b = policy_times[j]
            if seg_lo < b < seg_hi:
                cuts.append(b)
        cuts.append(seg_hi)
        for a, b in zip(cuts, cuts[1:]):
            # 整数微秒 → 分钟：整分钟换算在浮点下精确
            minutes = ((b - a) // _MICROSECOND) / _US_PER_MINUTE
            limit = limit_at(a)  # 左端点适用的上限在 [a, b) 内恒定
            if temperature > limit:
                # 逐段累加、不做任何舍入；预算比较也用该未舍入值
                exposure += (temperature - limit) * minutes

    observed_minutes = known_us / _US_PER_MINUTE
    unknown_us = window_us - known_us
    unknown_minutes = unknown_us / _US_PER_MINUTE

    if exposure > exposure_budget:
        conclusion = "fail"
    elif unknown_us > 0:
        conclusion = "insufficient_evidence"
    else:
        conclusion = "pass"

    return ExposureResult(
        window_minutes=window_us / _US_PER_MINUTE,
        observed_minutes=observed_minutes,
        unknown_minutes=unknown_minutes,
        cumulative_exposure=exposure,
        conclusion=conclusion,
    )
