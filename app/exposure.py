"""时间窗累计温度暴露评估（纯函数，便于单测）。

规则：
- 评估窗为左闭右开 [window_start, window_end)；
- 仅未撤销读数参与；同一采样时刻以事件号最大者为准；
- 每条读数的温度自采样时刻保持，至下一条有效读数或最大沿用间隔届满
  （取先到者）；窗前最近读数同样可覆盖窗头；未被覆盖的时间记为未知；
- 温度上限按时间点适用最近生效策略（生效时间不晚于该时刻的最近一条），
  策略切换时刻拆段，无策略覆盖的时间沿用全局上限；
- 逐段累计 max(温度 - 上限, 0) × 分钟；暴露量严格大于预算判不合格，
  未超过预算但存在未知时间判证据不足，其余判合格。
- 时间轴一律用整数微秒计算（datetime 差值精确），观测 / 未知分钟与
  暴露量仅在最终展示时换算为分钟；暴露量从不舍入，直接用于预算比较。
"""
from bisect import bisect_right
from datetime import datetime, timedelta
from typing import List, Sequence, Tuple

# 每分钟的微秒数（整数换算，避免浮点累计误差影响“有无未知时间”的判定）
_US_PER_MINUTE = 60_000_000


def _parse(ts: str) -> datetime:
    """解析 UTC 固定宽度 ISO 字符串（util.to_utc_iso 的产物）。"""
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _microseconds(delta: timedelta) -> int:
    """timedelta 精确换算为整数微秒。"""
    return delta // timedelta(microseconds=1)


def assess_exposure(
    readings: List[dict],
    policies: Sequence[Tuple[str, float]],
    default_limit: float,
    window_start: str,
    window_end: str,
    max_interval_minutes: float,
    budget: float,
) -> dict:
    """评估设备在时间窗内的累计温度暴露。

    readings 为未撤销读数，已按 (采样时间, 事件号) 升序；
    policies 为 (生效时间, 上限) 列表，按生效时间升序（可带第三位策略号，
    本函数只取前两位）；window_start / window_end 为 UTC 固定宽度 ISO 字符串；
    max_interval_minutes 为最大读数沿用间隔（正数分钟）；
    budget 为允许暴露量（非负 ℃·分钟）。

    返回 observed_minutes / unknown_minutes / exposure / verdict。
    """
    ws = _parse(window_start)
    we = _parse(window_end)
    carry = timedelta(minutes=max_interval_minutes)

    # 同一采样时刻只保留事件号最大者：输入已按 (采样时间, 事件号) 升序，
    # 同一时刻的后一条即事件号更大者，直接覆盖前一条
    deduped: List[dict] = []
    for r in readings:
        if deduped and deduped[-1]["sampled_at"] == r["sampled_at"]:
            deduped[-1] = r
        else:
            deduped.append(r)

    # 每条读数的覆盖区间 [t, min(t + 沿用间隔, 下一条采样时刻))，与评估窗求交；
    # 窗前最近读数的覆盖区间同样与窗头相交，天然覆盖窗头
    observed: List[Tuple[datetime, datetime, float]] = []
    for i, r in enumerate(deduped):
        t = _parse(r["sampled_at"])
        end = t + carry
        if i + 1 < len(deduped):
            nxt = _parse(deduped[i + 1]["sampled_at"])
            if nxt < end:
                end = nxt
        a = max(t, ws)
        b = min(end, we)
        if a < b:
            observed.append((a, b, r["temperature"]))

    total_us = _microseconds(we - ws)
    observed_us = sum(_microseconds(b - a) for a, b, _ in observed)
    unknown_us = total_us - observed_us

    # 温度上限按时间点适用最近生效策略：观测段在策略切换时刻拆段，
    # 无策略覆盖的时段沿用全局上限
    policy_points = [(_parse(p[0]), p[1]) for p in policies]
    policy_times = [pt for pt, _ in policy_points]

    exposure = 0.0
    for a, b, temp in observed:
        cuts = [a] + [s for s, _ in policy_points if a < s < b] + [b]
        for k in range(len(cuts) - 1):
            seg_start, seg_end = cuts[k], cuts[k + 1]
            idx = bisect_right(policy_times, seg_start) - 1
            limit = policy_points[idx][1] if idx >= 0 else default_limit
            minutes = _microseconds(seg_end - seg_start) / _US_PER_MINUTE
            exposure += max(temp - limit, 0.0) * minutes

    # 暴露量严格大于预算判不合格（比较使用未舍入值）；
    # 未超过预算但存在未知时间判证据不足；其余合格
    if exposure > budget:
        verdict = "fail"
    elif unknown_us > 0:
        verdict = "insufficient_evidence"
    else:
        verdict = "pass"

    return {
        "observed_minutes": observed_us / _US_PER_MINUTE,
        "unknown_minutes": unknown_us / _US_PER_MINUTE,
        "exposure": exposure,
        "verdict": verdict,
    }
