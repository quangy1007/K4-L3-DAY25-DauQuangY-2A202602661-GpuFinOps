"""Efficiency metrics — the numbers that actually drive GPU cost.

Key teaching point (deck §5): nvidia-smi "GPU-Util %" is a *time-active* clock,
not an efficiency metric. A GPU can read 100% util while its MFU is ~20% — you
are paying the full GPU-hour for a fraction of the FLOPs you rented.
"""
from __future__ import annotations


def compute_mfu(achieved_tflops: float, peak_tflops: float) -> float:
    """Model FLOPs Utilization = achieved / peak (clamped to 0..1).

    Good training MFU is ~0.35-0.45; >0.50 is excellent. Returns 0 if peak<=0.
    """
    if peak_tflops <= 0:
        return 0.0
    return max(0.0, min(1.0, achieved_tflops / peak_tflops))


def compute_mbu(achieved_bw_tbs: float, peak_bw_tbs: float) -> float:
    """Model Bandwidth Utilization = achieved HBM BW / peak BW (clamped 0..1).

    The right metric for memory-bound decode; target ~0.60 on H100-80GB batch-1.
    """
    if peak_bw_tbs <= 0:
        return 0.0
    return max(0.0, min(1.0, achieved_bw_tbs / peak_bw_tbs))


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    """FLOP / byte for a workload (the x-axis of the roofline model)."""
    if bytes_moved <= 0:
        return 0.0
    return flops / bytes_moved


def roofline_regime(intensity: float, ridge_point: float) -> str:
    """Below the ridge point a workload is memory-bound; at/above it is compute-bound.

    H100 ridge ~295 FLOP/byte (BF16). LLM decode (~1-2) is memory-bound; prefill
    (~455) is compute-bound — which is *why* prefill/decode disaggregation pays off.
    """
    return "compute-bound" if intensity >= ridge_point else "memory-bound"


def flag_util_lies(rows, util_threshold: float = 0.90, mfu_threshold: float = 0.30):
    """Return the rows where GPU-Util is high but MFU is low — money leaking.

    `rows` is an iterable of dicts each having 'gpu_util_pct' (0-100) and 'mfu' (0-1).
    These are GPUs you are billed full-rate for while they do little real compute.
    """
    out = []
    for r in rows:
        util = float(r.get("gpu_util_pct", 0)) / 100.0
        mfu = float(r.get("mfu", 0))
        if util >= util_threshold and mfu < mfu_threshold:
            out.append(r)
    return out


def idle_waste_usd(idle_hours: float, on_demand_hr: float) -> float:
    """Dollars burned by a GPU left running idle (training done, instance up)."""
    return max(0.0, idle_hours) * max(0.0, on_demand_hr)


def dollars_per_gb_vram(on_demand_hr: float, hbm_gb: float) -> float:
    """Calculate the $/GB-VRAM-hr metric to evaluate memory cost efficiency.

    For memory-bound LLM decode workloads, cost is dominated by VRAM footprint
    and bandwidth, not raw FP16 FLOPs.
    """
    if hbm_gb <= 0:
        return 0.0
    return on_demand_hr / hbm_gb


def mbu_rightsize_analysis(gpu_summary: list[dict], catalog: dict) -> list[dict]:
    """Identify memory-bound GPUs with underutilized bandwidth and recommend cost-effective replacements.

    Compares $/GB-VRAM, required bandwidth (achieved_bw_tbs), and recommends right-sized GPUs.
    """
    recommendations = []
    for g in gpu_summary:
        cur_type = g["gpu_type"]
        cur_cat = catalog.get(cur_type, {})
        cur_od = float(cur_cat.get("on_demand_hr", 0))
        cur_bw = float(cur_cat.get("peak_bw_tbs", 1.0))
        cur_vram = float(cur_cat.get("hbm_gb", 1.0))
        mbu = g.get("mbu", 0.0)
        achieved_bw = mbu * cur_bw

        # If MBU is low (< 0.35) and hourly price is high (>= $1.50)
        if mbu < 0.35 and cur_od >= 1.50:
            best_alt = None
            best_savings = 0.0
            for alt_type, alt_cat in catalog.items():
                alt_od = float(alt_cat.get("on_demand_hr", 0))
                alt_bw = float(alt_cat.get("peak_bw_tbs", 0))
                alt_vram = float(alt_cat.get("hbm_gb", 0))
                # Must satisfy achieved bandwidth requirement and have sufficient VRAM
                if alt_od < cur_od and alt_bw >= achieved_bw and alt_vram >= min(cur_vram * 0.5, 24):
                    savings_hr = cur_od - alt_od
                    if savings_hr > best_savings:
                        best_savings = savings_hr
                        best_alt = {
                            "current_gpu": g["gpu_id"],
                            "current_type": cur_type,
                            "current_od_hr": cur_od,
                            "current_mbu": mbu,
                            "achieved_bw_tbs": round(achieved_bw, 3),
                            "recommended_type": alt_type,
                            "recommended_od_hr": alt_od,
                            "recommended_bw_tbs": alt_bw,
                            "savings_hr": round(savings_hr, 2),
                            "savings_pct": round((savings_hr / cur_od) * 100, 1),
                            "monthly_savings": round(savings_hr * 24 * 30, 2),
                            "reason": f"Achieved BW ({achieved_bw:.2f} TB/s) fits on {alt_type} ({alt_bw:.2f} TB/s peak) while cutting hourly rate from ${cur_od} to ${alt_od}",
                        }
            if best_alt:
                recommendations.append(best_alt)
    return recommendations

