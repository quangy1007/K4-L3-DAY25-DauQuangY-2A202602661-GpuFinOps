import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finops import pricing, metrics, sustainability


def test_extension1_advanced_recommend_tier():
    # Backward compatibility with baseline 2-arg signature
    assert pricing.recommend_tier(2, True) == "spot"
    assert pricing.recommend_tier(24, False) == "reserved"
    assert pricing.recommend_tier(4, False) == "on_demand"

    # Extension 1: Multi-factor evaluation with gpu_type and job_days
    # Long-running (>=365 days) continuous H100 job -> reserved
    assert pricing.recommend_tier(24, False, gpu_type="H100", job_days=365) == "reserved"

    # Interruptible H100 with low interruption rate (3%) -> spot
    assert pricing.recommend_tier(16, True, gpu_type="H100", job_days=30) == "spot"

    # Spiky non-interruptible workload (< break-even duty cycle) -> on_demand
    assert pricing.recommend_tier(4, False, gpu_type="A100", job_days=30) == "on_demand"


def test_extension2_mbu_rightsizing():
    # $/GB-VRAM-hr metric
    h100_vram_cost = metrics.dollars_per_gb_vram(2.50, 80)
    assert abs(h100_vram_cost - 0.03125) < 1e-4

    a100_vram_cost = metrics.dollars_per_gb_vram(1.79, 80)
    assert abs(a100_vram_cost - 0.022375) < 1e-4
    assert a100_vram_cost < h100_vram_cost

    # Right-sizing low MBU GPU
    summary = [{"gpu_id": "gpu-test-1", "gpu_type": "H100", "mbu": 0.20}]
    cat = {
        "H100": {"on_demand_hr": 2.50, "peak_bw_tbs": 3.35, "hbm_gb": 80},
        "A100": {"on_demand_hr": 1.79, "peak_bw_tbs": 2.00, "hbm_gb": 80},
        "L4": {"on_demand_hr": 0.80, "peak_bw_tbs": 0.30, "hbm_gb": 24},
    }
    recs = metrics.mbu_rightsize_analysis(summary, cat)
    assert len(recs) == 1
    assert recs[0]["recommended_type"] == "A100"
    assert recs[0]["savings_hr"] > 0
    assert recs[0]["monthly_savings"] > 0


def test_extension3_cache_economics():
    # Break-even reads formula: write_cost / (price_in * (1 - read_discount))
    price_in = 3.00
    write_fee = 1.25 * price_in  # 3.75
    read_discount = 0.10         # 90% off -> saves 2.70 per read

    be = pricing.cache_break_even_reads(write_fee, read_discount, price_in)
    assert abs(be - (1.25 / 0.90)) < 1e-4  # ~1.389 reads

    # Reading 1 time is NOT worth it (1 < 1.389)
    assert pricing.cache_is_worth_it(1.0, write_fee, read_discount, price_in) is False

    # Reading 3 times IS worth it (3 > 1.389)
    assert pricing.cache_is_worth_it(3.0, write_fee, read_discount, price_in) is True


def test_extension4_reasoning_budget():
    standard_wh = sustainability.wh_per_query(1000, is_reasoning=False)
    reasoning_wh = sustainability.wh_per_query(1000, is_reasoning=True)

    assert abs(standard_wh - 0.30) < 1e-4
    assert abs(reasoning_wh - (0.30 * 80.0)) < 1e-4
    assert reasoning_wh == 80.0 * standard_wh


def test_extension5_carbon_aware_scheduling():
    kwh = 1000.0  # 1 MWh
    regions = sustainability.compare_all_regions(kwh)
    assert len(regions) == 5

    reg_map = {r["region"]: r for r in regions}
    assert reg_map["europe-north1"]["carbon_kg"] == 30.0   # 30 g/kWh * 1000 kWh = 30 kg
    assert reg_map["us-east-1"]["carbon_kg"] == 380.0      # 380 g/kWh * 1000 kWh = 380 kg

    # Europe-North1 is >90% cleaner than us-east-1
    saved_carbon = reg_map["us-east-1"]["carbon_kg"] - reg_map["europe-north1"]["carbon_kg"]
    assert saved_carbon == 350.0
    assert (saved_carbon / reg_map["us-east-1"]["carbon_kg"]) > 0.90
