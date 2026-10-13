"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing, sustainability

DAYS = 30


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = 0.0
    recs = []

    # Extension 1 tracking
    advanced_recs = []
    adv_optimized_monthly = 0.0

    # Extension 5 tracking: Carbon calculation for interruptible jobs
    total_interruptible_kwh = 0.0

    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        job_days = int(num(j.get("days", 30)))
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        watts = num(c.get("watts", 500))
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        # BASELINE tier recommendation
        tier = pricing.recommend_tier(hpd, interruptible)
        if tier == "spot":
            sim = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)
            opt_cost = sim["spot_cost"]
        elif tier == "reserved":
            opt_cost = gpu_hours * num(c["reserved_3yr_hr"])
        else:
            opt_cost = on_demand_cost

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost)})

        # ADVANCED tier recommendation (Extension 1)
        adv_tier = pricing.recommend_tier(hpd, interruptible, gpu_type=gtype, job_days=job_days)
        if adv_tier == "spot":
            sim_adv = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)
            adv_cost = sim_adv["spot_cost"]
        elif adv_tier == "reserved":
            adv_rate = num(c["reserved_3yr_hr"]) if job_days >= 365 else num(c.get("reserved_1yr_hr", c["reserved_3yr_hr"]))
            adv_cost = gpu_hours * adv_rate
        else:
            adv_cost = on_demand_cost
        adv_optimized_monthly += adv_cost
        advanced_recs.append({"job_id": j["job_id"], "gpu_type": gtype, "adv_tier": adv_tier,
                              "adv_cost": round(adv_cost)})

        # Extension 5: Power consumption of interruptible workloads
        if interruptible:
            total_interruptible_kwh += (gpu_hours * watts) / 1000.0

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0

    adv_savings = on_demand_monthly - adv_optimized_monthly
    adv_savings_pct = adv_savings / on_demand_monthly * 100 if on_demand_monthly else 0.0

    # Extension 5: Multi-region carbon economics
    region_rows = sustainability.compare_all_regions(total_interruptible_kwh)
    carbon_us_east = sustainability.carbon_g(total_interruptible_kwh * 1000.0, "us-east-1") / 1000.0  # kg
    carbon_norway = sustainability.carbon_g(total_interruptible_kwh * 1000.0, "europe-north1") / 1000.0  # kg
    carbon_saved_kg = carbon_us_east - carbon_norway
    carbon_reduction_pct = (carbon_saved_kg / carbon_us_east * 100.0) if carbon_us_east > 0 else 0.0

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")

        print("\n== Extension 1: Advanced Tier Policy Comparison ==")
        print(f"Baseline Policy Monthly:  ${optimized_monthly:,.0f} ({savings_pct:.1f}% saved)")
        print(f"Advanced Policy Monthly:  ${adv_optimized_monthly:,.0f} ({adv_savings_pct:.1f}% saved)")
        print("Tier Recommendation Decision Matrix:")
        print("  * Interruptible=1 & Low Interruption Risk (H100/A100) -> 'spot' with checkpointing")
        print("  * High Duty Cycle (>=55%) & Long Duration (>=1 yr)   -> 'reserved' (3yr/1yr)")
        print("  * Spiky / Short-lived non-interruptible               -> 'on_demand'")

        print("\n== Extension 5: Carbon-Aware Workload Scheduling ==")
        print(f"Interruptible Workloads Total Energy: {total_interruptible_kwh:,.1f} kWh/month")
        print(f"{'Region':18}{'Power $/kWh':>13}{'gCO2/kWh':>12}{'Power Cost':>14}{'Carbon (kg CO2e)':>18}")
        for reg in region_rows:
            print(f"{reg['region']:18}${reg['price_per_kwh']:>12.3f}{reg['carbon_g_per_kwh']:>12}${reg['electricity_cost_usd']:>13,.2f}{reg['carbon_kg']:>17,.1f} kg")
        print(f"\n  Green Migration Impact: Moving interruptible jobs from us-east-1 to europe-north1")
        print(f"  cuts carbon by {carbon_saved_kg:,.1f} kg CO2e/month (-{carbon_reduction_pct:.1f}% carbon reduction)!")

    return {
        "recommendations": recs,
        "on_demand_monthly": round(on_demand_monthly),
        "optimized_monthly": round(optimized_monthly),
        "savings_pct": round(savings_pct, 1),
        "advanced_recommendations": advanced_recs,
        "adv_optimized_monthly": round(adv_optimized_monthly),
        "carbon_savings_kg": round(carbon_saved_kg, 1),
    }


if __name__ == "__main__":
    run()

