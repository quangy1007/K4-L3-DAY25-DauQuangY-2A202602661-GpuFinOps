"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0

    # Tracking for Extension 3 (Prompt Caching Economics)
    cached_requests_count = 0
    total_cached_tokens = 0

    # Tracking for Extension 4 (Reasoning Budget)
    reasoning_reqs = 0
    non_reasoning_reqs = 0
    reasoning_tokens = 0
    non_reasoning_tokens = 0
    reasoning_opt_cost = 0.0
    non_reasoning_opt_cost = 0.0
    reasoning_wh = 0.0
    non_reasoning_wh = 0.0

    # Extension 3: Check cache viability upfront
    # Assume typical shared system prompt prefix is read ~4.5 times on average across sessions
    avg_prefix_reads = 4.5
    small_be_reads = pricing.cache_break_even_reads(write_cost_per_m=1.25 * MODEL_PRICES["small"][0],
                                                    read_discount=0.10, price_in_per_m=MODEL_PRICES["small"][0])
    large_be_reads = pricing.cache_break_even_reads(write_cost_per_m=1.25 * MODEL_PRICES["large"][0],
                                                    read_discount=0.10, price_in_per_m=MODEL_PRICES["large"][0])
    cache_worth_small = pricing.cache_is_worth_it(avg_prefix_reads, write_cost_per_m=1.25 * MODEL_PRICES["small"][0],
                                                  read_discount=0.10, price_in_per_m=MODEL_PRICES["small"][0])
    cache_worth_large = pricing.cache_is_worth_it(avg_prefix_reads, write_cost_per_m=1.25 * MODEL_PRICES["large"][0],
                                                  read_discount=0.10, price_in_per_m=MODEL_PRICES["large"][0])

    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        is_reason = bool(int(num(r["is_reasoning"])))
        toks = inp + out
        total_tokens += toks

        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)

        # OPTIMIZED: cascade (route_tier), prompt caching (if worth it), batch API
        tier = r["route_tier"]
        pin, pout = MODEL_PRICES[tier]
        effective_cached = cached if (cache_worth_small if tier == "small" else cache_worth_large) else 0
        req_opt = pricing.request_cost(inp, out, pin, pout, cached_in=effective_cached, batch=is_batch)
        opt_cost += req_opt

        if cached > 0:
            cached_requests_count += 1
            total_cached_tokens += cached

        # Extension 4: Reasoning breakdown
        wh = sustainability.wh_per_query(toks, is_reasoning=is_reason)
        if is_reason:
            reasoning_reqs += 1
            reasoning_tokens += toks
            reasoning_opt_cost += req_opt
            reasoning_wh += wh
        else:
            non_reasoning_reqs += 1
            non_reasoning_tokens += toks
            non_reasoning_opt_cost += req_opt
            non_reasoning_wh += wh

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    total_wh = reasoning_wh + non_reasoning_wh

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")

        print("\n== Extension 3: Prompt Caching Economics (cache_is_worth_it) ==")
        print(f"  * Small tier break-even read count: {small_be_reads:.2f} reads")
        print(f"  * Large tier break-even read count: {large_be_reads:.2f} reads")
        print(f"  * System prompt average reads: {avg_prefix_reads:.1f} reads -> Cache is worth it? {cache_worth_small and cache_worth_large}")
        print(f"  * Cached requests in dataset: {cached_requests_count}/{len(rows)} ({cached_requests_count/len(rows)*100:.1f}%), cached tokens: {total_cached_tokens:,}")

        print("\n== Extension 4: Reasoning Workload Budgeting ==")
        print(f"{'Category':18}{'Requests':>10}{'% Traffic':>11}{'Cost ($/day)':>14}{'% Cost':>10}{'Energy (kWh)':>14}{'% Energy':>10}")
        print(f"{'Non-Reasoning':18}{non_reasoning_reqs:>10}{non_reasoning_reqs/len(rows)*100:>10.1f}%${non_reasoning_opt_cost:>13.2f}{non_reasoning_opt_cost/opt_cost*100:>9.1f}%{non_reasoning_wh/1000:>13.2f} kWh{non_reasoning_wh/total_wh*100:>9.1f}%")
        print(f"{'Reasoning':18}{reasoning_reqs:>10}{reasoning_reqs/len(rows)*100:>10.1f}%${reasoning_opt_cost:>13.2f}{reasoning_opt_cost/opt_cost*100:>9.1f}%{reasoning_wh/1000:>13.2f} kWh{reasoning_wh/total_wh*100:>9.1f}%")
        print(f"\n  Key Insight: Reasoning is only {reasoning_reqs/len(rows)*100:.1f}% of traffic but consumes {reasoning_wh/total_wh*100:.1f}% of serving energy!")
        
        # Policy: Cap reasoning by 50% (route routine tasks)
        cap_pct = 0.50
        capped_savings_usd = reasoning_opt_cost * cap_pct
        capped_savings_wh = reasoning_wh * cap_pct
        print(f"  Routing Rule: Restrict reasoning to complex reasoning tasks (confidence < 0.85).")
        print(f"  Simulated 50% Reasoning Reduction: Saves ${capped_savings_usd:,.2f}/day (${capped_savings_usd*30:,.2f}/mo) and {capped_savings_wh/1000:,.1f} kWh/day")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "reasoning_stats": {
            "reasoning_reqs": reasoning_reqs, "reasoning_cost": round(reasoning_opt_cost, 2),
            "reasoning_wh": round(reasoning_wh, 2), "total_wh": round(total_wh, 2),
        },
    }


if __name__ == "__main__":
    run()

