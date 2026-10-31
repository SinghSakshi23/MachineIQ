"""
MaintenIQ — Comprehensive Live Simulation Verification Script
Tests all 5 scenarios with multiple ticks through the live Flask API endpoints:
- /api/simulate/next
- /api/predict
Verifies dynamic reading updates, model scoring, failure mode detection, and seed locking.
"""

import urllib.request
import json

BASE_URL = "http://127.0.0.1:5000"

def post_json(endpoint, data):
    req = urllib.request.Request(
        f"{BASE_URL}{endpoint}",
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def run_scenario_stream(scenario_name, num_ticks=4, seed=None):
    print(f"\n=======================================================")
    print(f" TESTING SCENARIO: {scenario_name.upper()} ({num_ticks} ticks)")
    print(f"=======================================================")
    state = None
    results = []

    for tick in range(1, num_ticks + 1):
        sim_payload = {
            **(state or {}),
            "scenario": scenario_name,
            "step": tick,
            "scenario_changed": (tick == 1),
            "seed": seed,
        }
        next_reading = post_json("/api/simulate/next", sim_payload)
        state = next_reading

        predict_payload = {
            "Type": next_reading["Type"],
            "Air temperature [K]": next_reading["Air temperature [K]"],
            "Process temperature [K]": next_reading["Process temperature [K]"],
            "Rotational speed [rpm]": next_reading["Rotational speed [rpm]"],
            "Torque [Nm]": next_reading["Torque [Nm]"],
            "Tool wear [min]": next_reading["Tool wear [min]"],
        }
        scored = post_json("/api/predict", predict_payload)
        
        prob = scored.get("failure_probability_pct", 0.0)
        level = scored.get("level", "unknown")
        label = scored.get("label", "unknown")
        modes = [m["code"] for m in scored.get("diagnostics", {}).get("predicted_modes", [])]
        delta_t = round(next_reading["Process temperature [K]"] - next_reading["Air temperature [K]"], 1)
        power = scored.get("physical_metrics", {}).get("power", {}).get("value", 0)
        strain = scored.get("physical_metrics", {}).get("overstrain", {}).get("value", 0)

        pwr_val = int(round(power))
        str_val = int(round(strain))

        print(f"Tick {tick:2d} | Air:{next_reading['Air temperature [K]']:5.1f}K Proc:{next_reading['Process temperature [K]']:5.1f}K (dT:{delta_t:4.1f}K) | "
              f"Speed:{next_reading['Rotational speed [rpm]']:4d}rpm Torque:{next_reading['Torque [Nm]']:4.1f}Nm (Pwr:{pwr_val:5d}W) | "
              f"Wear:{next_reading['Tool wear [min]']:3d}m (Str:{str_val:5d}) -> Risk: {prob:5.1f}% [{level.upper():8s}] Modes: {modes}")

        results.append({
            "tick": tick,
            "reading": next_reading,
            "scored": scored,
            "prob": prob,
            "level": level,
            "modes": modes
        })
    return results

def main():
    # 1. Normal Operation
    res_norm = run_scenario_stream("normal", num_ticks=4)
    assert all(r["level"] == "low" for r in res_norm), "Normal operation should produce Low Risk"
    # Readings must change between ticks
    speeds = [r["reading"]["Rotational speed [rpm]"] for r in res_norm]
    assert len(set(speeds)) > 1, "Operating readings should change over time!"

    # 2. Thermal Stress
    res_therm = run_scenario_stream("thermal_stress", num_ticks=4)
    assert any(r["level"] in ["high", "critical"] for r in res_therm), "Thermal stress should elevate risk"
    assert any("HDF" in r["modes"] for r in res_therm), "Thermal stress should trigger HDF diagnostic"

    # 3. Power-Related Stress
    res_pwr = run_scenario_stream("power_surge", num_ticks=4)
    assert any(r["level"] in ["high", "critical"] for r in res_pwr), "Power stress should elevate risk"
    assert any("PWF" in r["modes"] for r in res_pwr), "Power surge should trigger PWF diagnostic"

    # 4. Mechanical Overstrain
    res_mech = run_scenario_stream("mechanical_overstrain", num_ticks=4)
    assert any(r["level"] in ["high", "critical"] for r in res_mech), "Mechanical overstrain should elevate risk"
    assert any("OSF" in r["modes"] for r in res_mech), "Mechanical overstrain should trigger OSF diagnostic"

    # 5. Random Mixed Operation (Fluctuating across all regimes)
    print("\n=======================================================")
    print(" TESTING SCENARIO: RANDOM_MIXED (24-tick full cycle)")
    print("=======================================================")
    res_mixed = run_scenario_stream("random_mixed", num_ticks=24)
    levels_seen = {r["level"] for r in res_mixed}
    probs_seen = [r["prob"] for r in res_mixed]
    print(f"\nMixed operation distinct risk levels observed: {levels_seen}")
    print(f"Min probability: {min(probs_seen):.1f}%, Max probability: {max(probs_seen):.1f}%")
    assert "low" in levels_seen and ("high" in levels_seen or "critical" in levels_seen), \
        "Random mixed scenario must show dynamic variation across lower and higher risk tiers!"

    # 6. Seed Locking Determinism Test
    print("\n=======================================================")
    print(" TESTING SEED DETERMINISM (Seed=12345)")
    print("=======================================================")
    run1 = run_scenario_stream("normal", num_ticks=3, seed=12345)
    run2 = run_scenario_stream("normal", num_ticks=3, seed=12345)
    for t1, t2 in zip(run1, run2):
        assert t1["reading"]["Rotational speed [rpm]"] == t2["reading"]["Rotational speed [rpm]"], "Locked seed must produce repeatable ticks"
        assert t1["reading"]["Torque [Nm]"] == t2["reading"]["Torque [Nm]"], "Locked seed must produce repeatable ticks"
    print("Determinism verified: identical seeds produce identical telemetry sequences across ticks.")

    print("\nALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
