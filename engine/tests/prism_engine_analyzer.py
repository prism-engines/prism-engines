
# ------------------------- PRISM ENGINE ANALYZER --------------------------
# This is the universal analyzer used by SystemCheck™
# -------------------------------------------------------------------------

import importlib
import numpy as np
import json
import pandas as pd
from datetime import datetime
import os


# MAP ENGINE NAMES TO THEIR FULL MODULE PATHS -----------------------------
ENGINE_MAP = {
    "geometry":       "engine.core.geometry.prism_geometry.geometry_lens",
    "harmonic":       "engine.core.harmonics.wavelet_lens.wavelet_lens",
    "coherence":      "engine.core.coherence.prism_coherence.coherence_lens",
    "spectral":       "engine.core.harmonics.spectral_lens.spectral_lens",
    "var":            "engine.core.dynamics.var_granger_lens.var_lens",
    "regime_switch":  "engine.core.dynamics.regime_switching_lens.regime_lens",
    "entropy":        "engine.core.dynamics.transfer_entropy_lens.transfer_entropy",
    "anomaly":        "engine.core.dynamics.anomaly_detection_lens.anomaly_lens",
    "tda":            "engine.core.topology.tda_lens.tda_lens",
    "network":        "engine.core.topology.network_graph_lens.network_lens",
}


# ENGINE LOADER ------------------------------------------------------------

def load_engine(engine_name):
    if engine_name not in ENGINE_MAP:
        raise KeyError(f"Unknown engine: {engine_name}. Available engines: {list(ENGINE_MAP.keys())}")

    module_path, func_name = ENGINE_MAP[engine_name].rsplit('.', 1)

    module = importlib.import_module(module_path)
    return getattr(module, func_name)


# ---------------------- CORE EVALUATION LOGIC ----------------------------

def run_engine(engine_fn, data):
    try:
        return engine_fn(data)
    except Exception as e:
        return {"error": str(e)}


def test_behavior(result, expected):
    checks = []

    # Type check
    if "output_type" in expected:
        checks.append(("type_ok", isinstance(result, expected["output_type"])))

    # Required keys (for dict outputs)
    if "required_keys" in expected and isinstance(result, dict):
        ok = all(k in result for k in expected["required_keys"])
        checks.append(("required_keys_ok", ok))

    # Range check (scalar expected)
    if "range" in expected and np.isscalar(result):
        lo, hi = expected["range"]
        checks.append(("range_ok", lo <= result <= hi))

    return checks


def robustness_tests(engine_fn, data):
    noise = data + np.random.normal(0, 0.01, size=len(data))
    scaled = data * 50

    base = run_engine(engine_fn, data)
    noisy = run_engine(engine_fn, noise)
    scaled_out = run_engine(engine_fn, scaled)

    def dist(a, b):
        if isinstance(a, dict) and isinstance(b, dict):
            return float(sum(abs(a[k] - b[k]) for k in a if k in b))
        if np.isscalar(a):
            return float(abs(a - b))
        return np.nan

    return {
        "noise_sensitivity": dist(base, noisy),
        "scale_sensitivity": dist(base, scaled_out)
    }


# ---------------------- MAIN ANALYZER FUNCTION ---------------------------

def analyze_engine(engine_name, input_series, expected, save_report=False):
    engine_fn = load_engine(engine_name)

    result = run_engine(engine_fn, input_series)
    behavior = test_behavior(result, expected)
    robustness = robustness_tests(engine_fn, input_series)

    status = "PASS" if all(ok for _, ok in behavior) else "FAIL"

    report = {
        "engine_name": engine_name,
        "result": result,
        "behavior_checks": {name: ok for name, ok in behavior},
        "robustness": robustness,
        "status": status,
        "timestamp": datetime.now().isoformat()
    }

    if save_report:
        out_path = "/content/drive/MyDrive/prism-engine/engine/tests/reports"
        os.makedirs(out_path, exist_ok=True)
        out_file = f"{out_path}/{engine_name}_report.json"

        with open(out_file, "w") as f:
            json.dump(report, f, indent=4)

    return report
