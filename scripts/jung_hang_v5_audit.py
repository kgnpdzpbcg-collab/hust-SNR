"""Real-source preflight checks for the remaining five-phase adaptation assumptions.

Never reads target Dev / holdout. Produces measurements, not training choices.
"""
from __future__ import annotations
import numpy as np
from jung_hang_models_v3 import five_phase_vsd, _interpolate_current_window
from jung_hang_models_v5 import _fundamental_thd_rms


def audit_source(raw, meta, config):
    native=[]
    bad=[]
    for rid in sorted(raw):
        t,x,w = raw[rid]
        try:
            fundamental, thd = _fundamental_thd_rms(t,x,w)
            native.extend(np.asarray(thd).tolist())
        except ValueError as e:
            bad.append(dict(record_id=str(rid),reason=str(e)))
    levels = config["jung_train_thdn_percent"]
    fractions = {str(p): float(np.mean(np.asarray(native) < p/100.0))
                 if native else 0.0 for p in levels}
    ratios=[]
    source_healthy=[row for row in meta if int(row["label"])==0]
    visited=set()
    for row in source_healthy:
        rid=row["record_id"]
        if rid in visited:
            continue
        visited.add(rid)
        t,x,w=raw[rid]
        _, segment = _interpolate_current_window(t,x,row["start"],row["end"],256)
        v=five_phase_vsd(segment.T,tuple(config["hang_phase_angles_deg"]))
        f_energy=float(np.mean(v[0]**2+v[1]**2))
        h_energy=float(np.mean(v[2]**2+v[3]**2))
        if f_energy>1e-12:
            ratios.append(float(np.sqrt(h_energy/f_energy)))
    return {
        "native_thd_reference":"source recording 0.10..0.45 s healthy interval",
        "native_thd_samples":len(native),
        "native_thd_min":float(min(native)) if native else None,
        "native_thd_median":float(np.median(native)) if native else None,
        "native_thd_max":float(max(native)) if native else None,
        "thdn_nonzero_injection_fraction_by_target_percent":fractions,
        "thdn_failed_records":bad,
        "healthy_vsd_records":len(ratios),
        "healthy_vsd_third_to_fundamental_rms_median":
            float(np.median(ratios)) if ratios else None,
        "hang_phase_angles_deg":config["hang_phase_angles_deg"],
        "phase_order_physically_verified":False,
    }
