"""ICEWS14 time-bucket sensitivity: does rho_E survive coarsening the timestamp?
If rho_E stays high as buckets coarsen, the irreducibility is real; if it collapses toward 0,
the day-granularity number was inflation. (successor/STEP1.md non-blocking Step-2 item.)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from metric import compute_rho_e
from compute_spectrum import load_icews14

d = load_icews14()
src, dst, tau, x = d["src"], d["dst"], d["tau"], d["x"]
ts = d["z_dims"][0][1]                       # monotone day index
print(f"ICEWS14: {len(src)} edges, ts range [{ts.min():.0f},{ts.max():.0f}] ~ {int((ts.max()-ts.min())/31)} months")
for label, bucket_days in [("day",1),("week",7),("month",30),("quarter",91),("year",365)]:
    b = np.floor(ts / bucket_days)
    n_buckets = len(np.unique(b))
    rep = compute_rho_e(src, dst, tau, [(f"ts_{label}", b, "continuous")], x, seed=0)
    d0 = rep.dims[0]
    print(f"  {label:8s} ({n_buckets:4d} buckets): rho_E={rep.rho_e:.3f}  "
          f"within_key={d0.share_within_key:.3f}  V_across={d0.v_summary:.3f}")
