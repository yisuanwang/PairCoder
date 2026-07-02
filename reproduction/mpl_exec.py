#!/usr/bin/env python3
"""Execute matplotlib chart code headless, save the figure to PNG. Usage: mpl_exec.py <code.py> <out.png>
Prints MPL_OK or MPL_ERR <msg>."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
code = open(sys.argv[1]).read(); out = sys.argv[2]
code = code.replace("plt.show()", "").replace(".show()", "")
g = {"__name__": "__main__"}
try:
    exec(compile(code, "chart", "exec"), g)
    nums = plt.get_fignums()
    fig = plt.figure(nums[-1]) if nums else plt.gcf()
    fig.savefig(out, dpi=80, bbox_inches="tight")
    import os
    print("MPL_OK" if os.path.exists(out) and os.path.getsize(out) > 1000 else "MPL_ERR empty")
except Exception as e:
    print("MPL_ERR", repr(e)[:250])
