#!/usr/bin/env python3
"""Execute CadQuery code, export the resulting solid to STL. Usage: cad_exec.py <code.py> <out.stl>
Prints CAD_OK <nverts> or CAD_ERR <msg>."""
import sys
try:
    import cadquery as cq
    code = open(sys.argv[1]).read(); out = sys.argv[2]
    g = {"cq": cq, "cadquery": cq}
    exec(code, g)
    # find a result solid: prefer a var named result/solid/r, else last Workplane with a solid
    cand = None
    for name in ["result", "solid", "r", "res", "part"]:
        if name in g and isinstance(g[name], cq.Workplane): cand = g[name]; break
    if cand is None:
        wps = [v for v in g.values() if isinstance(v, cq.Workplane)]
        for v in reversed(wps):
            try:
                if v.vals() and v.val() is not None: cand = v; break
            except Exception: pass
    if cand is None:
        print("CAD_ERR no_solid"); sys.exit(0)
    cand.val().exportStl(out)
    import trimesh; m = trimesh.load(out, force="mesh")
    print("CAD_OK", len(m.vertices))
except Exception as e:
    print("CAD_ERR", repr(e)[:250])
