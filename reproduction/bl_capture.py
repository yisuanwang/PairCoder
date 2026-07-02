#!/usr/bin/env python3
"""Helper: run a user Blender script, then JOIN all meshes, EXPORT an OBJ, and RENDER one
canonical view to PNG. Invoked as: blender --background --python bl_capture.py -- <code.py> <out.obj> <out.png>
Prints 'CAP_OK <nverts>' or 'CAP_ERR <msg>'."""
import bpy, sys, os, math, mathutils

argv = sys.argv[sys.argv.index("--") + 1:]
code_path, out_obj, out_png = argv[0], argv[1], argv[2]

for o in list(bpy.data.objects):
    try: bpy.data.objects.remove(o, do_unlink=True)
    except Exception: pass
try:
    exec(compile(open(code_path).read(), "gen", "exec"), {"__name__": "__main__", "bpy": bpy})
    meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.data and len(o.data.vertices) > 0]
    nv = sum(len(o.data.vertices) for o in meshes)
    if nv == 0:
        print("CAP_ERR no_geometry"); sys.exit(0)
    # join meshes
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes: o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    # normalize: center + unit-scale (so Chamfer/render are scale/position invariant)
    bb = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    cx = sum(v.x for v in bb)/8; cy = sum(v.y for v in bb)/8; cz = sum(v.z for v in bb)/8
    size = max(max(v.x for v in bb)-min(v.x for v in bb),
               max(v.y for v in bb)-min(v.y for v in bb),
               max(v.z for v in bb)-min(v.z for v in bb), 1e-6)
    obj.location = (obj.location.x-cx, obj.location.y-cy, obj.location.z-cz)
    obj.scale = tuple(s/size for s in obj.scale)
    bpy.context.view_layer.update()
    bpy.ops.wm.obj_export(filepath=out_obj, export_selected_objects=True)
    # camera + light + render
    cam_data = bpy.data.cameras.new("C"); cam = bpy.data.objects.new("C", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = (2.2, -2.2, 1.6)
    d = mathutils.Vector((0,0,0)) - cam.location
    cam.rotation_euler = d.to_track_quat('-Z','Y').to_euler()
    bpy.context.scene.camera = cam
    light_data = bpy.data.lights.new("L", type='SUN'); light = bpy.data.objects.new("L", light_data)
    light.location = (3,-3,4); bpy.context.scene.collection.objects.link(light); light_data.energy = 3
    sc = bpy.context.scene
    engines = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items]
    sc.render.engine = 'BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in engines else ('BLENDER_EEVEE' if 'BLENDER_EEVEE' in engines else engines[0])
    sc.render.resolution_x = sc.render.resolution_y = 224
    sc.render.filepath = out_png
    sc.world = bpy.data.worlds.new("W"); sc.world.use_nodes = True
    bpy.ops.render.render(write_still=True)
    print("CAP_OK", nv)
except Exception as e:
    print("CAP_ERR", repr(e)[:300])
