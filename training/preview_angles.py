"""Tanda SISTEMÁTICA de renders para revisión humana: recorre TODOS los ángulos configurados
(no al azar) y arma un contact-sheet etiquetado, para decidir qué encuadres hay que corregir.

    "C:/Program Files/Blender Foundation/Blender 5.1/blender.exe" --background ^
        --python preview_angles.py -- --foot models/foot_rig_full.glb --shoe_dir models/shoes ^
        --out data_angles

Por cada configuración de cámara (cenital/diagonal/frontal/lateral) renderiza varias elevaciones
× azimuts, alternando zapato y variante de pierna. Salida:
    <out>/images, <out>/masks, <out>/angles.html   ← abrí este HTML
Cada tarjeta dice: ángulo, elevación, azimut, distancia, lente y qué % del cuadro ocupa cada clase.
"""
import bpy
import sys
import math
import json
import argparse
from pathlib import Path

import numpy as np
from mathutils import Vector, Euler

sys.path.insert(0, str(Path(__file__).parent))
import importlib.util as _u
_spec = _u.spec_from_file_location("rnd", str(Path(__file__).parent / "0b_blender_render.py"))
# 0b_blender_render.py ejecuta main() sólo bajo __main__, así que es seguro importarlo
rnd = _u.module_from_spec(_spec)
sys.argv_backup = list(sys.argv)
sys.argv = [sys.argv[0]]          # que su parser no vea nuestros flags
_spec.loader.exec_module(rnd)
sys.argv = sys.argv_backup

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--foot", default="models/foot_rig_full.glb")
    p.add_argument("--shoe_dir", default="models/shoes")
    p.add_argument("--out", default="data_angles")
    p.add_argument("--size", type=int, default=320)
    p.add_argument("--hdri_dir", default=None)
    p.add_argument("--floor_dir", default=None)
    return p.parse_args(argv)


# (nombre, elevaciones, azimuts) — barrido sistemático de lo que hoy configura CAMERA_CONFIGS
SWEEP = [
    ("cenital",  [85, 75, 65], [0, 90, 180, 270]),
    ("diagonal", [60, 50, 40], [0, 90, 180, 270]),
    ("frontal",  [35, 25, 15], [0, 40, -40]),
    ("lateral",  [45, 30],     [90, 105, 75]),
]


def main():
    args = parse_args()
    out = Path(args.out).resolve()
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    import random
    rng = random.Random(7)
    np.random.seed(7)

    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.resolution_percentage = 100
    scene.render.resolution_x = scene.render.resolution_y = args.size

    rnd.clear_scene()
    root = bpy.data.objects.new("Root", None)
    scene.collection.objects.link(root)

    foot_objs = rnd.import_glb(args.foot)
    for o in foot_objs:
        if o.parent is None:
            o.parent = root

    # piernas por nombre leg_*
    leg_groups = {}
    for o in foot_objs:
        base = o.name.lower().split('.')[0]
        if base.startswith("leg_"):
            leg_groups.setdefault(base, []).append(o)
    leg_meshes = {m for v in leg_groups.values() for m in v}
    foot_only = [o for o in foot_objs if o not in leg_meshes]

    shoes = {}
    sd = Path(args.shoe_dir)
    for sf in sorted(list(sd.glob("*.glb")) + list(sd.glob("*.gltf"))):
        objs = rnd.import_glb(sf)
        for o in objs:
            if o.parent is None:
                o.parent = root
        shoes[sf.stem] = objs
    print(f"pie={[o.name for o in foot_only if o.type=='MESH']} piernas={list(leg_groups)} "
          f"zapatos={list(shoes)}")

    kp_objs = rnd.find_or_make_keypoints([o for o in foot_only if o.type == 'MESH'], root)

    hdris = []
    if args.hdri_dir and Path(args.hdri_dir).is_dir():
        hdris = [p for e in ("*.hdr", "*.exr") for p in Path(args.hdri_dir).glob(e)]
    floors = []
    if args.floor_dir and Path(args.floor_dir).is_dir():
        floors = [p for e in ("*.jpg", "*.png") for p in Path(args.floor_dir).glob(e)]

    shoe_names = list(shoes)
    leg_names = list(leg_groups)
    cards = []
    idx = 0
    for angname, elevs, azims in SWEEP:
        for elev in elevs:
            for azim in azims:
                shoe = shoe_names[idx % len(shoe_names)]
                legname = leg_names[idx % len(leg_names)] if leg_names else None
                for nm, objs in shoes.items():
                    for o in objs:
                        o.hide_render = (nm != shoe)
                for nm, objs in leg_groups.items():
                    for o in objs:
                        o.hide_render = (nm != legname)
                active_shoe = shoes[shoe]
                active_leg = leg_groups.get(legname, [])

                # colores + sin rotación aleatoria (queremos comparar ángulos, no poses)
                rnd.randomize_skin_and_fabric(
                    [o for o in foot_only if o.type == 'MESH'],
                    [o for o in active_leg if "pants" not in o.name.lower()],
                    [o for o in active_leg if "pants" in o.name.lower()], rng)
                root.rotation_euler = Euler((0, 0, 0))
                root.location.z = 0.0
                bpy.context.view_layer.update()

                visible = list(foot_only) + list(active_shoe) + list(active_leg)
                mn, _ = rnd.mesh_world_bounds(visible)
                if mn is not None:
                    root.location.z = -mn.z
                    bpy.context.view_layer.update()

                smn, smx = rnd.mesh_world_bounds([o for o in foot_only if o.type == 'MESH'] + list(active_shoe))
                center = (smn + smx) / 2
                radius = max((smx - smn)) / 2
                lmn, lmx = rnd.mesh_world_bounds(list(active_leg))
                leg_top = lmx.z if lmx is not None else None
                leg_r = (max(lmx.x - lmn.x, lmx.y - lmn.y) / 2) if lmn is not None else None

                # cámara determinista en este ángulo (misma fórmula de seguridad que el render real)
                # misma lógica que place_camera(): distancia DESDE EL SUJETO + alejar hasta que la
                # cámara no quede dentro de la pierna
                e, a = math.radians(elev), math.radians(azim)
                cfg = next(c for c in rnd.CAMERA_CONFIGS if c[0] == angname)
                dist = (cfg[4] + cfg[5]) / 2
                d_hat = Vector((math.cos(e) * math.sin(a), math.cos(e) * math.cos(a), math.sin(e)))
                r_leg = (leg_r or 0.08) + 0.04
                for _ in range(24):
                    loc = center + d_hat * dist
                    if not leg_top or loc.z > leg_top or math.hypot(loc.x, loc.y) > r_leg:
                        break
                    dist *= 1.15
                loc = center + d_hat * dist
                cam = bpy.data.objects.get("Camera")
                if cam is None:
                    cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
                    scene.collection.objects.link(cam)
                cam.location = loc
                cam.rotation_euler = (center - loc).to_track_quat('-Z', 'Y').to_euler()
                lens = 28.0
                cam.data.lens = lens
                cam.data.clip_start = 0.01
                scene.camera = cam

                stem = f"ang_{idx:03d}"
                if hdris:
                    rnd.set_world_hdri(hdris, rng)
                else:
                    rnd.set_world_hdri(None, rng); rnd.add_random_lights(rng)
                floor = rnd.make_floor(floors, rng)

                scene.cycles.samples = 28
                rnd.set_color_management(scene, 'AgX')
                tmp = out / "_tmp.png"
                scene.render.image_settings.file_format = 'PNG'
                scene.render.filepath = str(tmp)
                bpy.ops.render.render(write_still=True)
                color = rnd.np_load(tmp, non_color=True)
                rnd.np_save(color, out / "images" / f"{stem}.jpg", 'JPEG', scene, quality=92)

                kps = rnd.project_keypoints(scene, cam, kp_objs)

                # máscara
                orig = {o.name: list(o.data.materials) for o in bpy.data.objects if o.type == 'MESH'}
                def assign(objs, cls):
                    m = rnd.emission_mat(rnd.MASK_COLORS[cls])
                    for o in objs:
                        if o.type == 'MESH':
                            o.data.materials.clear(); o.data.materials.append(m)
                assign([floor], rnd.CLASS_BG)
                assign(active_leg, rnd.CLASS_LEG)
                assign([o for o in foot_only if o.type == 'MESH'], rnd.CLASS_FOOT)
                assign(active_shoe, rnd.CLASS_SHOE)
                pf, pw = scene.cycles.pixel_filter_type, scene.cycles.filter_width
                scene.cycles.pixel_filter_type = 'BOX'; scene.cycles.filter_width = 0.01
                scene.cycles.samples = 1
                bgn = scene.world.node_tree.nodes.get("Background")
                if bgn:
                    bgn.inputs["Strength"].default_value = 0.0
                rnd.set_color_management(scene, 'Raw')
                scene.render.filepath = str(tmp)
                bpy.ops.render.render(write_still=True)
                mrgb = rnd.np_load(tmp, non_color=True)
                maxc = mrgb.max(axis=2)
                im = np.where(maxc < 0.25, 0, mrgb.argmax(axis=2) + 1).astype(np.float32)
                rnd.np_save(np.repeat((im / 255.0)[..., None], 3, axis=2),
                            out / "masks" / f"{stem}.png", 'PNG', scene)
                tmp.unlink(missing_ok=True)
                for o in bpy.data.objects:
                    if o.type == 'MESH' and o.name in orig:
                        o.data.materials.clear()
                        for m in orig[o.name]:
                            o.data.materials.append(m)
                scene.cycles.pixel_filter_type, scene.cycles.filter_width = pf, pw

                pct = {c: round(100 * float((im == c).mean()), 1) for c in (1, 2, 3)}
                vis = sum(1 for k in kps if k[2] > 0)
                cards.append(dict(stem=stem, angle=angname, elev=elev, azim=azim,
                                  dist=round(dist, 3), lens=lens, shoe=shoe, leg=legname,
                                  pct=pct, kps_vis=vis))
                print(f"  [{idx:3}] {angname:9} elev={elev:3}° azim={azim:4}° d={dist:.2f}m "
                      f"{shoe:6} {str(legname):10} zapato={pct[3]:4.1f}% pierna={pct[1]:4.1f}% kp={vis}/12")
                idx += 1

    (out / "angles.json").write_text(json.dumps(cards, indent=1), encoding="utf-8")
    rows = "".join(
        f'<figure><div class=imgs><img src="images/{c["stem"]}.jpg" loading=lazy>'
        f'<img src="masks/{c["stem"]}.png" class=mask loading=lazy></div>'
        f'<figcaption><b>{c["stem"]}</b> · {c["angle"]}<br>'
        f'elev {c["elev"]}° · azim {c["azim"]}° · {c["dist"]} m · {c["lens"]:.0f} mm<br>'
        f'{c["shoe"]} / {c["leg"]}<br>'
        f'<span class=l3>zapato {c["pct"][3]}%</span> '
        f'<span class=l1>pierna {c["pct"][1]}%</span> '
        f'<span class=l2>pie {c["pct"][2]}%</span><br>kp {c["kps_vis"]}/12</figcaption></figure>'
        for c in cards)
    (out / "angles.html").write_text(f"""<!doctype html><meta charset=utf-8>
<title>Ángulos — {len(cards)} renders</title>
<style>body{{background:#111;color:#ddd;font-family:system-ui,sans-serif;margin:14px}}
h1{{font-size:17px}} figure{{display:inline-block;margin:6px;width:270px;vertical-align:top}}
.imgs img{{width:132px;height:132px;object-fit:contain;background:#000;border-radius:4px}}
.mask{{filter:brightness(70)}} figcaption{{font-size:11px;color:#9aa;line-height:1.5;margin-top:3px}}
.l1{{color:#4af}} .l2{{color:#f66}} .l3{{color:#5e6}}</style>
<h1>{len(cards)} renders — barrido sistemático de ángulos</h1>
<p style=font-size:12px;color:#89a>Izquierda: render. Derecha: máscara (verde zapato, azul pierna,
rojo pie). Decime los <b>ang_NNN</b> que haya que corregir.</p>
{rows}""", encoding="utf-8")
    print(f"\n✅ {len(cards)} renders en {out}")
    print(f"   Abrí: {out / 'angles.html'}")


if __name__ == "__main__":
    main()
