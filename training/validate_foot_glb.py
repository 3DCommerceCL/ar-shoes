"""Validador del foot.glb — revisa el contrato de GUIA_ESCENA_BLENDER.md automáticamente.

Correr DESPUÉS de exportar el GLB desde Blender:

    "C:/Program Files/Blender Foundation/Blender 5.1/blender.exe" --background ^
        --python validate_foot_glb.py -- --glb models/foot.glb

Salida: checklist ✔/✖/⚠ con el detalle de qué corregir, y (con --render) un preview
validate_preview.jpg con los keypoints proyectados sobre el render.
Código de salida: 0 = todo OK (warnings permitidos), 2 = hay errores que corregir.
"""
import bpy
import sys
import math
import argparse
from pathlib import Path

import numpy as np
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

KP_NAMES = ["heel", "toe", "ankle_in", "ankle_out", "ball", "toe_tip"]
LEG_CANON = ["leg_bare", "leg_pants_dark", "leg_pants_light"]

results = []          # (nivel, etiqueta, detalle)  nivel: OK | WARN | FAIL


def report(level, label, detail=""):
    results.append((level, label, detail))


def base_name(o):
    return o.name.lower().split(".")[0]


def world_bbox(objs):
    mn = Vector((1e9, 1e9, 1e9)); mx = Vector((-1e9, -1e9, -1e9))
    for o in objs:
        if o.type != 'MESH':
            continue
        for c in o.bound_box:
            wc = o.matrix_world @ Vector(c)
            for i in range(3):
                mn[i] = min(mn[i], wc[i]); mx[i] = max(mx[i], wc[i])
    return mn, mx


def open_boundary_edges(obj):
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.transform(obj.matrix_world)
    # El export a glTF divide los vértices por normal (aristas duras) → una malla cerrada
    # parecería toda "borde abierto". Re-soldamos por distancia antes de contar.
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    n = sum(1 for e in bm.edges if len(e.link_faces) == 1)
    bm.free()
    return n


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", default="models/foot.glb")
    ap.add_argument("--render", action="store_true", help="genera validate_preview.jpg con los kp proyectados")
    args = ap.parse_args(argv)

    glb = Path(args.glb)
    if not glb.exists():
        print(f"✖ No existe: {glb}")
        sys.exit(2)

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=str(glb))
    objs = list(bpy.data.objects)

    # ---- clasificar ----
    kp = {}
    legs = {}
    for o in objs:
        b = base_name(o)
        if o.type == 'EMPTY' and b.startswith("kp_"):
            kp[b[3:]] = o
        elif b.startswith("leg_"):
            legs.setdefault(b, []).append(o)
    leg_meshes = [m for v in legs.values() for m in v if m.type == 'MESH']
    # hijos mesh de un empty leg_* también cuentan
    for v in list(legs.values()):
        for o in v:
            for ch in o.children_recursive:
                if ch.type == 'MESH' and ch not in leg_meshes:
                    leg_meshes.append(ch)
    foot_meshes = [o for o in objs if o.type == 'MESH' and o not in leg_meshes]

    if not foot_meshes:
        report("FAIL", "Malla del pie", "no encontré ninguna malla que no sea leg_*")
        finish(args, None, kp)
    fmn, fmx = world_bbox(foot_meshes)
    L = fmx.y - fmn.y     # largo del pie (eje Y)
    W = fmx.x - fmn.x
    H = fmx.z - fmn.z

    # ---- 1. escala ----
    if 0.22 <= L <= 0.32:
        report("OK", "Largo del pie", f"{L*100:.1f} cm (objetivo ≈26 cm)")
    elif 0.15 <= L <= 0.45:
        report("WARN", "Largo del pie", f"{L*100:.1f} cm — revisá la escala (objetivo ≈26 cm)")
    else:
        report("FAIL", "Largo del pie", f"{L*100:.1f} cm — escala mal (1 unidad = 1 m; el pie debe medir ≈0.26)")
    if not (0.05 <= W <= 0.16):
        report("WARN", "Ancho del pie", f"{W*100:.1f} cm — esperable 7–12 cm")

    # ---- 2. orientación / origen ----
    if L >= W:
        report("OK", "Orientación", "el eje largo es +Y")
    else:
        report("FAIL", "Orientación", f"el pie es más ancho (X {W*100:.0f}cm) que largo (Y {L*100:.0f}cm): debe apuntar a +Y")
    if abs(fmn.z) <= 0.015:
        report("OK", "Apoyado en el suelo", f"min Z = {fmn.z*100:.1f} cm")
    else:
        report("FAIL", "Apoyado en el suelo", f"min Z = {fmn.z*100:.1f} cm — la planta debe estar en Z=0")
    if abs(fmn.y) <= 0.03:
        report("OK", "Origen en el talón", f"el talón arranca en Y = {fmn.y*100:.1f} cm")
    else:
        report("FAIL", "Origen en el talón", f"el talón está en Y = {fmn.y*100:.1f} cm — debe estar en Y≈0 (origen en el talón)")

    # ---- 3. keypoints ----
    missing = [n for n in KP_NAMES if n not in kp]
    if not missing:
        report("OK", "Empties kp_*", "los 6 presentes y sobreviven al export")
        p = {n: kp[n].matrix_world.translation for n in KP_NAMES}
        frac = lambda n: (p[n].y - fmn.y) / max(L, 1e-6)
        checks = [
            ("kp_heel atrás",      frac("heel") <= 0.20,  f"y={frac('heel'):.2f} del largo (esperado ≤0.20)"),
            ("kp_toe adelante",    0.60 <= frac("toe") <= 1.05, f"y={frac('toe'):.2f} (esperado 0.60–1.05)"),
            ("kp_ball planta",     0.55 <= frac("ball") <= 0.95, f"y={frac('ball'):.2f} (esperado 0.55–0.95)"),
            ("kp_toe_tip punta",   frac("toe_tip") >= 0.80, f"y={frac('toe_tip'):.2f} (esperado ≥0.80)"),
            ("tobillos en altura", p["ankle_in"].z > 0.03 and p["ankle_out"].z > 0.03,
             f"z={p['ankle_in'].z*100:.1f}/{p['ankle_out'].z*100:.1f} cm (esperado >3 cm)"),
        ]
        for label, ok, det in checks:
            report("OK" if ok else "FAIL", label, det)
        if p["ankle_in"].x < p["ankle_out"].x:
            report("OK", "ankle_in = lado MEDIAL", "in.x < out.x (pie derecho correcto)")
        else:
            report("FAIL", "ankle_in = lado MEDIAL",
                   "kp_ankle_in quedó del lado EXTERNO. Para un pie DERECHO que apunta a +Y, el maléolo "
                   "medial (interno) va hacia -X. O están intercambiados los Empties, o modelaste un pie izquierdo.")
        dentro = sum(1 for n in KP_NAMES
                     if fmn.x - 0.03 <= p[n].x <= fmx.x + 0.03
                     and fmn.y - 0.03 <= p[n].y <= fmx.y + 0.03
                     and fmn.z - 0.02 <= p[n].z <= fmx.z + 0.10)
        if dentro == 6:
            report("OK", "Keypoints sobre el pie", "los 6 dentro del volumen del pie")
        else:
            report("FAIL", "Keypoints sobre el pie", f"{6-dentro} empties quedaron lejos de la malla (¿sin parentar / se movieron?)")
    else:
        report("FAIL", "Empties kp_*", f"faltan: {', '.join('kp_'+m for m in missing)} "
               "(¿los exportaste? ¿nombres exactos?)")

    # ---- 4. piernas ----
    # Requisito real: una pierna con piel + al menos una con tela. Los TONOS se randomizan en el
    # render (piel y pantalón, distinto en cada imagen), así que no hacen falta variantes de color
    # horneadas en el GLB — basta leg_bare + leg_pants.
    has_bare = "leg_bare" in legs
    pants_variants = [n for n in legs if n.startswith("leg_pants")]
    if has_bare and pants_variants:
        report("OK", "Variantes leg_*", f"leg_bare + {len(pants_variants)} de tela: {pants_variants}")
    elif not legs:
        report("FAIL", "Variantes leg_*", "no hay ningún objeto leg_* (la máscara saldría sin pierna)")
    else:
        falta = ("leg_bare (pierna con piel)" if not has_bare else "alguna leg_pants* (tela)")
        report("FAIL", "Variantes leg_*", f"falta {falta}. Presentes: {sorted(legs)}")
    for name, group in legs.items():
        meshes = [m for m in group if m.type == 'MESH'] + \
                 [ch for o in group for ch in o.children_recursive if ch.type == 'MESH']
        if not meshes:
            report("FAIL", f"{name}", "no tiene ninguna malla")
            continue
        lmn, lmx = world_bbox(meshes)
        top = lmx.z
        if top < 0.20:
            report("WARN", f"{name}", f"llega solo a {top*100:.0f} cm — recomendado ~30 cm sobre el tobillo")
        elif top > 0.60:
            report("WARN", f"{name}", f"llega a {top*100:.0f} cm — más de lo necesario (la cámara se aleja)")
        else:
            report("OK", f"{name}", f"altura {top*100:.0f} cm")
        holes = sum(open_boundary_edges(m) for m in meshes)
        if holes:
            report("WARN", f"{name} cerrada", f"{holes} aristas de borde abierto (tapala arriba: F / Alt+F)")

    # ---- 5. mallas del pie cerradas + material ----
    holes_foot = sum(open_boundary_edges(m) for m in foot_meshes)
    if holes_foot:
        report("WARN", "Pie cerrado", f"{holes_foot} aristas de borde abierto en la malla del pie")
    sin_mat = [m.name for m in foot_meshes if not m.data.materials]
    if sin_mat:
        report("WARN", "Materiales", f"mallas sin material: {sin_mat}")

    finish(args, (fmn, fmx), kp)


def finish(args, bbox, kp):
    icons = {"OK": "✔", "WARN": "⚠", "FAIL": "✖"}
    print("\n===== VALIDACIÓN DE " + args.glb + " =====")
    for level, label, detail in results:
        print(f" {icons[level]} {label:<26} {detail}")
    fails = sum(1 for l, *_ in results if l == "FAIL")
    warns = sum(1 for l, *_ in results if l == "WARN")
    print(f"\n Resultado: {fails} errores, {warns} avisos")
    if fails == 0:
        print(" ✅ El GLB cumple el contrato — listo para el preview de renders (0b_blender_render.py --preview)")
    else:
        print(" ❌ Corregí los ✖ y volvé a exportar/validar")

    if args.render and bbox:
        render_preview(bbox, kp)
    sys.exit(0 if fails == 0 else 2)


def render_preview(bbox, kp):
    fmn, fmx = bbox
    center = (fmn + fmx) / 2
    radius = max((fmx - fmn)) or 0.3
    cam_d = bpy.data.cameras.new("VCam"); cam_d.lens = 28; cam_d.clip_start = 0.01
    cam = bpy.data.objects.new("VCam", cam_d)
    bpy.context.scene.collection.objects.link(cam)
    dist = radius * 2.6
    cam.location = center + Vector((dist * 0.55, -dist * 0.6, dist * 0.55))
    cam.rotation_euler = (center - cam.location).to_track_quat('-Z', 'Y').to_euler()
    bpy.context.scene.camera = cam
    bpy.ops.object.light_add(type='SUN')
    bpy.context.object.data.energy = 3
    bpy.context.object.rotation_euler = (math.radians(50), 0, math.radians(30))
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'; sc.cycles.samples = 24
    sc.render.resolution_x = sc.render.resolution_y = 640
    sc.render.image_settings.file_format = 'PNG'
    out = str(Path(bpy.path.abspath("//")) / "validate_preview_raw.png") if bpy.data.filepath \
        else "validate_preview_raw.png"
    sc.render.filepath = out
    bpy.ops.render.render(write_still=True)

    img = bpy.data.images.load(out)
    w, h = img.size
    arr = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
    for name, e in kp.items():
        co = world_to_camera_view(sc, cam, e.matrix_world.translation)
        px, py = int(co.x * w), int(co.y * h)   # pixels bottom-up (igual que arr)
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                if dx * dx + dy * dy <= 16 and 0 <= py + dy < h and 0 <= px + dx < w:
                    arr[py + dy, px + dx, :3] = (1.0, 0.85, 0.0)
    img.pixels[:] = arr.ravel()
    final = out.replace("_raw.png", ".jpg")
    img.file_format = 'JPEG'
    img.filepath_raw = final
    img.save()
    Path(out).unlink(missing_ok=True)
    print(f" 📷 Preview con keypoints: {final}")


if __name__ == "__main__":
    main()
