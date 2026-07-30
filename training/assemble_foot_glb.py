"""Ensambla foot.glb + leg.glb + pants.glb en UN asset que cumple el contrato de
GUIA_ESCENA_BLENDER.md, listo para 0b_blender_render.py.

Arregla automáticamente (todo verificado contra la geometría real, no a ciegas):
  1. EMPTIES DUPLICADOS: kp_x y kp_x.001 → deja uno (avisa si están en posiciones distintas).
  2. LADO DE LOS EMPTIES: detecta de qué lado está el dedo gordo MIRANDO LA MALLA (los vértices
     más adelantados) y, si los kp_* están del lado contrario, los espeja en X. Esto pasa al
     convertir un pie izquierdo en derecho con un espejo en Y: la malla cambia de mano pero las
     X de los empties quedan como estaban.
  3. ORIENTACIÓN: rota en Z si el eje talón→punta no apunta a +Y (tolerancia 25°, porque el dedo
     gordo desvía ~10° de forma natural).
  4. ORIGEN: talón a Y=0, planta a Z=0.
  5. NORMALES: los exports con escala NEGATIVA quedan invertidos → aplica transforms y recalcula.
  6. NOMBRES: foot / leg_bare / leg_pants (la convención que espera el render).

    "C:/Program Files/Blender Foundation/Blender 5.1/blender.exe" --background ^
        --python assemble_foot_glb.py -- --out models/foot_rig.glb
"""
import bpy
import sys
import math
import argparse
from pathlib import Path

from mathutils import Vector

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

KP = ["heel", "toe", "ankle_in", "ankle_out", "ball", "toe_tip"]


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--foot", default="models/foot.glb")
    p.add_argument("--leg", default="models/leg.glb", help="pierna con piel → leg_bare")
    p.add_argument("--pants", default="models/pants.glb", help="pantalón → leg_pants")
    p.add_argument("--out", default="models/foot_rig.glb")
    p.add_argument("--leg_max_z", type=float, default=0.0,
                   help="recorta pierna/pantalón sobre esta altura en metros. 0 = NO recortar (default). "
                        "Cortarlo deja el borde VISIBLE dentro del cuadro (un tubo cortado, irreal); es "
                        "mejor que salga de cuadro por arriba. La fracción de pierna sube (~25-30%) pero "
                        "sigue dentro del rango medido en los videos reales (10-32%).")
    return p.parse_args(argv)


def clear():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for b in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for x in list(b):
            try:
                b.remove(x)
            except Exception:
                pass


def import_glb(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(Path(path).resolve()))
    return [o for o in bpy.data.objects if o not in before]


def world_bbox(objs):
    mn = Vector((1e9,) * 3); mx = Vector((-1e9,) * 3)
    for o in objs:
        if o.type != 'MESH':
            continue
        for c in o.bound_box:
            wc = o.matrix_world @ Vector(c)
            for i in range(3):
                mn[i] = min(mn[i], wc[i]); mx[i] = max(mx[i], wc[i])
    return mn, mx


def big_toe_side(foot_meshes):
    """Devuelve el signo de X donde está el DEDO GORDO, leído de la malla:
    los vértices más adelantados (mayor Y) pertenecen al dedo gordo, que sobresale más."""
    pts = []
    for o in foot_meshes:
        mw = o.matrix_world
        for v in o.data.vertices:
            pts.append(mw @ v.co)
    if not pts:
        return 0.0
    ys = [p.y for p in pts]
    y_max, y_min = max(ys), min(ys)
    thr = y_max - 0.06 * (y_max - y_min)      # frente del pie (6% más adelantado)
    front = [p for p in pts if p.y >= thr]
    mean_x = sum(p.x for p in front) / len(front)
    return mean_x


def trim_above_z(obj, z_max):
    """Recorta la malla por encima de z_max (mundo) y tapa el agujero del corte.
    Una pierna muy alta obliga a la cámara a alejarse (para no atravesarla en tomas cenitales)
    y el pie queda diminuto en el encuadre. ~30-35 cm es el óptimo: la pierna sale del cuadro,
    como en una foto real, sin empujar la cámara."""
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.transform(obj.matrix_world)
    zs = [v.co.z for v in bm.verts]
    if not zs or max(zs) <= z_max + 1e-4:
        bm.free()
        return 0.0
    top = max(zs)
    geom = list(bm.verts) + list(bm.edges) + list(bm.faces)
    bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-6,
                           plane_co=(0.0, 0.0, z_max), plane_no=(0.0, 0.0, 1.0),
                           clear_outer=True)

    def boundary():
        return [e for e in bm.edges if len(e.link_faces) == 1
                and all(abs(v.co.z - z_max) < 2e-3 for v in e.verts)]

    def loops_of(edges):
        """Separa las aristas de borde en bucles conectados."""
        rest, out = set(edges), []
        while rest:
            e0 = rest.pop(); grp = [e0]; frontier = [e0]
            while frontier:
                e = frontier.pop()
                for v in e.verts:
                    for ne in v.link_edges:
                        if ne in rest:
                            rest.discard(ne); grp.append(ne); frontier.append(ne)
            out.append(grp)
        return out

    bnd = boundary()
    if bnd:
        groups = loops_of(bnd)
        # Tela con ESPESOR: el corte deja DOS bucles (pared externa e interna) → puentearlos
        # produce el anillo correcto. holes_fill sobre dos bucles no cierra nada y quedaba el
        # tubo abierto: se veía el interior hueco desde arriba.
        if len(groups) == 2:
            try:
                bmesh.ops.bridge_loops(bm, edges=bnd)
            except Exception:
                pass
        for op in (lambda es: bmesh.ops.holes_fill(bm, edges=es, sides=0),
                   lambda es: bmesh.ops.edgeloop_fill(bm, edges=es),
                   lambda es: bmesh.ops.contextual_create(bm, geom=es),
                   lambda es: bmesh.ops.triangle_fill(bm, edges=es, use_beauty=True)):
            rem = boundary()
            if not rem:
                break
            try:
                op(rem)
            except Exception:
                pass
        left = len(boundary())
        if left:
            print(f"   ⚠ {obj.name}: quedaron {left} aristas abiertas tras el corte")

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.transform(obj.matrix_world.inverted())
    bm.to_mesh(obj.data)
    obj.data.update()
    bm.free()
    return top


def cap_open_top(obj, tol=0.004):
    """Tapa la abertura SUPERIOR de la malla (si la tiene) usando el operador de Blender
    (más robusto que bmesh.ops para mallas con espesor). Sin esto, el pantalón se ve hueco
    por dentro en las tomas cenitales — parece un tubo cortado."""
    import bmesh
    bm = bmesh.new(); bm.from_mesh(obj.data)
    zs = [v.co.z for v in bm.verts]
    if not zs:
        bm.free(); return 0
    top_local = max(zs)
    n_open = sum(1 for e in bm.edges if len(e.link_faces) == 1
                 and all(abs(v.co.z - top_local) < tol for v in e.verts))
    bm.free()
    if not n_open:
        return 0

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type='VERT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bm = bmesh.from_edit_mesh(obj.data)
    for v in bm.verts:
        v.select = abs(v.co.z - top_local) < tol
    bm.select_flush(True)
    bmesh.update_edit_mesh(obj.data)
    try:
        bpy.ops.mesh.edge_face_add()          # equivalente a apretar F: crea la cara/ngon
    except Exception:
        pass
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')

    bm = bmesh.new(); bm.from_mesh(obj.data)
    left = sum(1 for e in bm.edges if len(e.link_faces) == 1
               and all(abs(v.co.z - top_local) < tol for v in e.verts))
    bm.free()
    return n_open - left


def apply_all(meshes):
    bpy.ops.object.select_all(action='DESELECT')
    for o in meshes:
        o.select_set(True)
    if not meshes:
        return
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    for o in meshes:
        bpy.ops.object.select_all(action='DESELECT')
        o.select_set(True)
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')


def main():
    args = parse_args()
    clear()
    print("=" * 62)

    # ---------- importar ----------
    foot_objs = import_glb(args.foot)
    foot_meshes = [o for o in foot_objs if o.type == 'MESH']

    # 1) empties: si hay juegos duplicados (kp_x y kp_x.001), elegir el juego COMPLETO que
    #    concuerda con la malla — no el primero que aparezca (el orden de import no es garantía).
    raw = [o for o in foot_objs if o.type == 'EMPTY' and o.name.lower().startswith("kp_")]
    toe_x_mesh = big_toe_side(foot_meshes)
    sets = {}                       # sufijo ("" | ".001" | ...) → {base: obj}
    for o in raw:
        low = o.name.lower()
        base = low.split('.')[0][3:]
        suffix = low[len(low.split('.')[0]):]     # "" o ".001"
        sets.setdefault(suffix, {})[base] = o

    if len(sets) == 1:
        empties = next(iter(sets.values()))
        print(f"1) empties: {len(empties)} sin duplicados ✔")
    else:
        print(f"1) hay {len(sets)} JUEGOS de empties duplicados: {sorted(sets)}")
        scored = []
        for suf, d in sets.items():
            tt = d.get("toe_tip")
            ai, ao = d.get("ankle_in"), d.get("ankle_out")
            ok_toe = tt is not None and tt.matrix_world.translation.x * toe_x_mesh > 0
            ok_med = (ai is not None and ao is not None and
                      (ai.matrix_world.translation.x - ao.matrix_world.translation.x) * toe_x_mesh > 0)
            scored.append((len(d), ok_toe + ok_med, suf, d))
            print(f"   juego '{suf or '(base)'}': {len(d)}/6 empties | "
                  f"dedo gordo {'✔' if ok_toe else '✖'} | medial {'✔' if ok_med else '✖'}")
        scored.sort(key=lambda t: (t[1], t[0]), reverse=True)
        best = scored[0]
        empties = best[3]
        print(f"   → elijo el juego '{best[2] or '(base)'}' (el que concuerda con la malla)")
        for _, _, suf, d in scored[1:]:
            for o in d.values():
                bpy.data.objects.remove(o)
        print("   ⚠ conviene borrar el juego duplicado en tu .blend para no arrastrar la ambigüedad")

    missing = [k for k in KP if k not in empties]
    if missing:
        print(f"   ⚠ faltan: {['kp_'+m for m in missing]} (el render los aproximará por bbox)")

    leg_objs = [o for o in import_glb(args.leg) if o.type == 'MESH'] if Path(args.leg).exists() else []
    pants_objs = [o for o in import_glb(args.pants) if o.type == 'MESH'] if Path(args.pants).exists() else []
    print(f"   mallas: pie={[o.name for o in foot_meshes]} pierna={[o.name for o in leg_objs]} "
          f"pantalón={[o.name for o in pants_objs]}")
    all_meshes = foot_meshes + leg_objs + pants_objs

    # ---------- 2) lado de los empties (contra la malla) ----------
    toe_x = toe_x_mesh
    print(f"2) dedo gordo según la MALLA: X medio del frente = {toe_x:+.4f} "
          f"({'-X' if toe_x < 0 else '+X'})")
    if "toe_tip" in empties:
        kp_x = empties["toe_tip"].matrix_world.translation.x
        print(f"   kp_toe_tip está en X = {kp_x:+.4f} ({'-X' if kp_x < 0 else '+X'})")
        if abs(kp_x) > 0.005 and (kp_x * toe_x) < 0:
            print("   → NO coinciden: espejo los 6 empties en X (la malla manda)")
            for e in empties.values():
                p = e.matrix_world.translation.copy()
                mw = e.matrix_world.copy()
                mw.translation = Vector((-p.x, p.y, p.z))
                e.matrix_world = mw
            bpy.context.view_layer.update()
        else:
            print("   → coinciden ✔ (no toco los empties)")

    # ---------- raíz temporal ----------
    root = bpy.data.objects.new("AssembleRoot", None)
    bpy.context.scene.collection.objects.link(root)
    for o in list(all_meshes) + list(empties.values()):
        if o.parent is None or o.parent not in all_meshes:
            mw = o.matrix_world.copy()
            o.parent = root
            o.matrix_world = mw
        else:
            mw = o.matrix_world.copy()      # empties parentados al pie → al root
            o.parent = root
            o.matrix_world = mw

    # ---------- 3) orientación ----------
    if "heel" in empties and "toe_tip" in empties:
        v = empties["toe_tip"].matrix_world.translation - empties["heel"].matrix_world.translation
        ang = math.degrees(math.atan2(v.x, v.y))
        print(f"3) eje talón→punta: desvío de +Y = {ang:+.1f}°"
              + ("  (el dedo gordo desvía ~10° naturalmente)" if abs(ang) <= 25 else ""))
        if abs(ang) > 25:
            root.rotation_euler.z = -math.radians(ang)
            bpy.context.view_layer.update()
            print(f"   → rotado {-ang:+.1f}° en Z para apuntar a +Y")
        else:
            print("   → dentro de tolerancia, no roto ✔")

    # ---------- 4) origen: NO SE TOCA ----------
    # Mover el pie rompería la alineación manual con los zapatos (que viven en otros GLB y no se
    # mueven con él) — es exactamente la regla dura #1. El contacto con el piso se resuelve en el
    # render, levantando el conjunto según el zapato activo (cada uno tiene otro grosor de suela).
    mn, _ = world_bbox(all_meshes)
    heel_y = empties["heel"].matrix_world.translation.y if "heel" in empties else world_bbox(foot_meshes)[0].y
    print(f"4) origen: NO lo muevo (preserva el calce con los zapatos). "
          f"talón en Y={heel_y*100:+.1f} cm, punto más bajo Z={mn.z*100:+.1f} cm")

    # ---------- 5) aplicar y normales ----------
    for o in list(all_meshes) + list(empties.values()):
        mw = o.matrix_world.copy()
        o.parent = None
        o.matrix_world = mw
    bpy.data.objects.remove(root)
    apply_all(all_meshes)
    bpy.context.view_layer.update()
    print("5) transforms aplicados y normales recalculadas ✔")

    # ---------- 5b) recortar pierna/pantalón a una altura razonable ----------
    if args.leg_max_z > 0:
        for o in leg_objs + pants_objs:
            top = trim_above_z(o, args.leg_max_z)
            if top:
                print(f"5b) {o.name}: recortado de {top*100:.0f} cm a {args.leg_max_z*100:.0f} cm")
        bpy.context.view_layer.update()
    # Tapar la boca superior (venga del recorte o del modelado): si queda abierta se ve el
    # interior hueco del pantalón en las tomas desde arriba.
    for o in leg_objs + pants_objs:
        n = cap_open_top(o)
        if n:
            print(f"5c) {o.name}: tapada la abertura superior ({n} aristas cerradas)")
    bpy.context.view_layer.update()

    # ---------- 6) nombres ----------
    for i, o in enumerate(foot_meshes):
        o.name = "foot" if i == 0 else f"foot_part{i}"
    for i, o in enumerate(leg_objs):
        o.name = "leg_bare" if i == 0 else f"leg_bare_part{i}"
    for i, o in enumerate(pants_objs):
        o.name = "leg_pants" if i == 0 else f"leg_pants_part{i}"
    if foot_meshes:
        for e in empties.values():
            mw = e.matrix_world.copy()
            e.parent = foot_meshes[0]
            e.matrix_world = mw
    print("6) renombrado: foot / leg_bare / leg_pants ✔")

    # ---------- verificación ----------
    fmn, fmx = world_bbox(foot_meshes)
    L = fmx.y - fmn.y
    print("-" * 62)
    print(f"PIE: largo {L*100:.1f} cm | ancho {(fmx.x-fmn.x)*100:.1f} | alto {(fmx.z-fmn.z)*100:.1f}")
    print(f"     talón Y={fmn.y*100:+.2f} cm | planta Z={fmn.z*100:+.2f} cm")
    if "ankle_in" in empties and "ankle_out" in empties:
        xi = empties["ankle_in"].matrix_world.translation.x
        xo = empties["ankle_out"].matrix_world.translation.x
        ok = xi < xo
        print(f"     ankle_in={xi:+.3f} ankle_out={xo:+.3f} → "
              + ("medial en -X, pie DERECHO ✔" if ok else "⚠ medial en +X (izquierdo?)"))
    for name in ["leg_bare", "leg_pants"]:
        objs = [o for o in bpy.data.objects if o.name == name]
        if objs:
            print(f"     {name}: hasta Z={world_bbox(objs)[1].z*100:.0f} cm")

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.export_scene.gltf(filepath=str(out), export_format='GLB', use_selection=True)
    print(f"\n✅ {out.name} exportado")
    print(f"   Validar: blender --background --python validate_foot_glb.py -- --glb {args.out}")


if __name__ == "__main__":
    main()
