"""
PASO 0B — Render sintético v2 (Blender 5.1) — imagen + máscara multiclase + keypoints GT
========================================================================================
Genera por cada render:
  images/synth_XXXXX.jpg   ← color realista (con degradación tipo cámara móvil)
  masks/synth_XXXXX.png    ← máscara uint8 con índices {0 fondo, 1 pierna, 2 pie, 3 zapato}
  (+ una línea en keypoints.jsonl con los 6 keypoints 2D proyectados)
  (+ manifest.json con seed/zapato/hdri/piso por imagen)

Corrige los bugs del script anterior:
  - piso randomizado POR render (antes: una sola vez fuera del loop)
  - pie + zapato transformados JUNTOS vía un Empty raíz, reseteado cada iteración
    (antes: solo el pie → desalineación acumulativa)
  - SIN reescalar los GLB (regla dura #1): el tamaño aparente varía solo por distancia de cámara
  - máscara por materiales de emisión + color management 'Raw' (índices exactos), no doble render tonemapeado

Requiere Blender 5.1 (su Python trae numpy; NO trae PIL/cv2 → todo el post-proceso es numpy).
Uso:
  "C:/Program Files/Blender Foundation/Blender 5.1/blender.exe" --background --python 0b_blender_render.py -- \
      --foot models/foot.glb --shoe ../models/shoe.glb --out data_synthetic --count 5000 --seed 42
  # con carpeta de zapatos, HDRIs y texturas de piso:
  ... --shoe_dir models/shoes --hdri_dir hdri --floor_dir floor_textures
  # preview de 20 con contact-sheet:
  ... --preview
"""

import bpy
import sys
import math
import json
import random
import argparse
from pathlib import Path

import numpy as np
from mathutils import Vector, Euler
from bpy_extras.object_utils import world_to_camera_view

# ---- Clases de la máscara ----
CLASS_BG, CLASS_LEG, CLASS_FOOT, CLASS_SHOE = 0, 1, 2, 3
# Emisión por CANAL de color (pierna=R, pie=G, zapato=B): decodificar por argmax evita clases
# "fantasma" en los bordes con anti-aliasing (un blend fondo↔pie sigue siendo pie, no pierna).
MASK_COLORS = {CLASS_BG: (0, 0, 0), CLASS_LEG: (1, 0, 0), CLASS_FOOT: (0, 1, 0), CLASS_SHOE: (0, 0, 1)}

# ---- Keypoints (mismo orden que train_model.py KP_NAMES) ----
KP_NAMES = ["heel", "toe", "ankle_in", "ankle_out", "ball", "toe_tip"]

# ---- Configuración de cámara: (nombre, prob, elev_deg, azim_deg, dist_min_m, dist_max_m) ----
# Distancias ABSOLUTAS en metros, calibradas contra los videos reales del proyecto: ahí el zapato
# ocupa 15-18% del frame (medido sobre las máscaras de SAM2 de clip01/02/03). Con distancias
# proporcionales al radio el sujeto salía al 3% → domain gap grande.
# La elevación se corta en 85°: mirando exactamente a plomo la cámara queda sobre el eje de la
# pierna y hay que alejarla mucho para no atravesarla.
CAMERA_CONFIGS = [
    ("cenital",  0.35, (65, 85),  (0, 360),   0.42, 0.60),
    ("diagonal", 0.35, (40, 65),  (0, 360),   0.34, 0.55),
    ("frontal",  0.20, (15, 40),  (-40, 40),  0.34, 0.55),
    ("lateral",  0.10, (20, 50),  (75, 105),  0.32, 0.50),
]

FLOOR_COLORS = [
    (0.65, 0.50, 0.35), (0.35, 0.25, 0.15), (0.85, 0.85, 0.85), (0.50, 0.50, 0.50),
    (0.70, 0.60, 0.45), (0.20, 0.40, 0.20), (0.80, 0.75, 0.70), (0.15, 0.15, 0.15),
]

# ---- Paletas para randomizar PIEL y PANTALÓN en cada render (domain randomization) ----
# En sRGB (como se ven); se convierten a lineal para Blender. La piel se aplica al pie Y a
# leg_bare con el MISMO tono (si no, el tobillo se vería de otro color que la pantorrilla).
SKIN_SRGB = [
    "#F7D9C4", "#EEBFA0", "#E0AC8B", "#D9A47A", "#C68642",
    "#A9714B", "#8D5524", "#6B4226", "#503116", "#3A2214",
]
PANTS_SRGB = [
    "#23324F", "#33486E", "#4C6A94", "#7B93B8", "#8FA8C8",   # jeans oscuro→claro
    "#17181C", "#3A3D42", "#55585F", "#8A8D93",              # negro→gris
    "#C9B48A", "#A08A5E", "#4A5340", "#2F3A2A",              # caqui / oliva
    "#E3E0D8", "#B03A34", "#5A4632",                          # crema / rojo / marrón
]


def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def hex_to_linear(h, jitter=0.0, rng=None):
    """'#RRGGBB' → tupla lineal para Blender, con variación multiplicativa opcional."""
    h = h.lstrip("#")
    rgb = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    lin = [srgb_to_linear(v) for v in rgb]
    if jitter and rng:
        f = 1.0 + rng.uniform(-jitter, jitter)
        lin = [max(0.0, min(1.0, v * f)) for v in lin]
    return tuple(lin)


def set_base_color(meshes, color, roughness=None):
    """Pinta el Base Color (y opcionalmente Roughness) de todos los materiales de esas mallas."""
    seen = set()
    for o in meshes:
        if o.type != 'MESH':
            continue
        for m in o.data.materials:
            if not m or m.name in seen or not m.use_nodes:
                continue
            seen.add(m.name)
            for n in m.node_tree.nodes:
                if n.type == 'BSDF_PRINCIPLED':
                    n.inputs["Base Color"].default_value = (*color, 1.0)
                    if roughness is not None and "Roughness" in n.inputs:
                        n.inputs["Roughness"].default_value = roughness


def randomize_skin_and_fabric(foot_meshes, bare_meshes, pants_meshes, rng):
    """Piel: un tono por render, el MISMO para pie y pierna desnuda. Pantalón: tono independiente."""
    skin = hex_to_linear(rng.choice(SKIN_SRGB), jitter=0.10, rng=rng)
    set_base_color(list(foot_meshes) + list(bare_meshes), skin, roughness=rng.uniform(0.45, 0.75))
    fabric = hex_to_linear(rng.choice(PANTS_SRGB), jitter=0.12, rng=rng)
    set_base_color(list(pants_meshes), fabric, roughness=rng.uniform(0.75, 0.95))
    return skin, fabric


# =====================================================================
# ARGUMENTOS
# =====================================================================
def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--foot", default="models/foot.glb")
    p.add_argument("--shoe", default="../models/shoe.glb")
    p.add_argument("--shoe_dir", default=None, help="carpeta con GLB de calzado (uno aleatorio por render)")
    p.add_argument("--out", default="data_synthetic")
    p.add_argument("--count", type=int, default=5000)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hdri_dir", default=None, help="carpeta con .hdr/.exr (fallback: luces aleatorias)")
    p.add_argument("--floor_dir", default=None, help="carpeta con texturas de piso (fallback: colores planos)")
    p.add_argument("--preview", action="store_true", help="=--count 20 + contact-sheet HTML")
    p.add_argument("--foot_side", default="right", choices=["left", "right"],
                   help="lado del GLB de pie canónico (el otro lado lo genera el flip del training)")
    return p.parse_args(argv)


# =====================================================================
# ESCENA
# =====================================================================
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for b in list(block):
            try: block.remove(b)
            except Exception: pass


def import_glb(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    return [o for o in bpy.data.objects if o not in before]


def mesh_world_bounds(objs):
    """bbox mundial (min, max) de las mallas visibles; (None, None) si no hay ninguna."""
    mn = Vector((1e9,) * 3); mx = Vector((-1e9,) * 3)
    found = False
    for o in objs:
        if o.type != 'MESH' or o.hide_render:
            continue
        found = True
        for v in o.bound_box:
            wv = o.matrix_world @ Vector(v)
            for i in range(3):
                mn[i] = min(mn[i], wv[i]); mx[i] = max(mx[i], wv[i])
    return (mn, mx) if found else (None, None)


def mesh_world_radius(objs):
    max_r = 0.0
    for o in objs:
        if o.type != 'MESH':
            continue
        for v in o.bound_box:
            wv = o.matrix_world @ Vector(v)
            max_r = max(max_r, wv.length)
    return max_r or 1.0


# =====================================================================
# EMPTIES DE KEYPOINTS
# =====================================================================
def find_or_make_keypoints(foot_objs, parent):
    """Usa los Empties kp_* del GLB si existen; si no, los coloca por proporciones del bbox del pie."""
    existing = {o.name.lower(): o for o in bpy.data.objects if o.type == 'EMPTY'}
    kp_objs = []
    have_named = all(f"kp_{n}" in existing for n in KP_NAMES)
    if have_named:
        for n in KP_NAMES:
            e = existing[f"kp_{n}"]
            # Si el Empty YA viene parentado (p.ej. al mesh del pie, como recomienda la guía),
            # NO reparentar: cambiar el padre sin corregir matrix_parent_inverse lo desplazaría.
            # Su cadena de padres termina igual en Root, así que rota con el conjunto.
            if e.parent is None:
                e.parent = parent
                e.matrix_parent_inverse = parent.matrix_world.inverted()
            kp_objs.append(e)
        print("  [kp] usando Empties kp_* del GLB")
        return kp_objs

    # Fallback: bbox mundial del pie
    mn = Vector((1e9, 1e9, 1e9)); mx = Vector((-1e9, -1e9, -1e9))
    for o in foot_objs:
        if o.type != 'MESH':
            continue
        for v in o.bound_box:
            wv = o.matrix_world @ Vector(v)
            for i in range(3):
                mn[i] = min(mn[i], wv[i]); mx[i] = max(mx[i], wv[i])
    size = mx - mn
    # eje horizontal más largo = longitud del pie (heel→toe)
    long_axis = 0 if size.x >= size.y else 1
    def P(fl, fw, fh):
        p = Vector((0, 0, 0))
        p[long_axis]     = mn[long_axis] + fl * size[long_axis]
        p[1 - long_axis] = mn[1 - long_axis] + fw * size[1 - long_axis]
        p.z              = mn.z + fh * size.z
        return p
    frac = {  # (a lo largo, ancho, alto)
        "heel": (0.05, 0.5, 0.15), "toe": (0.95, 0.5, 0.15),
        "ankle_in": (0.15, 0.30, 0.9), "ankle_out": (0.15, 0.70, 0.9),
        "ball": (0.72, 0.5, 0.12), "toe_tip": (1.0, 0.5, 0.1),
    }
    for n in KP_NAMES:
        e = bpy.data.objects.new(f"kp_{n}", None)
        e.empty_display_size = 0.01
        bpy.context.scene.collection.objects.link(e)
        e.location = P(*frac[n])
        e.parent = parent
        e.matrix_parent_inverse = parent.matrix_world.inverted()
        kp_objs.append(e)
    print("  [kp] Empties generados por bbox (el GLB no traía kp_*)")
    return kp_objs


def project_keypoints(scene, cam, kp_objs):
    """Keypoints AMODALES: se etiquetan siempre que caigan DENTRO del frame, aunque el zapato
    (o el pantalón) los tape. Es intencional: el modelo debe inferir dónde está el pie DEBAJO del
    zapato — de ahí sale la pose 6DoF para calzarle el zapato virtual.
    (Antes había un ray_cast que ponía vis=0 al ocluirse: con el pie calzado anulaba casi todos
    los keypoints y le enseñaba al modelo a no predecir nada cuando ve un zapato. Bug corregido.)
    vis=0 sólo si el punto queda fuera de cuadro o detrás de la cámara."""
    kps = []
    for e in kp_objs:
        P = e.matrix_world.translation
        co = world_to_camera_view(scene, cam, P)   # x,y en [0,1] (0,0 abajo-izq), z profundidad
        x_img, y_img = co.x, 1.0 - co.y
        vis = 1.0 if (co.z > 0 and 0.0 <= co.x <= 1.0 and 0.0 <= co.y <= 1.0) else 0.0
        kps.append([round(float(x_img), 5), round(float(y_img), 5), float(vis)])
    return kps


# =====================================================================
# ILUMINACIÓN / PISO
# =====================================================================
def set_world_hdri(hdri_paths, rng):
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    bg = nt.nodes.new("ShaderNodeBackground")
    out = nt.nodes.new("ShaderNodeOutputWorld")
    if hdri_paths:
        env = nt.nodes.new("ShaderNodeTexEnvironment")
        mapping = nt.nodes.new("ShaderNodeMapping")
        texco = nt.nodes.new("ShaderNodeTexCoord")
        env.image = bpy.data.images.load(str(rng.choice(hdri_paths)), check_existing=True)
        mapping.inputs["Rotation"].default_value[2] = rng.uniform(0, 2 * math.pi)
        nt.links.new(texco.outputs["Generated"], mapping.inputs["Vector"])
        nt.links.new(mapping.outputs["Vector"], env.inputs["Vector"])
        nt.links.new(env.outputs["Color"], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = rng.uniform(0.4, 1.4)
    else:
        bg.inputs["Color"].default_value = (rng.uniform(0.05, 0.4),) * 3 + (1,)
        bg.inputs["Strength"].default_value = rng.uniform(0.2, 0.6)
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])


def add_random_lights(rng):
    for o in [o for o in bpy.data.objects if o.type == 'LIGHT']:
        bpy.data.objects.remove(o)
    kind = rng.choice(['SUN', 'AREA', 'MIXED'])
    if kind in ('SUN', 'MIXED'):
        bpy.ops.object.light_add(type='SUN')
        s = bpy.context.object
        s.data.energy = rng.uniform(1.5, 5.0)
        s.rotation_euler = Euler((math.radians(rng.uniform(30, 80)), 0, math.radians(rng.uniform(0, 360))))
    if kind in ('AREA', 'MIXED'):
        bpy.ops.object.light_add(type='AREA', location=(0, 0, rng.uniform(0.8, 1.5)))
        a = bpy.context.object
        a.data.energy = rng.uniform(150, 700); a.data.size = rng.uniform(0.3, 1.0)


def make_floor(floor_paths, rng):
    old = bpy.data.objects.get("Floor")
    if old:
        bpy.data.objects.remove(old)
    bpy.ops.mesh.primitive_plane_add(size=6.0, location=(0, 0, 0))
    floor = bpy.context.object; floor.name = "Floor"
    mat = bpy.data.materials.new("FloorMat"); mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = rng.uniform(0.3, 0.9)
    if floor_paths:
        nt = mat.node_tree
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(str(rng.choice(floor_paths)), check_existing=True)
        mapping = nt.nodes.new("ShaderNodeMapping")
        texco = nt.nodes.new("ShaderNodeTexCoord")
        sc = rng.uniform(1.0, 4.0)
        mapping.inputs["Scale"].default_value = (sc, sc, sc)
        mapping.inputs["Rotation"].default_value[2] = rng.uniform(0, 2 * math.pi)
        nt.links.new(texco.outputs["UV"], mapping.inputs["Vector"])
        nt.links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        c = list(rng.choice(FLOOR_COLORS))
        c = [max(0, min(1, v + rng.uniform(-0.1, 0.1))) for v in c]
        bsdf.inputs["Base Color"].default_value = (*c, 1)
    floor.data.materials.append(mat)
    return floor


# =====================================================================
# CÁMARA
# =====================================================================
def place_camera(radius, rng, leg_top=None, leg_radius=None, aim_z=None):
    """Distancia ABSOLUTA en metros (ver CAMERA_CONFIGS), con una garantía geométrica: la cámara
    nunca queda dentro de la pierna. Es seguro si está POR ENCIMA del tope de la pierna o bien
    lo bastante afuera de su eje; alcanza cumplir la MENOS exigente de las dos:
        d ≥ min( leg_top/sin(elev) , (leg_radius+3cm)/cos(elev) )
    Así una toma diagonal puede acercarse mucho (la cámara pasa al costado de la pierna) y sólo
    las casi-cenitales necesitan алejarse. Antes se exigía salir de la esfera de TODA la escena,
    lo que empujaba la cámara a ~0.8 m y dejaba el zapato al 3% del frame."""
    cfg = rng.choices(CAMERA_CONFIGS, weights=[c[1] for c in CAMERA_CONFIGS])[0]
    name, _, elev_r, azim_r, dmin, dmax = cfg
    elev = math.radians(rng.uniform(*elev_r))
    azim = math.radians(rng.uniform(*azim_r))
    dist = rng.uniform(dmin, dmax)
    if leg_top:
        se, ce = math.sin(elev), max(math.cos(elev), 1e-3)
        d_safe = min(leg_top / max(se, 1e-3), ((leg_radius or 0.08) + 0.03) / ce)
        dist = max(dist, d_safe)
    loc = Vector((dist * math.cos(elev) * math.sin(azim),
                  dist * math.cos(elev) * math.cos(azim),
                  dist * math.sin(elev)))
    cam = bpy.data.objects.get("Camera")
    if cam is None:
        cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
        bpy.context.scene.collection.objects.link(cam)
    cam.location = loc
    # apuntar al SUJETO (centro del pie/zapato), no al origen del mundo: el pie no está centrado
    # en el origen y si no se apunta ahí queda descentrado o fuera de cuadro.
    target = aim_z if aim_z is not None else Vector((0, 0, radius * 0.2))
    cam.rotation_euler = (target - loc).to_track_quat('-Z', 'Y').to_euler()
    cam.data.lens = rng.uniform(24, 34)          # rango de un teléfono (~26 mm equiv.)
    cam.data.clip_start = 0.01                  # evita recortes en primeros planos
    bpy.context.scene.camera = cam
    return name


# =====================================================================
# MÁSCARA (materiales de emisión + Raw)
# =====================================================================
def emission_mat(rgb):
    m = bpy.data.materials.new("MaskMat"); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    e = nt.nodes.new("ShaderNodeEmission")
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    e.inputs["Color"].default_value = (*rgb, 1)
    e.inputs["Strength"].default_value = 1.0
    nt.links.new(e.outputs[0], o.inputs[0])
    return m


def set_color_management(scene, transform):
    scene.render.image_settings.color_management = 'OVERRIDE'
    scene.render.image_settings.view_settings.view_transform = transform


# =====================================================================
# NUMPY I/O (Blender trae numpy; no PIL/cv2)
# =====================================================================
def np_load(path, non_color=True):
    img = bpy.data.images.load(str(path), check_existing=False)
    if non_color:
        img.colorspace_settings.name = 'Non-Color'
    w, h = img.size
    arr = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
    arr = np.flipud(arr)  # bottom-up → top-down
    bpy.data.images.remove(img)
    return arr[:, :, :3]  # 0..1


def np_save(arr01, path, file_format, scene, quality=90):
    """arr01: (H,W,3) float 0..1 top-down. Guarda en bytes exactos (Raw)."""
    h, w = arr01.shape[:2]
    rgba = np.dstack([arr01, np.ones((h, w, 1), np.float32)])
    rgba = np.flipud(rgba)
    img = bpy.data.images.new("tmp_out", w, h, alpha=False)
    img.colorspace_settings.name = 'Non-Color'
    img.pixels[:] = rgba.ravel()
    img.file_format = file_format
    scene.render.image_settings.file_format = file_format
    if file_format == 'JPEG':
        scene.render.image_settings.quality = int(quality)
    set_color_management(scene, 'Raw')
    img.save_render(str(path), scene=scene)
    bpy.data.images.remove(img)


# =====================================================================
# DEGRADACIÓN TIPO CÁMARA MÓVIL (numpy)
# =====================================================================
def degrade(rgb01, rng):
    arr = rgb01.copy() * 255.0
    h, w = arr.shape[:2]
    if rng.random() < 0.5:  # balance de blancos
        arr *= (1 + (np.array([rng.uniform(-1, 1) for _ in range(3)])) * 0.10)
    if rng.random() < 0.3:  # motion blur (numpy, promedio desplazado)
        k = rng.randint(3, 9); ang = rng.uniform(0, math.pi)
        dx, dy = math.cos(ang), math.sin(ang)
        acc = np.zeros_like(arr); n = 0
        for t in np.linspace(-k / 2, k / 2, k):
            acc += np.roll(np.roll(arr, int(round(dy * t)), axis=0), int(round(dx * t)), axis=1); n += 1
        arr = acc / n
    arr += np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(1, 4), arr.shape)  # ruido
    if rng.random() < 0.3:  # viñeteo
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / math.sqrt((w / 2) ** 2 + (h / 2) ** 2)
        arr *= (1 - rng.uniform(0.2, 0.5) * (r ** 2))[..., None]
    return np.clip(arr, 0, 255) / 255.0


# =====================================================================
# RENDER DE UN PAR
# =====================================================================
def render_one(idx, scene, foot_objs, shoe_variants, leg_objs, root, kp_objs,
               out_dir, size, rng, hdri_paths, floor_paths):
    # zapato aleatorio visible
    chosen_shoe = rng.choice(list(shoe_variants.keys())) if shoe_variants else None
    for name, objs in shoe_variants.items():
        for o in objs:
            o.hide_render = (name != chosen_shoe)
    active_shoe = shoe_variants.get(chosen_shoe, [])

    # pierna aleatoria (si hay varias colecciones leg_*)
    active_leg = leg_objs
    if isinstance(leg_objs, dict):
        legname = rng.choice(list(leg_objs.keys())) if leg_objs else None
        for nm, objs in leg_objs.items():
            for o in objs:
                o.hide_render = (nm != legname)
        active_leg = leg_objs.get(legname, [])

    # colores: tono de piel por render (mismo para pie y pierna desnuda) + tela del pantalón
    bare_meshes = []
    pants_meshes = []
    if isinstance(leg_objs, dict):
        for nm, objs in leg_objs.items():
            (pants_meshes if "pants" in nm else bare_meshes).extend(objs)
    skin, fabric = randomize_skin_and_fabric(foot_objs, bare_meshes, pants_meshes, rng)

    # transformar el conjunto (rotación Z aleatoria) SIN reescalar
    root.rotation_euler = Euler((0, 0, rng.uniform(0, 2 * math.pi)))
    root.location.z = 0.0
    bpy.context.view_layer.update()

    # CONTACTO CON EL PISO: levantar el conjunto para que la suela del zapato ACTIVO toque Z=0.
    # Cada zapato tiene su grosor de suela, y el pie va por dentro (más arriba) — de ahí que esto
    # se resuelva acá y no moviendo los GLB (eso rompería el calce hecho a mano en Blender).
    visible = list(foot_objs) + list(active_shoe) + list(active_leg)
    mn_all, mx_all = mesh_world_bounds(visible)
    if mn_all is not None:
        root.location.z = -mn_all.z
        bpy.context.view_layer.update()
        mn_all, mx_all = mesh_world_bounds(visible)

    # encuadre: centro y radio del SUJETO (pie+zapato), y datos de la pierna para la seguridad
    smn, smx = mesh_world_bounds(list(foot_objs) + list(active_shoe))
    center = (smn + smx) / 2 if smn is not None else Vector((0, 0, 0))
    radius = max((smx - smn)) / 2 if smn is not None else 0.2
    lmn, lmx = mesh_world_bounds(list(active_leg))
    leg_top = lmx.z if lmx is not None else None
    leg_radius = (max(lmx.x - lmn.x, lmx.y - lmn.y) / 2) if lmn is not None else None
    angle_name = place_camera(radius, rng, leg_top, leg_radius, aim_z=center)

    # --- COLOR ---
    if hdri_paths:
        set_world_hdri(hdri_paths, rng)
    else:
        set_world_hdri(None, rng); add_random_lights(rng)
    floor = make_floor(floor_paths, rng)
    scene.render.resolution_x = scene.render.resolution_y = size
    scene.cycles.samples = 32
    set_color_management(scene, 'AgX')
    tmp_color = out_dir / "_tmp_color.png"
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = str(tmp_color)
    bpy.ops.render.render(write_still=True)

    # keypoints (geometría, independiente de materiales)
    kps = project_keypoints(scene, scene.camera, kp_objs)

    # degradar color → JPEG
    color = np_load(tmp_color, non_color=True)
    color = degrade(color, rng)
    np_save(color, out_dir / "images" / f"synth_{idx:05d}.jpg", 'JPEG', scene,
            quality=rng.randint(40, 85))
    tmp_color.unlink(missing_ok=True)

    # --- MÁSCARA ---
    orig = {o.name: list(o.data.materials) for o in bpy.data.objects if o.type == 'MESH'}
    def assign(objs, cls):
        m = emission_mat(MASK_COLORS[cls])
        for o in objs:
            if o.type != 'MESH':
                continue
            o.data.materials.clear(); o.data.materials.append(m)
    assign([floor], CLASS_BG)
    assign(active_leg, CLASS_LEG)
    assign(foot_objs, CLASS_FOOT)
    assign(active_shoe, CLASS_SHOE)

    prev_filter = scene.cycles.pixel_filter_type
    prev_width = scene.cycles.filter_width
    scene.cycles.pixel_filter_type = 'BOX'
    scene.cycles.filter_width = 0.01
    scene.cycles.samples = 1
    world_bg = scene.world.node_tree.nodes.get("Background")
    if world_bg:
        world_bg.inputs["Strength"].default_value = 0.0  # fondo negro = clase 0
    set_color_management(scene, 'Raw')
    tmp_mask = out_dir / "_tmp_mask.png"
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = str(tmp_mask)
    bpy.ops.render.render(write_still=True)

    mask_rgb = np_load(tmp_mask, non_color=True)      # (H,W,3) en 0..1: R=pierna, G=pie, B=zapato
    maxc = mask_rgb.max(axis=2)
    idx_mask = np.where(maxc < 0.25, 0, mask_rgb.argmax(axis=2) + 1).astype(np.float32)
    np_save(np.repeat((idx_mask / 255.0)[..., None], 3, axis=2),
            out_dir / "masks" / f"synth_{idx:05d}.png", 'PNG', scene)
    tmp_mask.unlink(missing_ok=True)

    # restaurar materiales y filtro
    for o in bpy.data.objects:
        if o.type == 'MESH' and o.name in orig:
            o.data.materials.clear()
            for m in orig[o.name]:
                o.data.materials.append(m)
    scene.cycles.pixel_filter_type = prev_filter
    scene.cycles.filter_width = prev_width

    return {"file": f"synth_{idx:05d}.jpg", "kps": kps,
            "angle": angle_name, "shoe": chosen_shoe,
            "leg": (legname if isinstance(leg_objs, dict) else None),
            "skin": [round(c, 4) for c in skin], "fabric": [round(c, 4) for c in fabric]}


# =====================================================================
# PREVIEW CONTACT-SHEET
# =====================================================================
def write_contact_sheet(out_dir, records):
    rows = []
    for r in records:
        stem = Path(r["file"]).stem
        kp_txt = " ".join(f"{n}:{'✓' if k[2] > 0 else '·'}" for n, k in zip(KP_NAMES, r["kps"]))
        rows.append(
            f'<div class=c><div class=imgs>'
            f'<img src="images/{stem}.jpg"><img src="masks/{stem}.png" class=mask></div>'
            f'<div class=meta>{stem} · {r["angle"]} · {r.get("shoe") or "-"}<br>{kp_txt}</div></div>')
    html = ("<!doctype html><meta charset=utf-8><title>preview</title>"
            "<style>body{background:#111;color:#ccc;font-family:sans-serif}"
            ".c{display:inline-block;margin:6px;vertical-align:top}"
            ".imgs img{width:180px;height:180px;object-fit:contain;background:#000;image-rendering:pixelated}"
            ".mask{filter:brightness(60)}.meta{font-size:11px;width:360px}</style>"
            "<h3>Contact sheet — revisá alineación pie/zapato, máscara y keypoints</h3>"
            + "".join(rows))
    (out_dir / "preview.html").write_text(html, encoding="utf-8")
    print(f"  📄 preview.html")


# =====================================================================
# MAIN
# =====================================================================
def main():
    args = parse_args()
    if args.preview:
        args.count = 20
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # ABSOLUTA: Blender resuelve rutas relativas de render desde su propio CWD, no el de Python.
    out_dir = Path(args.out).resolve()
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.resolution_percentage = 100

    hdri_paths = []
    if args.hdri_dir and Path(args.hdri_dir).is_dir():
        hdri_paths = [p for e in ("*.hdr", "*.exr") for p in Path(args.hdri_dir).glob(e)]
    floor_paths = []
    if args.floor_dir and Path(args.floor_dir).is_dir():
        floor_paths = [p for e in ("*.jpg", "*.png", "*.jpeg") for p in Path(args.floor_dir).glob(e)]
    print(f"HDRIs: {len(hdri_paths)} | texturas de piso: {len(floor_paths)}")

    clear_scene()

    # Empty raíz: pie + zapatos cuelgan de acá y se transforman JUNTOS
    root = bpy.data.objects.new("Root", None)
    scene.collection.objects.link(root)

    foot_objs = import_glb(args.foot)
    print(f"Pie: {[o.name for o in foot_objs]}")
    for o in foot_objs:
        if o.parent is None:
            o.parent = root

    # pierna: variantes por NOMBRE DE OBJETO leg_* (las colecciones NO sobreviven el export a GLB).
    # Agrupa por nombre base sin sufijo numérico de Blender (leg_bare.001 → leg_bare) e incluye
    # los hijos MESH de un objeto/empty que matchee (permite parentar la tela bajo un empty leg_pants_dark).
    def _descendant_meshes(o):
        out = [o] if o.type == 'MESH' else []
        for ch in o.children:
            out += _descendant_meshes(ch)
        return out

    leg_groups = {}
    for o in foot_objs:
        base = o.name.lower().split('.')[0]
        if base.startswith("leg_"):
            leg_groups.setdefault(base, [])
            for m in _descendant_meshes(o):
                if m not in leg_groups[base]:
                    leg_groups[base].append(m)
    # fallback: colecciones (solo útil corriendo dentro de un .blend, no tras importar GLB)
    if not leg_groups:
        for coll in bpy.data.collections:
            if coll.name.lower().startswith("leg_"):
                leg_groups[coll.name.lower()] = [o for o in coll.objects if o.type == 'MESH']
    # los meshes de pierna NO son parte del "pie" para keypoints/máscara de pie
    leg_meshes = {m for objs in leg_groups.values() for m in objs}
    foot_objs = [o for o in foot_objs if o not in leg_meshes]
    leg_objs = leg_groups if leg_groups else []
    print(f"Pierna: {list(leg_groups.keys()) or 'ningún objeto leg_* (máscara sin pierna)'}")

    # zapatos: --shoe_dir (varios) o --shoe (uno). Cada GLB YA viene alineado al pie (regla #1).
    shoe_variants = {}
    shoe_files = []
    if args.shoe_dir and Path(args.shoe_dir).is_dir():
        shoe_files = sorted(list(Path(args.shoe_dir).glob("*.glb")) + list(Path(args.shoe_dir).glob("*.gltf")))
    elif Path(args.shoe).exists():
        shoe_files = [Path(args.shoe)]
    for sf in shoe_files:
        objs = import_glb(sf)
        for o in objs:
            if o.parent is None:
                o.parent = root
        shoe_variants[sf.stem] = objs
    print(f"Zapatos: {list(shoe_variants.keys()) or 'ninguno'}")

    kp_objs = find_or_make_keypoints(foot_objs, root)

    records = []
    for i in range(args.count):
        rec = render_one(i, scene, foot_objs, shoe_variants, leg_objs, root, kp_objs,
                         out_dir, args.size, rng, hdri_paths, floor_paths)
        records.append(rec)
        if (i + 1) % 25 == 0 or args.preview:
            print(f"  [{i+1}/{args.count}] {rec['file']} ({rec['angle']}, {rec.get('shoe')})")

    # keypoints.jsonl
    with open(out_dir / "keypoints.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            # formato v2 por lado: el GLB canónico es de UN lado (--foot_side);
            # el training genera el lado contrario con flip_lr (swap de bloques)
            f.write(json.dumps({"file": r["file"], "kps": {args.foot_side: r["kps"]}}) + "\n")
    # manifest.json
    (out_dir / "manifest.json").write_text(json.dumps({
        "seed": args.seed, "count": args.count, "size": args.size,
        "shoes": list(shoe_variants.keys()), "hdris": len(hdri_paths), "floors": len(floor_paths),
        "items": records}, indent=1), encoding="utf-8")

    if args.preview:
        write_contact_sheet(out_dir, records)

    print(f"\n✅ {len(records)} pares en {out_dir}/  (+ keypoints.jsonl, manifest.json)")


if __name__ == "__main__":
    main()
