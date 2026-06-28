"""
PASO 0B — Renderizado sintético con Blender (domain randomization)
===================================================================
Genera imágenes sintéticas del pie 3D con variaciones automáticas:
- Ángulo de cámara (cenital, 45°, frontal, lateral)
- Iluminación (HDRI, puntual, mixta)
- Textura de piso (madera, cerámico, alfombra, exterior)
- Escala y posición del pie
- Opcionalmente: poner el zapato shoe.glb encima del pie

Cada render genera AUTOMÁTICAMENTE:
  - imagen_XXXX.jpg  ← imagen realista
  - imagen_XXXX.png  ← máscara perfecta (sin necesidad de SAM)

Requisitos:
    Blender 3.6+ instalado

Uso desde línea de comandos (sin abrir Blender UI):
    "C:/Program Files/Blender Foundation/Blender 3.6/blender.exe" ^
        --background --python 0b_blender_render.py -- ^
        --foot models/foot.glb --shoe ../models/shoe.glb ^
        --out data_synthetic/ --count 5000

O desde el Scripting panel dentro de Blender:
    Abrir Blender → Scripting → Pegar este script → Run Script
"""

import bpy
import math
import random
import os
import sys
import argparse
from pathlib import Path
from mathutils import Vector, Euler

# ---- Leer argumentos (después de "--") ----
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

parser = argparse.ArgumentParser()
parser.add_argument("--foot",  default="models/foot.glb",    help="Modelo 3D del pie (.glb/.obj)")
parser.add_argument("--shoe",  default="../models/shoe.glb",  help="Modelo 3D del zapato (opcional)")
parser.add_argument("--out",   default="data_synthetic",      help="Carpeta de salida")
parser.add_argument("--count", type=int, default=5000,        help="Cuántas imágenes generar")
parser.add_argument("--size",  type=int, default=256,         help="Resolución de salida")
args = parser.parse_args(argv)

OUT_DIR = Path(args.out)
(OUT_DIR / "images").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "masks").mkdir(parents=True, exist_ok=True)

# =====================================================================
# UTILIDADES DE ESCENA
# =====================================================================

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for col in bpy.data.collections:
        bpy.data.collections.remove(col)

def import_glb(path):
    bpy.ops.import_scene.gltf(filepath=str(path))
    return bpy.context.selected_objects

def set_render_settings(size, samples_color=32, samples_mask=1):
    bpy.context.scene.render.engine          = 'CYCLES'
    bpy.context.scene.cycles.samples         = samples_color
    bpy.context.scene.render.resolution_x    = size
    bpy.context.scene.render.resolution_y    = size
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.file_format = 'JPEG'
    bpy.context.scene.render.image_settings.quality     = 90

# =====================================================================
# RANDOMIZACIÓN DE CÁMARA
# =====================================================================

CAMERA_CONFIGS = [
    # (nombre, prob, rango_elevation_deg, rango_azimuth_deg, dist_min, dist_max)
    ("cenital",   0.40, (75, 90),   (0, 360),  0.40, 0.90),  # Desde arriba
    ("diagonal",  0.30, (40, 75),   (0, 360),  0.50, 1.00),  # 45°
    ("frontal",   0.20, (15, 40),   (-30, 30), 0.60, 1.10),  # Desde el frente
    ("lateral",   0.10, (20, 50),   (80, 100), 0.55, 0.95),  # Desde el lado
]

def random_camera():
    """Elige configuración de cámara aleatoriamente con pesos."""
    weights = [c[1] for c in CAMERA_CONFIGS]
    config  = random.choices(CAMERA_CONFIGS, weights=weights)[0]
    _, _, elev_range, azim_range, dmin, dmax = config

    elevation = math.radians(random.uniform(*elev_range))
    azimuth   = math.radians(random.uniform(*azim_range))
    distance  = random.uniform(dmin, dmax)

    # Convertir coordenadas esféricas → cartesianas
    x = distance * math.cos(elevation) * math.sin(azimuth)
    y = distance * math.cos(elevation) * math.cos(azimuth)
    z = distance * math.sin(elevation)

    return Vector((x, y, z)), config[0]

# =====================================================================
# ILUMINACIÓN ALEATORIA
# =====================================================================

def setup_random_lighting():
    # Eliminar luces existentes
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj)

    light_type = random.choice(['SUN', 'POINT', 'AREA', 'MIXED'])

    if light_type in ('SUN', 'MIXED'):
        bpy.ops.object.light_add(type='SUN')
        sun = bpy.context.object
        sun.data.energy = random.uniform(1.5, 5.0)
        sun.rotation_euler = Euler((
            math.radians(random.uniform(30, 80)),
            math.radians(random.uniform(-30, 30)),
            math.radians(random.uniform(0, 360))
        ))

    if light_type in ('POINT', 'MIXED'):
        bpy.ops.object.light_add(type='POINT',
            location=(random.uniform(-0.5, 0.5),
                      random.uniform(-0.5, 0.5),
                      random.uniform(0.5, 1.5)))
        point = bpy.context.object
        point.data.energy = random.uniform(50, 200)
        point.data.color  = (
            random.uniform(0.9, 1.0),
            random.uniform(0.8, 1.0),
            random.uniform(0.7, 1.0),
        )

    if light_type == 'AREA':
        bpy.ops.object.light_add(type='AREA',
            location=(0, 0, random.uniform(0.8, 1.5)))
        area = bpy.context.object
        area.data.energy = random.uniform(200, 800)
        area.data.size   = random.uniform(0.3, 1.0)

    # Ambiente (simula HDRI con color aleatorio)
    world = bpy.data.worlds['World']
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    bg.inputs[0].default_value = (
        random.uniform(0.05, 0.4),
        random.uniform(0.05, 0.4),
        random.uniform(0.05, 0.4),
        1
    )
    bg.inputs[1].default_value = random.uniform(0.1, 0.5)

# =====================================================================
# PISO ALEATORIO
# =====================================================================

FLOOR_COLORS = [
    (0.65, 0.50, 0.35),  # madera clara
    (0.35, 0.25, 0.15),  # madera oscura
    (0.85, 0.85, 0.85),  # cerámico blanco
    (0.50, 0.50, 0.50),  # concreto gris
    (0.70, 0.60, 0.45),  # parquet
    (0.20, 0.40, 0.20),  # pasto / exterior
    (0.80, 0.75, 0.70),  # alfombra beige
    (0.15, 0.15, 0.15),  # suelo oscuro
]

def create_floor():
    bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0, 0, 0))
    floor = bpy.context.object
    floor.name = "Floor"

    mat = bpy.data.materials.new("FloorMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]

    color = random.choice(FLOOR_COLORS)
    # Variación de color ±10%
    color = tuple(max(0, min(1, c + random.uniform(-0.1, 0.1))) for c in color)
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Roughness"].default_value  = random.uniform(0.3, 0.9)
    bsdf.inputs["Specular"].default_value   = random.uniform(0.0, 0.3)

    floor.data.materials.append(mat)
    return floor

# =====================================================================
# MATERIAL PARA MÁSCARA (pie = blanco, todo lo demás = negro)
# =====================================================================

def create_mask_material(for_foot=True):
    mat = bpy.data.materials.new("MaskMat")
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    emit = mat.node_tree.nodes.new('ShaderNodeEmission')
    out  = mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
    emit.inputs["Color"].default_value = (1, 1, 1, 1) if for_foot else (0, 0, 0, 1)
    emit.inputs["Strength"].default_value = 1.0
    mat.node_tree.links.new(emit.outputs[0], out.inputs[0])
    return mat

# =====================================================================
# RENDER PRINCIPAL
# =====================================================================

def render_pair(index, foot_objects, shoe_objects=None, size=256):
    """Renderiza: imagen a color + máscara binaria."""
    out_base = str(OUT_DIR / "images" / f"synth_{index:05d}")
    msk_base = str(OUT_DIR / "masks"  / f"synth_{index:05d}")

    # Configurar cámara
    cam_pos, angle_name = random_camera()
    if "Camera" not in bpy.data.objects:
        bpy.ops.object.camera_add()
    cam = bpy.data.objects["Camera"]
    cam.location = cam_pos
    # Apuntar la cámara al origen (donde está el pie)
    direction = Vector((0, 0, 0)) - cam_pos
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    cam.data.lens = random.uniform(18, 35)  # focal length mm
    bpy.context.scene.camera = cam

    # Randomización del pie (posición/rotación/escala leve)
    for obj in foot_objects:
        obj.rotation_euler.z += random.uniform(-0.3, 0.3)  # rotación aleatoria
        scale = random.uniform(0.85, 1.15)
        obj.scale = (scale, scale, scale)

    # --- RENDER COLOR ---
    setup_random_lighting()
    bpy.context.scene.render.filepath = out_base
    bpy.context.scene.cycles.samples  = 32
    bpy.ops.render.render(write_still=True)

    # --- RENDER MÁSCARA ---
    # Guardar materiales originales
    orig_materials = {}
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            orig_materials[obj] = list(obj.data.materials)

    # Asignar materiales de máscara
    mask_foot  = create_mask_material(for_foot=True)
    mask_black = create_mask_material(for_foot=False)

    for obj in bpy.data.objects:
        if obj.type != 'MESH': continue
        obj.data.materials.clear()
        is_foot = obj in foot_objects or (shoe_objects and obj in shoe_objects)
        obj.data.materials.append(mask_foot if is_foot else mask_black)

    # Render máscara sin iluminación (solo emisión)
    bpy.context.scene.cycles.samples = 1
    bpy.context.scene.render.filepath = msk_base
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.ops.render.render(write_still=True)
    bpy.context.scene.render.image_settings.file_format = 'JPEG'

    # Restaurar materiales originales
    for obj, mats in orig_materials.items():
        obj.data.materials.clear()
        for m in mats:
            obj.data.materials.append(m)

    return angle_name

# =====================================================================
# MAIN
# =====================================================================

print(f"\n[Blender Render] Generando {args.count} imágenes sintéticas...")
print(f"  Pie:    {args.foot}")
print(f"  Zapato: {args.shoe}")
print(f"  Salida: {args.out}\n")

set_render_settings(args.size)
clear_scene()
create_floor()

# Cargar modelo de pie
foot_objs = import_glb(Path(args.foot))
print(f"  Pie cargado: {[o.name for o in foot_objs]}")

# Cargar zapato (opcional — se coloca encima del pie)
shoe_objs = []
if Path(args.shoe).exists():
    shoe_objs = import_glb(Path(args.shoe))
    print(f"  Zapato cargado: {[o.name for o in shoe_objs]}")
    # Ajustar posición del zapato sobre el pie
    for obj in shoe_objs:
        obj.location.z += 0.02  # ligeramente sobre el pie

# Contador por ángulo
angle_counts = {}

for i in range(args.count):
    angle = render_pair(i, foot_objs, shoe_objs, args.size)
    angle_counts[angle] = angle_counts.get(angle, 0) + 1

    if (i + 1) % 100 == 0:
        print(f"  [{i+1}/{args.count}] Distribución: {angle_counts}")

print(f"\n✅ {args.count} pares imagen+máscara en {args.out}/")
print(f"   Distribución de ángulos: {angle_counts}")
print(f"   Siguiente paso: python 1_augment.py --data_dir {args.out}")
