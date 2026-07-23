"""
Genera 1 render por ángulo (4 total) — carga foot.glb + shoe.glb tal cual.
Salida: training/previews/cenital.jpg, diagonal.jpg, frontal.jpg, lateral.jpg
"""

import bpy, math
from pathlib import Path

FOOT_GLB  = "C:/Users/Iga/Documents/ar-shoes/training/models/foot.glb"
SHOE_GLB  = "C:/Users/Iga/Documents/ar-shoes/models/shoe.glb"
OUT_DIR   = Path("C:/Users/Iga/Documents/ar-shoes/training/previews")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SIZE = 512

# ---- Limpiar ----
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# ---- Render settings ----
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 48
scene.render.resolution_x = SIZE
scene.render.resolution_y = SIZE
scene.render.resolution_percentage = 100

# ---- Piso ----
def make_floor(color=(0.62, 0.46, 0.30)):
    bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0, 0, 0))
    f = bpy.context.object; f.name = "Floor"
    m = bpy.data.materials.new("FloorMat"); m.use_nodes = True
    m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*color, 1)
    m.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value  = 0.7
    f.data.materials.append(m)
    return f

# ---- Cargar GLB sin modificaciones ----
def load_glb(path, label):
    p = Path(path)
    if not p.exists():
        print(f"[SKIP] {label} no encontrado: {path}")
        return []
    bpy.ops.import_scene.gltf(filepath=str(p))
    objs = list(bpy.context.selected_objects)
    print(f"[OK] {label}: {[o.name for o in objs]}")
    return objs

# ---- Iluminación ----
def setup_lights(sun_angle_deg=60, sun_rot_deg=30):
    for obj in [o for o in bpy.data.objects if o.type == 'LIGHT']:
        bpy.data.objects.remove(obj)
    bpy.ops.object.light_add(type='SUN', location=(1,-1,2))
    sun = bpy.context.object
    sun.data.energy = 3.5
    sun.rotation_euler = (math.radians(sun_angle_deg), 0, math.radians(sun_rot_deg))
    bpy.ops.object.light_add(type='AREA', location=(-0.5, 0.5, 0.9))
    fill = bpy.context.object; fill.data.energy = 80; fill.data.size = 1.0
    world = bpy.data.worlds['World']; world.use_nodes = True
    world.node_tree.nodes['Background'].inputs[0].default_value = (0.35,0.45,0.55,1)
    world.node_tree.nodes['Background'].inputs[1].default_value = 0.25

# ---- Bounding box de los modelos (para ajustar cámara) ----
def get_subject_size(objs):
    """Retorna el radio máximo desde el origen del conjunto de objetos."""
    max_r = 0.0
    for o in objs:
        if o.type != 'MESH': continue
        for v in o.data.vertices:
            wv = o.matrix_world @ v.co
            r = (wv.x**2 + wv.y**2 + wv.z**2) ** 0.5
            if r > max_r: max_r = r
    return max_r if max_r > 0 else 1.0

# ---- Cámara (distancia relativa al tamaño real del modelo) ----
def set_camera(elev_deg, azim_deg, dist_factor, subject_objs):
    from mathutils import Vector
    radius = get_subject_size(subject_objs)
    dist = radius * dist_factor          # distancia proporcional al modelo
    e = math.radians(elev_deg); a = math.radians(azim_deg)
    x = dist * math.cos(e) * math.sin(a)
    y = dist * math.cos(e) * math.cos(a)
    z = dist * math.sin(e)
    if "Camera" not in bpy.data.objects:
        bpy.ops.object.camera_add()
    cam = bpy.data.objects["Camera"]
    cam.location = (x, y, z)
    d = Vector((0, 0, 0)) - Vector((x, y, z))
    cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    cam.data.lens = 28
    bpy.context.scene.camera = cam
    print(f"  [cam] radio={radius:.3f}  dist={dist:.3f}")

# ---- Materiales máscara ----
def emit(color):
    m = bpy.data.materials.new("M"); m.use_nodes = True
    m.node_tree.nodes.clear()
    e = m.node_tree.nodes.new('ShaderNodeEmission')
    o = m.node_tree.nodes.new('ShaderNodeOutputMaterial')
    e.inputs["Color"].default_value = (*color, 1)
    e.inputs["Strength"].default_value = 1
    m.node_tree.links.new(e.outputs[0], o.inputs[0])
    return m

# ---- Render par (color + máscara) ----
def render_pair(name, subject_parts):
    # Color
    scene.render.image_settings.file_format = 'JPEG'
    scene.render.image_settings.quality     = 95
    scene.cycles.samples = 48
    scene.render.filepath = str(OUT_DIR / f"{name}.jpg")
    bpy.ops.render.render(write_still=True)
    print(f"  ✅ {name}.jpg")

    # Guardar materiales
    orig = {o.name: list(o.data.materials) for o in bpy.data.objects if o.type == 'MESH'}

    white = emit((1,1,1)); black = emit((0,0,0))
    subject_set = set(subject_parts)
    for obj in bpy.data.objects:
        if obj.type != 'MESH': continue
        obj.data.materials.clear()
        obj.data.materials.append(white if obj in subject_set else black)

    scene.cycles.samples = 1
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = str(OUT_DIR / f"{name}_mask.png")
    bpy.ops.render.render(write_still=True)
    print(f"  ✅ {name}_mask.png")

    # Restaurar
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.name in orig:
            obj.data.materials.clear()
            for m in orig[obj.name]: obj.data.materials.append(m)

# =====================================================================
# MAIN
# =====================================================================
ANGLES = [
    # nombre,  elevación, azimut, dist_factor, luz_elev
    ("cenital",  88,  0,   2.2, 70),
    ("diagonal", 50, 30,   2.8, 55),
    ("frontal",  25,  0,   3.2, 45),
    ("lateral",  35, 90,   2.6, 50),
]

make_floor()
foot_parts = load_glb(FOOT_GLB, "Pie")
shoe_parts = load_glb(SHOE_GLB, "Zapato")
all_parts  = foot_parts + shoe_parts

for name, elev, azim, dist_f, luz in ANGLES:
    print(f"\n→ {name} (elev={elev}°, azim={azim}°)...")
    set_camera(elev, azim, dist_f, all_parts)
    setup_lights(luz, azim + 30)
    render_pair(name, all_parts)

print(f"\n✅ 4 renders en: {OUT_DIR}")
