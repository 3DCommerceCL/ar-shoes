"""
Render de prueba — crea un pie procedural simple y genera:
  - test_color.jpg   (imagen con iluminación realista)
  - test_mask.png    (máscara binaria perfecta)
"""

import bpy
import math
import random
from mathutils import Vector

OUT_COLOR = "C:/Users/Iga/Documents/ar-shoes/training/test_color.jpg"
OUT_MASK  = "C:/Users/Iga/Documents/ar-shoes/training/test_mask.png"
SIZE = 512

# ---- Limpiar escena ----
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# ---- Configurar render (CYCLES, 512×512) ----
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 64
scene.render.resolution_x = SIZE
scene.render.resolution_y = SIZE
scene.render.resolution_percentage = 100

# ---- Cámara cenital (desde arriba, 45 cm) ----
bpy.ops.object.camera_add(location=(0, 0, 0.45))
cam = bpy.context.object
cam.rotation_euler = (0, 0, 0)   # apunta directo hacia abajo
cam.data.type  = 'PERSP'
cam.data.lens  = 28
scene.camera   = cam

# ---- Piso (madera clara) ----
bpy.ops.mesh.primitive_plane_add(size=3.0, location=(0, 0, 0))
floor = bpy.context.object
floor.name = "Floor"
mat_floor = bpy.data.materials.new("FloorMat")
mat_floor.use_nodes = True
bsdf = mat_floor.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.65, 0.48, 0.30, 1)
bsdf.inputs["Roughness"].default_value  = 0.7
floor.data.materials.append(mat_floor)

# ---- Pie procedural (elipse aplastada = planta del pie) ----
# Cuerpo principal del pie
bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(0, 0, 0.04))
foot_body = bpy.context.object
foot_body.name = "Foot"
foot_body.scale = (0.055, 0.13, 0.04)   # ancho x largo x alto

# Talón (esfera más grande en la parte trasera)
bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(0, -0.09, 0.035))
heel = bpy.context.object
heel.name = "Heel"
heel.scale = (0.05, 0.05, 0.035)

# Dedos (5 esferas pequeñas en la parte delantera)
toe_positions = [(-0.044, 0.12, 0.025), (-0.022, 0.125, 0.025),
                 (0.0,    0.127, 0.025), (0.022, 0.122, 0.025),
                 (0.040,  0.115, 0.022)]
toe_scales    = [(0.018,0.022,0.018),(0.020,0.025,0.020),(0.020,0.026,0.020),
                 (0.018,0.022,0.018),(0.015,0.018,0.015)]
toes = []
for pos, sc in zip(toe_positions, toe_scales):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=pos)
    t = bpy.context.object
    t.scale = sc
    toes.append(t)

# Tobillo (cilindro pequeño)
bpy.ops.mesh.primitive_cylinder_add(radius=0.038, depth=0.06, location=(0, -0.10, 0.07))
ankle = bpy.context.object
ankle.name = "Ankle"

# ---- Material de piel ----
mat_skin = bpy.data.materials.new("SkinMat")
mat_skin.use_nodes = True
bsdf2 = mat_skin.node_tree.nodes["Principled BSDF"]
bsdf2.inputs["Base Color"].default_value = (0.80, 0.60, 0.45, 1)  # piel media
bsdf2.inputs["Roughness"].default_value  = 0.6
bsdf2.inputs["Subsurface Weight"].default_value = 0.3

foot_parts = [foot_body, heel, ankle] + toes
for obj in foot_parts:
    obj.data.materials.clear()
    obj.data.materials.append(mat_skin)

# ---- Iluminación ----
# Luz principal (sol)
bpy.ops.object.light_add(type='SUN', location=(1, -1, 2))
sun = bpy.context.object
sun.data.energy = 3.0
sun.rotation_euler = (math.radians(50), 0, math.radians(30))

# Luz de relleno suave
bpy.ops.object.light_add(type='AREA', location=(-0.5, 0.5, 0.8))
fill = bpy.context.object
fill.data.energy = 100
fill.data.size   = 0.8

# Ambiente
world = bpy.data.worlds['World']
world.use_nodes = True
world.node_tree.nodes['Background'].inputs[0].default_value = (0.4, 0.5, 0.6, 1)
world.node_tree.nodes['Background'].inputs[1].default_value = 0.3

# =====================================================================
# RENDER 1 — Imagen a color
# =====================================================================
scene.render.image_settings.file_format = 'JPEG'
scene.render.image_settings.quality = 95
scene.render.filepath = OUT_COLOR
bpy.ops.render.render(write_still=True)
print(f"[OK] Color: {OUT_COLOR}")

# =====================================================================
# RENDER 2 — Máscara binaria
# =====================================================================
# Guardar materiales originales
orig = {}
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        orig[obj.name] = list(obj.data.materials)

# Material blanco (pie) y negro (fondo)
def emit_mat(color):
    m = bpy.data.materials.new("M")
    m.use_nodes = True
    m.node_tree.nodes.clear()
    e = m.node_tree.nodes.new('ShaderNodeEmission')
    o = m.node_tree.nodes.new('ShaderNodeOutputMaterial')
    e.inputs["Color"].default_value    = (*color, 1)
    e.inputs["Strength"].default_value = 1.0
    m.node_tree.links.new(e.outputs[0], o.inputs[0])
    return m

white = emit_mat((1,1,1))
black = emit_mat((0,0,0))

for obj in bpy.data.objects:
    if obj.type != 'MESH': continue
    obj.data.materials.clear()
    obj.data.materials.append(white if obj in foot_parts else black)

scene.cycles.samples = 1
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = OUT_MASK
bpy.ops.render.render(write_still=True)
print(f"[OK] Máscara: {OUT_MASK}")

# Restaurar materiales
for obj in bpy.data.objects:
    if obj.type == 'MESH' and obj.name in orig:
        obj.data.materials.clear()
        for m in orig[obj.name]:
            obj.data.materials.append(m)

print("\n✅ Renders completados")
