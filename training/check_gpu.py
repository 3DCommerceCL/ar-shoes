import bpy

prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.refresh_devices()

print("\n=== DISPOSITIVOS CYCLES ===")
for device_type in ('CUDA', 'OPTIX', 'HIP', 'METAL', 'ONEAPI'):
    try:
        prefs.compute_device_type = device_type
        devices = prefs.devices
        if devices:
            for d in devices:
                print(f"  [{device_type}] {d.name}  use={d.use}")
    except:
        pass

prefs.compute_device_type = 'NONE'
print("===========================\n")
