"""Diagnóstico de GPU del proyecto — PyTorch (entrenamiento/SAM2) y Blender/Cycles (renders).

    py -3 check_gpu.py                 # revisa PyTorch (y avisa si torch quedó en versión CPU)
    blender --background --python check_gpu.py    # revisa además los dispositivos de Cycles

OJO: la versión anterior sólo miraba Cycles (importaba bpy), así que nunca avisó de que torch
estaba instalado en build CPU-only teniendo una GPU disponible. Si ves 'CUDA disponible: False'
con una NVIDIA presente, reinstalá torch con:
    py -3 -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu126
"""
import shutil
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def check_nvidia_smi():
    if not shutil.which("nvidia-smi"):
        print("  nvidia-smi no encontrado (¿sin GPU NVIDIA o sin drivers?)")
        return
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        for line in out.stdout.strip().splitlines():
            print(f"  GPU física: {line}")
    except Exception as e:
        print(f"  nvidia-smi falló: {e}")


def check_torch():
    print("\n=== PyTorch (entrenamiento y SAM2) ===")
    check_nvidia_smi()
    try:
        import torch
    except ImportError:
        print("  torch NO instalado")
        return
    print(f"  torch: {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"  CUDA disponible: {cuda_ok}")
    if cuda_ok:
        p = torch.cuda.get_device_properties(0)
        print(f"  Dispositivo: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {p.total_memory / 1e9:.1f} GB | compute capability {p.major}.{p.minor}")
        try:
            x = torch.randn(1024, 1024, device="cuda")
            (x @ x).sum().item()
            print("  Test de cómputo en GPU: OK")
        except Exception as e:
            print(f"  Test de cómputo en GPU: FALLÓ ({e})")
    elif "+cpu" in torch.__version__:
        print("  ⚠️  torch está en build CPU-ONLY. Si tenés GPU NVIDIA, reinstalá con:")
        print("      py -3 -m pip install --upgrade torch torchvision "
              "--index-url https://download.pytorch.org/whl/cu126")


def check_cycles():
    try:
        import bpy
    except ImportError:
        print("\n(Para revisar Cycles, correr este script desde Blender:")
        print(" blender --background --python check_gpu.py)")
        return
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.refresh_devices()
    print("\n=== Blender / Cycles (renders sintéticos) ===")
    for device_type in ('CUDA', 'OPTIX', 'HIP', 'METAL', 'ONEAPI'):
        try:
            prefs.compute_device_type = device_type
            for d in prefs.devices:
                print(f"  [{device_type}] {d.name}  use={d.use}")
        except Exception:
            pass
    prefs.compute_device_type = 'NONE'


if __name__ == "__main__":
    check_torch()
    check_cycles()
    print()
