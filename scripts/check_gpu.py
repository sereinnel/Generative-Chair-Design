# scripts/check_gpu.py
import torch
import subprocess
import sys

def check_gpu_capability():
    """Проверяет возможности GPU"""
    
    print("=" * 60)
    print("ПРОВЕРКА СИСТЕМЫ ДЛЯ ОБУЧЕНИЯ НА GPU")
    print("=" * 60)
    
    # 1. Проверка PyTorch и CUDA
    print("\n🔧 PyTorch информация:")
    print(f"  Версия PyTorch: {torch.__version__}")
    print(f"  CUDA доступна: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  Количество GPU: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\n  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Память: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            print(f"    CUDA Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
            
            # Оценим максимальный размер батча
            if "4070" in torch.cuda.get_device_name(i):
                print("    ✅ RTX 4070 обнаружена!")
                print("    Рекомендуемые настройки для VAE:")
                print("      - Batch size: 16-32")
                print("      - Points: 16384")
                print("      - Latent dim: 256-512")
    
    # 2. Проверка памяти
    print("\n💾 Системная память:")
    try:
        import psutil
        ram = psutil.virtual_memory()
        print(f"  Всего: {ram.total / 1e9:.2f} GB")
        print(f"  Доступно: {ram.available / 1e9:.2f} GB")
    except:
        print("  Установите psutil: pip install psutil")
    
    # 3. Проверка Python окружения
    print("\n🐍 Python окружение:")
    print(f"  Python версия: {sys.version}")
    
    # 4. Проверка open3d
    try:
        import open3d as o3d
        print(f"  Open3D версия: {o3d.__version__}")
    except:
        print("  ⚠️ Open3D не установлен")
    
    # 5. Рекомендации
    print("\n🎯 РЕКОМЕНДАЦИИ ДЛЯ RTX 4070:")
    print("  1. Увеличьте точки до 16384 в convert_to_npy.py")
    print("  2. Используйте batch_size=16-24 в обучении")
    print("  3. latent_dim=256-512")
    print("  4. Используйте mixed precision (AMP) для ускорения")
    
    return torch.cuda.is_available()

if __name__ == "__main__":
    check_gpu_capability()