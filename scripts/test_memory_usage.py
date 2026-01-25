# scripts/test_memory_usage.py
import torch
import numpy as np

def estimate_memory_usage():
    """Оценивает использование памяти для разных конфигураций"""
    
    configs = [
        {'points': 4096, 'batch': 32, 'latent': 128},
        {'points': 8192, 'batch': 16, 'latent': 256},
        {'points': 16384, 'batch': 8, 'latent': 256},
        {'points': 16384, 'batch': 16, 'latent': 256},  # Для RTX 4070
        {'points': 16384, 'batch': 12, 'latent': 512},
    ]
    
    print("ОЦЕНКА ИСПОЛЬЗОВАНИЯ ПАМЯТИ GPU")
    print("=" * 60)
    
    for cfg in configs:
        # Примерный расчет памяти
        points = cfg['points']
        batch = cfg['batch']
        latent = cfg['latent']
        
        # Память для данных
        data_memory = batch * points * 3 * 4  # float32 = 4 байта
        data_memory_gb = data_memory / 1e9
        
        # Память для активаций и градиентов (приблизительно)
        model_memory = (batch * points * latent * 4 * 10) / 1e9  # Упрощенная формула
        
        total_gb = data_memory_gb + model_memory
        
        print(f"\nКонфигурация: {points} точек, batch={batch}, latent={latent}")
        print(f"  Данные: {data_memory_gb:.2f} GB")
        print(f"  Модель: {model_memory:.2f} GB")
        print(f"  Всего: {total_gb:.2f} GB")
        
        if total_gb < 10:  # RTX 4070 имеет 12GB
            print(f"  ✅ Вписывается в 12GB VRAM")
        elif total_gb < 12:
            print(f"  ⚠️ Близко к лимиту 12GB")
        else:
            print(f"  ❌ Превышает 12GB VRAM")

if __name__ == "__main__":
    estimate_memory_usage()