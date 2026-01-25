#2_convert_to_npy.py
"""
Конвертация нормализованных мешей в облака точек (.npy)
Используем равномерное сэмплирование с фиксированным количеством точек
"""

import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
import multiprocessing
from functools import partial

def convert_single_file(off_path, npy_path, num_points=4096):
    """
    Конвертирует один .off файл в .npy с фиксированным количеством точек
    
    Аргументы:
        off_path: путь к .off файлу
        npy_path: путь для сохранения .npy
        num_points: количество точек в облаке
    
    Возвращает:
        True при успехе, False при ошибке
    """
    try:
        # Загрузка меша
        mesh = o3d.io.read_triangle_mesh(off_path)
        if len(mesh.vertices) == 0:
            print(f"  ⚠️ Пустой меш: {os.path.basename(off_path)}")
            return False
        
        # Равномерное сэмплирование точек с поверхности
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        
        # Получаем точки как numpy array
        points = np.asarray(pcd.points, dtype=np.float32)
        
        # Проверяем корректность
        if points.shape != (num_points, 3):
            print(f"  ⚠️ Неправильная форма: {points.shape} для {os.path.basename(off_path)}")
            return False
        
        # Сохраняем в .npy
        np.save(npy_path, points)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Ошибка при конвертации {off_path}: {e}")
        return False

def process_batch(file_list, input_dir, output_dir, num_points=4096):
    """
    Обрабатывает батч файлов (для многопроцессорности)
    """
    success_count = 0
    for filename in file_list:
        off_path = os.path.join(input_dir, filename)
        npy_filename = filename.replace('.off', '.npy')
        npy_path = os.path.join(output_dir, npy_filename)
        
        if convert_single_file(off_path, npy_path, num_points):
            success_count += 1
    
    return success_count

def convert_dataset(input_dir, output_dir, num_points=4096, num_workers=4):
    """
    Конвертирует все .off файлы в директории в .npy
    
    Аргументы:
        input_dir: папка с .off файлами
        output_dir: папка для сохранения .npy
        num_points: количество точек в каждом облаке
        num_workers: количество процессов для параллельной обработки
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Получаем список файлов
    off_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.off')]
    
    if not off_files:
        print(f"⚠️ Нет .off файлов в {input_dir}")
        return 0
    
    print(f"📁 Конвертация {len(off_files)} файлов из {input_dir}")
    print(f"  Количество точек на облако: {num_points}")
    print(f"  Количество процессов: {num_workers}")
    
    # Если мало файлов или один воркер, обрабатываем последовательно
    if len(off_files) < 50 or num_workers == 1:
        success_count = 0
        for filename in tqdm(off_files, desc="Конвертация"):
            off_path = os.path.join(input_dir, filename)
            npy_filename = filename.replace('.off', '.npy')
            npy_path = os.path.join(output_dir, npy_filename)
            
            if convert_single_file(off_path, npy_path, num_points):
                success_count += 1
        
        return success_count
    
    # Многопроцессорная обработка
    else:
        # Разделяем файлы на батчи
        batch_size = len(off_files) // num_workers
        batches = [off_files[i:i + batch_size] for i in range(0, len(off_files), batch_size)]
        
        # Создаем partial функцию с фиксированными аргументами
        process_func = partial(
            process_batch,
            input_dir=input_dir,
            output_dir=output_dir,
            num_points=num_points
        )
        
        # Запускаем процессы
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_func, batches),
                total=len(batches),
                desc="Конвертация (многопроцесс)"
            ))
        
        return sum(results)

def validate_point_clouds(npy_dir, num_points=4096, num_samples=5):
    """
    Проверяет корректность сгенерированных облаков точек
    
    Аргументы:
        npy_dir: папка с .npy файлами
        num_points: ожидаемое количество точек
        num_samples: сколько файлов проверить
    """
    print("\n🔍 ВАЛИДАЦИЯ ОБЛАКОВ ТОЧЕК:")
    
    npy_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]
    
    if not npy_files:
        print("  ⚠️ Нет .npy файлов для проверки")
        return
    
    # Проверяем несколько случайных файлов
    import random
    sample_files = random.sample(npy_files, min(num_samples, len(npy_files)))
    
    for filename in sample_files:
        npy_path = os.path.join(npy_dir, filename)
        points = np.load(npy_path)
        
        print(f"\n  Файл: {filename}")
        print(f"    Форма: {points.shape} (ожидается ({num_points}, 3))")
        print(f"    Диапазон X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
        print(f"    Диапазон Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
        print(f"    Диапазон Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
        print(f"    Среднее: [{points[:, 0].mean():.3f}, {points[:, 1].mean():.3f}, {points[:, 2].mean():.3f}]")
        
        # Проверяем наличие NaN или Inf
        if np.any(np.isnan(points)):
            print("    ❌ Обнаружены NaN значения!")
        if np.any(np.isinf(points)):
            print("    ❌ Обнаружены Inf значения!")
        
        # Проверяем, что высота нормализована к 1.0
        height = points[:, 2].max() - points[:, 2].min()
        if abs(height - 1.0) > 0.05:  # допуск 5%
            print(f"    ⚠️ Высота отличается от 1.0: {height:.3f}")

def main():
    print("=" * 60)
    print("КОНВЕРТАЦИЯ МЕШЕЙ В ОБЛАКА ТОЧЕК")
    print("=" * 60)
    
    # Настройки
    NUM_POINTS = 16384  # Увеличиваем для лучшей детализации
    NUM_WORKERS = 6    # Количество процессов
    
    # Пути
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    
    # Входные директории (нормализованные .off)
    norm_train_dir = os.path.join(project_root, "data", "normalized_off", "train")
    norm_test_dir = os.path.join(project_root, "data", "normalized_off", "test")
    
    # Выходные директории (.npy)
    npy_train_dir = os.path.join(project_root, "data", "normalized_npy", "train")
    npy_test_dir = os.path.join(project_root, "data", "normalized_npy", "test")
    
    # Проверяем существование входных данных
    if not os.path.exists(norm_train_dir):
        print(f"❌ Не найдена папка с нормализованными данными: {norm_train_dir}")
        print("Сначала запустите скрипт 1_normalize_off.py")
        return
    
    # Конвертируем тренировочные данные
    print("\n🔧 КОНВЕРТАЦИЯ ТРЕНИРОВОЧНЫХ ДАННЫХ")
    train_success = convert_dataset(
        norm_train_dir, npy_train_dir, 
        num_points=NUM_POINTS, 
        num_workers=NUM_WORKERS
    )
    
    # Конвертируем тестовые данные
    print("\n🔧 КОНВЕРТАЦИЯ ТЕСТОВЫХ ДАННЫХ")
    test_success = convert_dataset(
        norm_test_dir, npy_test_dir, 
        num_points=NUM_POINTS, 
        num_workers=NUM_WORKERS
    )
    
    # Итоги
    print("\n" + "=" * 60)
    print("ИТОГИ КОНВЕРТАЦИИ:")
    print(f"  Тренировочные: {train_success} файлов")
    print(f"  Тестовые: {test_success} файлов")
    
    # Проверяем результат
    if train_success > 0:
        validate_point_clouds(npy_train_dir, NUM_POINTS)
    
    # Создаем файл с информацией о датасете
    info_path = os.path.join(project_root, "data", "dataset_info.txt")
    with open(info_path, 'w') as f:
        f.write(f"CHAIR DATASET INFO\n")
        f.write(f"==================\n")
        f.write(f"Training samples: {train_success}\n")
        f.write(f"Test samples: {test_success}\n")
        f.write(f"Points per cloud: {NUM_POINTS}\n")
        f.write(f"Normalized: yes (height=1.0, floor at z=0)\n")
        f.write(f"Data format: .npy (float32, shape: ({NUM_POINTS}, 3))\n")
    
    print(f"\n📊 Информация о датасете сохранена в: {info_path}")

if __name__ == "__main__":
    # На Windows при использовании многопроцессорности нужно это условие
    multiprocessing.freeze_support()
    main()