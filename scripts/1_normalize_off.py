#1_normalize_off.py
"""
Нормализация мешей стульев с сохранением структуры
Ключевые изменения:
1. Сохраняем ориентацию (не центрируем по X/Y)
2. Масштабируем пропорционально высоте
3. Гарантируем, что пол всегда в Z=0
"""

import os
import numpy as np
import open3d as o3d
from tqdm import tqdm

def normalize_chair_mesh(off_path, output_path):
    """
    Нормализует меш стула для генеративного обучения
    
    Аргументы:
        off_path: путь к исходному .off файлу
        output_path: путь для сохранения нормализованного .off
    
    Возвращает:
        True при успехе, False при ошибке
    """
    try:
        # Загрузка меша
        mesh = o3d.io.read_triangle_mesh(off_path)
        if len(mesh.vertices) == 0:
            print(f"  ⚠️ Пустой меш: {os.path.basename(off_path)}")
            return False
        
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        
        # === КРИТИЧЕСКИ ВАЖНО: ПРАВИЛЬНАЯ НОРМАЛИЗАЦИЯ ===
        
        # 1. Выравнивание по полу: самая низкая точка = 0
        z_min = np.min(vertices[:, 2])
        vertices[:, 2] -= z_min
        
        # 2. Находим ВЫСОТУ стула (максимум по Z после выравнивания)
        height = np.max(vertices[:, 2])
        if height < 0.001:  # защита от деления на 0
            print(f"  ⚠️ Нулевая высота: {os.path.basename(off_path)}")
            height = 1.0
        
        # 3. Масштабируем ВСЕ оси на одинаковый коэффициент
        #    Это сохраняет пропорции стула!
        scale = 1.0 / height
        vertices *= scale
        
        # 4. НЕ центрируем по X и Y! Это сохраняет естественную ориентацию
        #    Но можем слегка сдвинуть, чтобы центр масс был в XY-плоскости
        #    (не обязательно, но помогает обучению)
        center_x = np.mean(vertices[:, 0])
        center_y = np.mean(vertices[:, 1])
        vertices[:, 0] -= center_x
        vertices[:, 1] -= center_y
        
        # 5. Проверяем результат
        #    - Пол должен быть в Z=0
        #    - Высота должна быть примерно 1.0
        #    - Ширина/глубина должны быть < 1.0 (стулья обычно шире, чем выше)
        
        # Обновляем меш
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        
        # Сохраняем
        o3d.io.write_triangle_mesh(output_path, mesh)
        
        # Логирование для отладки
        bbox = mesh.get_axis_aligned_bounding_box()
        bbox_size = bbox.get_extent()
        print(f"  ✅ {os.path.basename(off_path):30} | "
              f"Размер: {bbox_size[0]:.2f}x{bbox_size[1]:.2f}x{bbox_size[2]:.2f} | "
              f"Высота: {bbox_size[2]:.2f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Ошибка при обработке {off_path}: {e}")
        return False

def process_dataset(input_dir, output_dir):
    """
    Обрабатывает все .off файлы в директории
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Получаем список файлов
    off_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.off')]
    
    if not off_files:
        print(f"⚠️ Нет .off файлов в {input_dir}")
        return 0
    
    print(f"📁 Обработка {len(off_files)} файлов из {input_dir}")
    
    success_count = 0
    for filename in tqdm(off_files, desc="Нормализация"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        if normalize_chair_mesh(input_path, output_path):
            success_count += 1
    
    return success_count

def main():
    print("=" * 60)
    print("НОРМАЛИЗАЦИЯ МЕШЕЙ СТУЛЬЕВ")
    print("=" * 60)
    
    # Пути для нового эксперимента
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)  # поднимаемся на уровень выше scripts/
    
    # Директории
    raw_train_dir = os.path.join(project_root, "data", "raw", "train")
    raw_test_dir = os.path.join(project_root, "data", "raw", "test")
    
    norm_train_dir = os.path.join(project_root, "data", "normalized_off", "train")
    norm_test_dir = os.path.join(project_root, "data", "normalized_off", "test")
    
    # Проверяем существование исходных данных
    if not os.path.exists(raw_train_dir):
        print(f"❌ Не найдена папка с исходными данными: {raw_train_dir}")
        print("Убедитесь, что вы скопировали .off файлы в data/raw/train/")
        return
    
    # Обработка тренировочных данных
    print("\n🔧 НОРМАЛИЗАЦИЯ ТРЕНИРОВОЧНЫХ ДАННЫХ")
    train_success = process_dataset(raw_train_dir, norm_train_dir)
    
    # Обработка тестовых данных
    print("\n🔧 НОРМАЛИЗАЦИЯ ТЕСТОВЫХ ДАННЫХ")
    test_success = process_dataset(raw_test_dir, norm_test_dir)
    
    # Итоги
    print("\n" + "=" * 60)
    print("ИТОГИ НОРМАЛИЗАЦИИ:")
    print(f"  Тренировочные: {train_success} из {len(os.listdir(raw_train_dir))}")
    print(f"  Тестовые: {test_success} из {len(os.listdir(raw_test_dir))}")
    
    # Пример проверки нормализации
    if train_success > 0:
        print("\n📏 ПРИМЕР НОРМАЛИЗОВАННОГО СТУЛА:")
        sample_file = os.listdir(norm_train_dir)[0]
        sample_path = os.path.join(norm_train_dir, sample_file)
        
        mesh = o3d.io.read_triangle_mesh(sample_path)
        vertices = np.asarray(mesh.vertices)
        
        print(f"  Файл: {sample_file}")
        print(f"  Количество вершин: {len(vertices)}")
        print(f"  Z диапазон: [{vertices[:, 2].min():.3f}, {vertices[:, 2].max():.3f}]")
        print(f"  Высота: {vertices[:, 2].max() - vertices[:, 2].min():.3f}")
        print(f"  Центр: [{vertices[:, 0].mean():.3f}, "
              f"{vertices[:, 1].mean():.3f}, {vertices[:, 2].mean():.3f}]")

if __name__ == "__main__":
    main()