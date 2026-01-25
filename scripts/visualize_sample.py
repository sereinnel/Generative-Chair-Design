# scripts/visualize_sample.py
import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def visualize_chair_samples():
    """Визуализирует несколько случайных стульев для проверки"""
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    
    # Пути к данным
    raw_dir = os.path.join(project_root, "data", "raw", "train")
    norm_off_dir = os.path.join(project_root, "data", "normalized_off", "train")
    npy_dir = os.path.join(project_root, "data", "normalized_npy", "train")
    
    import random
    files = [f for f in os.listdir(raw_dir) if f.endswith('.off')]
    samples = random.sample(files, min(3, len(files)))
    
    for i, filename in enumerate(samples):
        # 1. Исходный меш
        raw_path = os.path.join(raw_dir, filename)
        raw_mesh = o3d.io.read_triangle_mesh(raw_path)
        
        # 2. Нормализованный меш
        norm_path = os.path.join(norm_off_dir, filename)
        norm_mesh = o3d.io.read_triangle_mesh(norm_path)
        
        # 3. Облако точек (если существует)
        npy_file = filename.replace('.off', '.npy')
        npy_path = os.path.join(npy_dir, npy_file)
        
        fig = plt.figure(figsize=(15, 5))
        fig.suptitle(f"Стул: {filename}", fontsize=16)
        
        # Субплотов
        views = ['Исходный', 'Нормализованный', 'Облако точек']
        
        for j, (title, geometry) in enumerate([
            (views[0], raw_mesh),
            (views[1], norm_mesh),
            (views[2], None)
        ]):
            ax = fig.add_subplot(1, 3, j+1, projection='3d')
            
            if j == 2 and os.path.exists(npy_path):
                points = np.load(npy_path)
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                          c=points[:, 2], cmap='viridis', s=1, alpha=0.6)
                ax.set_title(f"{title}\n{len(points)} точек")
            else:
                # Визуализация меша
                vertices = np.asarray(geometry.vertices)
                ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                               triangles=np.asarray(geometry.triangles),
                               alpha=0.8, cmap='viridis')
                ax.set_title(f"{title}\n{len(vertices)} вершин")
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_box_aspect([1, 1, 1])  # одинаковый масштаб по осям
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    visualize_chair_samples()