#4_generate_chairs.py
import torch
import numpy as np
import open3d as o3d
import os
from pointnet.model import PointNetCls

# === Определение модели ===
class PointNetVAE(torch.nn.Module):
    def __init__(self, num_points=2048, latent_dim=128):
        super().__init__()
        self.encoder = PointNetCls(k=latent_dim * 2)
        self.encoder.fc2 = torch.nn.Linear(512, latent_dim * 2)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, num_points * 3)
        )

    def decode(self, z):
        x = self.decoder(z)
        return x.view(-1, 2048, 3)

# === Основной скрипт ===
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Загрузка модели
    model = PointNetVAE(latent_dim=128)
    model.load_state_dict(torch.load("model_normalized.pth", map_location=device, weights_only=False))
    model.eval()
    model.to(device)

    # Создание папки
    output_dir = "generated_chairs_10"
    os.makedirs(output_dir, exist_ok=True)

    # Генерация 25 стульев
    for i in range(10):
        z = torch.randn(1, 128).to(device)
        with torch.no_grad():
            points = model.decode(z).cpu().numpy()[0]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        filename = os.path.join(output_dir, f"chair_{i:02d}.ply")
        o3d.io.write_point_cloud(filename, pcd)
        print(f"✅ Сохранено: {filename}")

    print(f"\n🎉 Готово! 10 стульев сохранены в папке: {output_dir}")