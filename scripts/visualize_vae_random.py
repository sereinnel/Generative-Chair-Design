# scripts/visualize_vae_random.py
import os
import torch
import argparse
import numpy as np
import open3d as o3d
from model_improved import PointNetVAE

def sample_random_chairs(model, device, num_samples=5, latent_scale=1.0, save_dir="results/visualization", open3d_view=False):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        for i in range(num_samples):
            z = torch.randn(1, model.latent_dim).to(device) * latent_scale
            generated = model.decode(z).cpu().numpy().squeeze()  # (num_points, 3)
            
            ply_path = os.path.join(save_dir, f"random_{i:02d}.ply")
            points_o3d = o3d.geometry.PointCloud()
            points_o3d.points = o3d.utility.Vector3dVector(generated)
            o3d.io.write_point_cloud(ply_path, points_o3d)
            print(f"Saved: {ply_path}")
            
            if open3d_view:
                o3d.visualization.draw_geometries([points_o3d])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth model")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--latent_scale", type=float, default=1.0, help="Scale for random latent vector for diversity")
    parser.add_argument("--open3d", action="store_true", help="Open Open3D viewer")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Инициализируем модель
    model = PointNetVAE(latent_dim=args.latent_dim, num_points=16384)  # num_points должен соответствовать обученной модели
    model.to(device)
    
    # Загружаем веса
    checkpoint = torch.load(args.model_path, map_location=device)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    print(f"Model loaded: {args.model_path}")
    
    # Генерация случайных стульев
    sample_random_chairs(model, device, num_samples=args.num_samples, latent_scale=args.latent_scale, open3d_view=args.open3d)

if __name__ == "__main__":
    main()
