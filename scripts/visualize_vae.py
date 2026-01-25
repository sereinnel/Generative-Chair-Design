# scripts/visualize_vae.py
import os
import argparse
import torch
import numpy as np
import open3d as o3d

from model_improved import PointNetVAE

RESULTS_DIR = "results/visualization"

def save_ply(points, path):
    """Сохраняет облако точек в формате .ply"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(path, pcd)

def visualize_open3d(points):
    """Показывает облако точек через Open3D"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

def generate_random_samples(model, device, num_points=4096, n_samples=5, open3d_view=False):
    model.eval()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with torch.no_grad():
        for i in range(n_samples):
            z = torch.randn(1, model.latent_dim).to(device)
            x_recon = model.decode(z)  # [1, N, 3]
            points = x_recon.squeeze(0).cpu().numpy()

            ply_path = os.path.join(RESULTS_DIR, f"random_{i:02d}.ply")
            save_ply(points, ply_path)
            print(f"Saved: {ply_path}")

            if open3d_view:
                visualize_open3d(points)

def reconstruct_samples(model, device, npy_dir, open3d_view=False):
    model.eval()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]
    with torch.no_grad():
        for f in files:
            path = os.path.join(npy_dir, f)
            x = np.load(path)  # [N, 3]
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)  # [1, N, 3]
            x_tensor = x_tensor.transpose(1, 2).contiguous()  # [1, 3, N]

            z_mu, _ = model.encode(x_tensor)
            x_recon = model.decode(z_mu)  # [1, N, 3]
            points = x_recon.squeeze(0).cpu().numpy()

            ply_path = os.path.join(RESULTS_DIR, f"recon_{os.path.splitext(f)[0]}.ply")
            save_ply(points, ply_path)
            print(f"Saved reconstruction: {ply_path}")

            if open3d_view:
                visualize_open3d(points)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Создаём модель
    model = PointNetVAE(latent_dim=args.latent_dim, num_points=args.num_points).to(device)

    # Загружаем чекпойнт
    model_path = args.model_path
    checkpoint = torch.load(model_path, map_location=device)
    
    # Поддержка старых и новых форматов чекпойнта
    if "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    print("Model loaded:", model_path)

    # Генерация и/или реконструкция
    if args.mode in ["random", "both"]:
        print("Generating random samples...")
        generate_random_samples(model, device, num_points=args.num_points, open3d_view=args.open3d)
    
    if args.mode in ["reconstruct", "both"]:
        npy_dir = os.path.join("data", "normalized_npy", "test")
        print("Reconstructing samples from:", npy_dir)
        reconstruct_samples(model, device, npy_dir, open3d_view=args.open3d)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/model_improved_best.pth")
    parser.add_argument("--mode", type=str, choices=["random", "reconstruct", "both"], default="both")
    parser.add_argument("--num_points", type=int, default=4096)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--open3d", action="store_true")
    args = parser.parse_args()

    main(args)
