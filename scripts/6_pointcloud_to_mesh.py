# 6_pointcloud_to_mesh.py
"""
Конвертация облаков точек (.ply / .npy) -> watertight .obj с помощью Open3D (Poisson reconstruction).
Скрипт:
 - читает все .ply/.npy из input_dir
 - очищает шум (statistical outlier removal)
 - оценивает и ориентирует нормали
 - Poisson reconstruction -> mesh + densities
 - отбрасывает низкоплотные вершины по квантилю
 - опционально: fallback Ball Pivoting, упрощение mesh
 - сохраняет .obj и (по желанию) промежуточные .ply/.ply.cleaned
"""

import os
import argparse
import numpy as np
import open3d as o3d
from tqdm import tqdm
import traceback

SUPPORTED_IN = (".ply", ".pcd", ".xyz", ".npy")

def load_pointcloud(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        pts = np.load(path).astype(np.float32)
        pcd = o3d.geometry.PointCloud()
                # --- PREPROCESS POINT CLOUD ---

        # 1. Лёгкое voxel downsample (стабилизирует плотность)
        pcd = pcd.voxel_down_sample(voxel_size=0.005)

        # 2. Удаление статистических выбросов
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=30,
            std_ratio=1.5
        )

        # 3. Пересчёт нормалей (ВАЖНО для Poisson)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.02,
                max_nn=30
            )
        )
        pcd.normalize_normals()

        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd
    else:
        pcd = o3d.io.read_point_cloud(path)
        return pcd

def clean_and_estimate_normals(pcd, nb_neighbors=50, std_ratio=1.0, knn_normals=30):
    # remove statistical outliers
    try:
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    except Exception:
        # fallback: return original if removal fails
        pass

    # estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn_normals))
    # orient normals consistently (tangent plane)
    try:
        pcd.orient_normals_consistent_tangent_plane(k=knn_normals)
    except Exception:
        # if fails, try simpler orientation
        try:
            pcd.orient_normals_towards_camera_location(np.array([0., 0., 0.]))
        except Exception:
            pass
    return pcd

def poisson_reconstruct(pcd, depth=10):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    densities = np.asarray(densities)
    return mesh, densities

def filter_mesh_by_density(mesh, densities, keep_quantile=0.1):
    # densities mapped to vertices
    thr = np.quantile(densities, keep_quantile)
    mask = densities < thr
    try:
        mesh.remove_vertices_by_mask(mask)
    except Exception:
        # older Open3D versions require boolean mask of length == n_vertices
        mesh.remove_vertices_by_mask(mask)
    return mesh

def simplify_mesh(mesh, target_triangles):
    try:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=int(target_triangles))
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
    except Exception:
        pass
    return mesh

def ball_pivoting_reconstruct(pcd, radii=[0.01, 0.02, 0.04]):
    # fallback mesh: Ball Pivoting (may produce non-watertight)
    try:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
        mesh.compute_vertex_normals()
        return mesh
    except Exception:
        return None

def process_file(path, out_dir, params):
    basename = os.path.splitext(os.path.basename(path))[0]
    try:
        pcd = load_pointcloud(path)
        if len(pcd.points) == 0:
            return {"file": path, "status": "empty", "note": "no points"}

        # optional: downsample very dense clouds to speed up (if > max_points)
        if params["max_points"] and len(pcd.points) > params["max_points"]:
            pcd = pcd.random_down_sample(float(params["max_points"]) / len(pcd.points))

        # clean + normals
        pcd = clean_and_estimate_normals(pcd,
                                         nb_neighbors=params["nb_neighbors"],
                                         std_ratio=params["std_ratio"],
                                         knn_normals=params["knn_normals"])

        # save cleaned pointcloud optionally
        if params["keep_intermediate"]:
            cleaned_ply = os.path.join(out_dir, f"{basename}_cleaned.ply")
            o3d.io.write_point_cloud(cleaned_ply, pcd)

        # Poisson reconstruction
        mesh, densities = poisson_reconstruct(pcd, depth=params["depth"])

        # filter by density quantile
        mesh = filter_mesh_by_density(mesh, densities, keep_quantile=params["density_quantile"])

        # optional simplify
        if params["target_triangles"]:
            mesh = simplify_mesh(mesh, params["target_triangles"])

        # ensure normals and watertight-ish
        mesh.compute_vertex_normals()

        out_obj = os.path.join(out_dir, f"{basename}.obj")
        o3d.io.write_triangle_mesh(out_obj, mesh, write_triangle_uvs=False)
        return {"file": path, "status": "ok", "out_obj": out_obj}

    except Exception as e:
        # fallback: try Ball Pivoting
        try:
            pcd = load_pointcloud(path)
            pcd = clean_and_estimate_normals(pcd,
                                             nb_neighbors=params["nb_neighbors"],
                                             std_ratio=params["std_ratio"],
                                             knn_normals=params["knn_normals"])
            mesh_bp = ball_pivoting_reconstruct(pcd, radii=params["bp_radii"])
            if mesh_bp is not None:
                if params["target_triangles"]:
                    mesh_bp = simplify_mesh(mesh_bp, params["target_triangles"])
                out_obj = os.path.join(out_dir, f"{basename}_bp.obj")
                o3d.io.write_triangle_mesh(out_obj, mesh_bp)
                return {"file": path, "status": "ok_bp", "out_obj": out_obj, "note": str(e)}
        except Exception:
            pass

        return {"file": path, "status": "error", "error": str(e), "trace": traceback.format_exc()}

def main():
    parser = argparse.ArgumentParser(description="Convert pointclouds to watertight OBJ via Poisson (Open3D)")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder with .ply or .npy generated pointclouds")
    parser.add_argument("--output_dir", type=str, default="results/meshes", help="Where to save .obj")
    parser.add_argument("--depth", type=int, default=10, help="Poisson depth (increase -> more detail, more memory)")
    parser.add_argument("--density_quantile", type=float, default=0.1, help="Remove vertices with density < quantile (0..1)")
    parser.add_argument("--target_triangles", type=int, default=15000, help="Simplify mesh to this number of triangles (0 = skip)")
    parser.add_argument("--nb_neighbors", type=int, default=50, help="neighbors for statistical outlier removal")
    parser.add_argument("--std_ratio", type=float, default=1.0, help="std ratio for outlier removal")
    parser.add_argument("--knn_normals", type=int, default=30, help="knn for normal estimation")
    parser.add_argument("--max_points", type=int, default=200000, help="Downsample input if points > this")
    parser.add_argument("--keep_intermediate", action="store_true", help="Save cleaned pointclouds as _cleaned.ply")
    parser.add_argument("--open", action="store_true", help="Open each finished mesh in Open3D viewer (slow)")
    parser.add_argument("--bp_radii", type=float, nargs="+", default=[0.01, 0.02, 0.04], help="Radii for Ball Pivoting fallback")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(SUPPORTED_IN)]
    files.sort()
    if len(files) == 0:
        print("No supported files found in", args.input_dir)
        return

    params = {
        "depth": args.depth,
        "density_quantile": args.density_quantile,
        "target_triangles": args.target_triangles if args.target_triangles > 0 else None,
        "nb_neighbors": args.nb_neighbors,
        "std_ratio": args.std_ratio,
        "knn_normals": args.knn_normals,
        "max_points": args.max_points,
        "keep_intermediate": args.keep_intermediate,
        "bp_radii": args.bp_radii
    }

    results = []
    for fn in tqdm(files, desc="Convert"):
        path = os.path.join(args.input_dir, fn)
        res = process_file(path, args.output_dir, params)
        results.append(res)
        if res.get("status") in ("ok", "ok_bp"):
            print(f"[OK] {fn} -> {os.path.basename(res['out_obj'])}")
            if args.open:
                try:
                    mesh = o3d.io.read_triangle_mesh(res["out_obj"])
                    mesh.compute_vertex_normals()
                    o3d.visualization.draw_geometries([mesh])
                except Exception:
                    pass
        else:
            print(f"[ERR] {fn} : {res.get('error') or res.get('note')}")

    # Summary
    n_ok = sum(1 for r in results if r.get("status") in ("ok", "ok_bp"))
    n_err = sum(1 for r in results if r.get("status") == "error")
    print("Done. Success:", n_ok, "Errors:", n_err)
    print("Output dir:", args.output_dir)

if __name__ == "__main__":
    main()
