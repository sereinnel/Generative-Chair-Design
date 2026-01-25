#4_generate_chairs.py
import os
import argparse
import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d
from tqdm import tqdm
import csv
import itertools
import math

def load_pointcloud_auto(path):
    """
    Загружает облако из .ply или .npy и возвращает numpy array shape (N,3), dtype=float32.
    """
    path = str(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ply" or ext == ".pcd" or ext == ".xyz":
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points, dtype=np.float32)
        return pts
    elif ext == ".npy":
        pts = np.load(path).astype(np.float32)
        return pts
    else:
        raise ValueError(f"Unsupported file extension: {ext} for file {path}")

def sample_points(points, num_points):
    """
    Если points.shape[0] >= num_points — случайная подвыборка без замены.
    Иначе — повтор с replacement.
    Возвращает (num_points, 3) np.float32.
    """
    n = points.shape[0]
    if n == num_points:
        return points.copy().astype(np.float32)
    if n > num_points:
        idx = np.random.choice(n, num_points, replace=False)
    else:
        idx = np.random.choice(n, num_points, replace=True)
    return points[idx].astype(np.float32)

def chamfer_distance_np(a, b):
    """
    Chamfer distance between two point clouds a and b (np arrays shape (N,3) and (M,3)).
    Returns scalar = mean_{p in a} min_{q in b} ||p-q||^2 + mean_{q in b} min_{p in a} ||q-p||^2
    Note: returns the mean of squared distances (L2^2). We also return sqrt-mean if requested later.
    """
    # tree from b to query a
    tree_b = cKDTree(b)
    dists_ab, _ = tree_b.query(a, k=1)
    tree_a = cKDTree(a)
    dists_ba, _ = tree_a.query(b, k=1)
    # use squared distances or raw? We'll return mean of squared distances for consistency with many CD implementations.
    # Here dists are Euclidean distances; square them to get L2^2 if you want. We'll return mean(dists) (L2) and mean(dists**2).
    mean_l2 = 0.5 * (np.mean(dists_ab) + np.mean(dists_ba))
    mean_l2sq = 0.5 * (np.mean(dists_ab**2) + np.mean(dists_ba**2))
    return {"cd_l2": mean_l2, "cd_l2sq": mean_l2sq}

def evaluate_generated_vs_test(generated_paths, test_paths, num_points=4096, out_csv=None):
    """
    Для каждого сгенерированного файла находит минимальный Chamfer к тестовым.
    Возвращает summary dict и (optionally) пишет CSV with per-generated-file info.
    """
    # Preload & sample test clouds (to speed repeated queries)
    print(f"Loading and sampling {len(test_paths)} test clouds...")
    test_clouds = []
    for tp in tqdm(test_paths, desc="loading test"):
        pts = load_pointcloud_auto(tp)
        pts = sample_points(pts, num_points)
        test_clouds.append( (os.path.basename(tp), pts) )

    results = []
    # For speed, build cKDTree for each test cloud once
    test_trees = [cKDTree(pts) for _, pts in test_clouds]

    print(f"Evaluating {len(generated_paths)} generated clouds against {len(test_paths)} test clouds...")
    for gen_path in tqdm(generated_paths, desc="generated"):
        gen_name = os.path.basename(gen_path)
        try:
            gen_pts = load_pointcloud_auto(gen_path)
        except Exception as e:
            print(f"Skipping {gen_path}, load error: {e}")
            continue
        gen_pts = sample_points(gen_pts, num_points)

        # compute CD to each test cloud (we'll compute symmetric CD approx via two queries)
        best_cd = float("inf")
        best_test = None

        # To reduce repeated work, we compute distances from gen to test (gen->test) and test->gen separately:
        # For each test tree, query nearest distances from gen points (g->t)
        for (tname, tpts), ttree in zip(test_clouds, test_trees):
            d_gen_to_test, _ = ttree.query(gen_pts, k=1)  # distances array len=num_points
            # for test->gen we need tree on gen
            # compute only if candidate is promising? simple approach compute both
            tree_gen = cKDTree(gen_pts)
            d_test_to_gen, _ = tree_gen.query(tpts, k=1)
            cd_l2 = 0.5 * (np.mean(d_gen_to_test) + np.mean(d_test_to_gen))
            # cd_l2sq optional
            if cd_l2 < best_cd:
                best_cd = cd_l2
                best_test = tname

        results.append({
            "gen_file": gen_name,
            "gen_path": gen_path,
            "best_test": best_test,
            "best_cd_l2": best_cd
        })

    # Compute summary stats on best_cd across generated set
    cds = np.array([r["best_cd_l2"] for r in results], dtype=np.float32)
    summary = {
        "n_generated": len(results),
        "mean_best_cd": float(np.mean(cds)) if len(cds)>0 else None,
        "median_best_cd": float(np.median(cds)) if len(cds)>0 else None,
        "std_best_cd": float(np.std(cds)) if len(cds)>0 else None,
        "min_best_cd": float(np.min(cds)) if len(cds)>0 else None,
        "max_best_cd": float(np.max(cds)) if len(cds)>0 else None
    }

    # Pairwise diversity among generated clouds (mean pairwise CD)
    print("Computing pairwise diversity (pairwise Chamfer among generated samples)...")
    pairwise = []
    G = len(results)
    # Pre-load sampled generated clouds for pairwise
    gen_sampled = []
    for r in results:
        pts = load_pointcloud_auto(r["gen_path"])
        pts = sample_points(pts, num_points)
        gen_sampled.append(pts)
    # compute upper-triangle
    for i in tqdm(range(G), desc="pairwise"):
        for j in range(i+1, G):
            cd = chamfer_distance_np(gen_sampled[i], gen_sampled[j])["cd_l2"]
            pairwise.append(cd)
    if len(pairwise) > 0:
        summary["pairwise_mean_cd"] = float(np.mean(pairwise))
        summary["pairwise_median_cd"] = float(np.median(pairwise))
        summary["pairwise_std_cd"] = float(np.std(pairwise))
    else:
        summary["pairwise_mean_cd"] = None

    # Save CSV
    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["gen_file","gen_path","best_test","best_cd_l2"])
            for r in results:
                writer.writerow([r["gen_file"], r["gen_path"], r["best_test"], f"{r['best_cd_l2']:.6f}"])
        print(f"Wrote results to {out_csv}")

    return summary, results

def find_files(dir_path, exts=(".ply", ".npy")):
    files = []
    for fn in os.listdir(dir_path):
        if fn.lower().endswith(exts):
            files.append(os.path.join(dir_path, fn))
    files.sort()
    return files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated pointclouds vs test set using Chamfer Distance")
    parser.add_argument("--generated_dir", type=str, default="results/visualization/21", help="folder with generated .ply or .npy")
    parser.add_argument("--test_dir", type=str, default="data/normalized_npy/test", help="folder with test .npy pointclouds")
    parser.add_argument("--num_points", type=int, default=4096, help="number of points to sample for comparison")
    parser.add_argument("--out_csv", type=str, default="results/visualization/21/eval_generated_vs_test.csv", help="where to save csv")
    args = parser.parse_args()

    if not os.path.exists(args.generated_dir):
        raise SystemExit(f"Generated dir not found: {args.generated_dir}")
    if not os.path.exists(args.test_dir):
        raise SystemExit(f"Test dir not found: {args.test_dir}")

    gen_files = find_files(args.generated_dir, exts=(".ply",".npy"))
    test_files = find_files(args.test_dir, exts=(".npy",))

    if len(gen_files) == 0:
        raise SystemExit("No generated files found in generated_dir.")

    print("Found generated:", len(gen_files), "files. Found test:", len(test_files), "files.")
    summary, results = evaluate_generated_vs_test(gen_files, test_files, num_points=args.num_points, out_csv=args.out_csv)

    print("\n=== SUMMARY ===")
    for k,v in summary.items():
        print(f"{k}: {v}")
    print("\nPer-file results saved to:", args.out_csv)
