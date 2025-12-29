import numpy as np
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
import os
from PIL import Image


m_init = None
k_nn = 30
tol = 1e-4
max_iter = 100
merge_ratio = 5
seed = 42
img_width = 800
img_height = 600
sphere_r = 0.002

def fps(points, m, seed=None):
    rng = np.random.default_rng(seed)
    N = points.shape[0]
    m = m if m is not None else max(10, N // 500)
    centers_idx = np.empty(m, dtype=int)
    centers_idx[0] = rng.integers(N)
    dist_sq = np.sum((points - points[centers_idx[0]]) ** 2, axis=1)
    for i in range(1, m):
        centers_idx[i] = np.argmax(dist_sq)
        new_dist_sq = np.sum((points - points[centers_idx[i]]) ** 2, axis=1)
        dist_sq = np.minimum(dist_sq, new_dist_sq)
    return points[centers_idx]


def process_single_file(file_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    data = np.loadtxt(file_path)
    xyz = data[:, :3]
    normals = data[:, 3:6]
    orig_lbl = data[:, 6].astype(int)
    leaf_idx = np.where(orig_lbl == 1)[0]
    leaf_points = xyz[leaf_idx]
    n_leaf = leaf_points.shape[0]
    print(f"叶片点数量: {n_leaf}")
    centers = fps(leaf_points, m_init, seed=seed)
    m = centers.shape[0]
    print(f"初始采样 {m} 个中心点。")
    nbrs_k = NearestNeighbors(n_neighbors=k_nn + 1).fit(leaf_points)
    dists_k, idxs_k = nbrs_k.kneighbors(centers)
    B_local = dists_k[:, 1:].mean(axis=1)
    for it in range(max_iter):
        prev_centers = centers.copy()
        diff = leaf_points[None, :, :] - centers[:, None, :]
        d2 = np.sum(diff ** 2, axis=2)
        W = np.exp(-d2 / (B_local[:, None] ** 2))
        shifts = (W[:, :, None] * diff).sum(axis=1) / (W.sum(axis=1)[:, None] + 1e-8)
        centers += shifts
        max_shift = np.max(np.linalg.norm(centers - prev_centers, axis=1))
        if max_shift < tol:
            print(f"Mean Shift 在第 {it + 1} 轮收敛，最大位移 {max_shift:.4e}。")
            break
    else:
        print(f"未完全收敛，最终最大位移 {max_shift:.4e}。")
    nbrs_cent = NearestNeighbors(n_neighbors=2).fit(centers)
    dists_cent, idxs_cent = nbrs_cent.kneighbors(centers)
    keep = []
    for i, (d, j) in enumerate(zip(dists_cent[:, 1], idxs_cent[:, 1])):
        B_pair = (B_local[i] + B_local[j]) / 2
        if d > (B_pair * merge_ratio):
            keep.append(i)
    centers = centers[keep]
    B_local = B_local[keep]
    k_final = centers.shape[0]
    print(f"合并后保留 {k_final} 个中心。")
    nbrs_final = NearestNeighbors(n_neighbors=1).fit(centers)
    _, nn_idx = nbrs_final.kneighbors(leaf_points)
    cluster_of_leaf = nn_idx.flatten() + 1
    new_lbls = np.zeros(data.shape[0], dtype=int)
    new_lbls[leaf_idx] = cluster_of_leaf
    out = np.hstack([xyz, normals, orig_lbl.reshape(-1, 1), new_lbls.reshape(-1, 1)])
    fname_base = os.path.basename(file_path).replace('.txt', '')
    out_txt = os.path.join(output_dir, f"{fname_base}_segmented.txt")
    np.savetxt(out_txt, out, fmt="%.6f")
    print(f"分割完成，共 {k_final} 片叶；结果保存: {out_txt}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    colors = np.zeros_like(xyz)
    colors[orig_lbl == 0] = [0.7, 0.7, 0.7]
    rng = np.random.default_rng(seed)
    leaf_colors = rng.random((k_final, 3))
    for cid in range(1, k_final + 1):
        colors[new_lbls == cid] = leaf_colors[cid - 1]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    spheres = []
    for c in centers:
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_r)
        sph.translate(c)
        sph.paint_uniform_color([0, 0, 0])  # 中心点黑色
        spheres.append(sph)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=img_width, height=img_height)
    vis.add_geometry(pcd)
    for sph in spheres:
        vis.add_geometry(sph)
    ctr = vis.get_view_control()
    bounds = pcd.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    ctr.set_lookat(center)
    ctr.set_front([0.0, -1.0, 0.0])  # 前视方向沿-Y
    ctr.set_up([0.0, 0.0, 1.0])  # Z轴为上
    vis.poll_events()
    vis.update_renderer()
    img = vis.capture_screen_float_buffer(False)
    vis.destroy_window()
    img_arr = (np.asarray(img) * 255).astype(np.uint8)
    img_out = Image.fromarray(img_arr)
    out_png = os.path.join(output_dir, f"{fname_base}_segmented_zview.png")
    img_out.save(out_png)
    print(f"Z视图保存: {out_png}")


if __name__ == '__main__':
    input_file = "normal vector/populus2_normals.txt"
    output_directory = "result"


    if not os.path.exists(input_file):
        print(f"错误: 找不到文件 {input_file}")
    else:
        process_single_file(input_file, output_directory)
        print("文件处理完成。")