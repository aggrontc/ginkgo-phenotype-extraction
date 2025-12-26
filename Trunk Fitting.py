import numpy as np
import open3d as o3d
import networkx as nx


def load_branch_points(file_path):
    data = np.loadtxt(file_path)
    return data[data[:, 6] == 0][:, :3]


def preprocess(points, voxel_size=0.005, nb_neighbors=20, std_ratio=2.0):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd


def extract_skeleton_points(pcd, k=10, slice_fallback=False):
    pts = np.asarray(pcd.points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    G = nx.Graph()
    for i, p in enumerate(pts):
        _, idx, dist = kdtree.search_knn_vector_3d(p, k+1)
        for j, d in zip(idx[1:], dist[1:]):
            G.add_edge(i, j, weight=d)
    mst = nx.minimum_spanning_tree(G)

    z_vals = pts[:, 2]
    start, end = int(np.argmin(z_vals)), int(np.argmax(z_vals))
    try:
        path = nx.shortest_path(mst, source=start, target=end, weight='weight')
        return pts[path]
    except nx.NetworkXNoPath:
        if not slice_fallback:
            raise
        num_slices = 100
        z_min, z_max = z_vals.min(), z_vals.max()
        slices = np.linspace(z_min, z_max, num_slices)
        centers = []
        for i in range(num_slices-1):
            mask = (z_vals >= slices[i]) & (z_vals < slices[i+1])
            if np.any(mask):
                centers.append(pts[mask].mean(axis=0))
        return np.array(centers)


if __name__ == '__main__':
    input_path = r"normal vector/populus1_normals.txt"
    output_path = r"result/skeleton_1.npy"
    fallback = True
    k = 12
    pts = load_branch_points(input_path)
    pcd_clean = preprocess(pts, voxel_size=0.005)
    skeleton_pts = extract_skeleton_points(pcd_clean, k=k, slice_fallback=fallback)
    np.save(output_path, skeleton_pts)
    print(f"Skeleton saved to {output_path}, length={len(skeleton_pts)}")
    pcd_clean.paint_uniform_color([0.7, 0.7, 0.7])
    lines = [[i, i+1] for i in range(len(skeleton_pts)-1)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(skeleton_pts),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.paint_uniform_color([1.0, 0.0, 0.0])
    o3d.visualization.draw_geometries([pcd_clean, line_set],
                                      window_name='Skeleton Extraction',
                                      width=800, height=600)
