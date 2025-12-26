import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from sklearn.decomposition import PCA
import os

def calculate_leaf_area_and_visualize(points, cluster_idx, output_dir):
    pca = PCA(n_components=2)
    points_2d = pca.fit_transform(points)
    tri = Delaunay(points_2d)
    area = 0
    for simplex in tri.simplices:
        p1, p2, p3 = points_2d[simplex]
        triangle_area = 0.5 * abs(
            p1[0] * (p2[1] - p3[1]) +
            p2[0] * (p3[1] - p1[1]) +
            p3[0] * (p1[1] - p2[1])
        )
        area += triangle_area
    plt.figure(figsize=(8, 6))
    plt.triplot(points_2d[:, 0], points_2d[:, 1], tri.simplices, color='blue', alpha=0.8)
    plt.scatter(points_2d[:, 0], points_2d[:, 1], c='red', s=10, label='Leaf Points')
    plt.title(f"Leaf {cluster_idx} - Delaunay Triangulation")
    plt.xlabel("PCA X")
    plt.ylabel("PCA Y")
    plt.legend()
    triangulation_output_path = os.path.join(output_dir, f"leaf_{cluster_idx}_triangulation.png")
    plt.savefig(triangulation_output_path)
    plt.close()
    print(f"叶片簇 {cluster_idx} 的三角剖分结果已保存至: {triangulation_output_path}")
    return area


file_path = "../normal vector/populus1_normals.txt"
data = np.loadtxt(file_path)
xyz = data[:, :3]
labels = data[:, 6]
leaf_points = xyz[labels == 1]

bandwidth = estimate_bandwidth(leaf_points, quantile=0.07, n_samples=500)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(leaf_points)


cluster_centers = ms.cluster_centers_
cluster_labels = ms.labels_
n_clusters = len(np.unique(cluster_labels))

print(f"检测到的叶片簇数: {n_clusters}")

output_dir = "leaf_visualizations"
os.makedirs(output_dir, exist_ok=True)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
for cluster_idx in range(n_clusters):
    cluster_points = leaf_points[cluster_labels == cluster_idx]
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f"leaf {cluster_idx}")

ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], c='red', marker='x', s=100, label="Centroids")
ax.set_title("MeanShift Clustering of Leaf Points")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.legend()
overall_output_path = os.path.join(output_dir, "overall_clustering.png")
plt.savefig(overall_output_path)
plt.close(fig)
print(f"整体聚类结果已保存至: {overall_output_path}")


areas_output_path = os.path.join(output_dir, "leaf_areas.txt")
with open(areas_output_path, "w") as f:
    f.write("Leaf Cluster Areas:\n")

for cluster_idx in range(n_clusters):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    cluster_points = leaf_points[cluster_labels == cluster_idx]
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f"leaf {cluster_idx}", color=plt.cm.jet(cluster_idx / n_clusters))
    ax.set_title(f"Leaf {cluster_idx}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.legend()
    cluster_output_path = os.path.join(output_dir, f"leaf_{cluster_idx}.png")
    plt.savefig(cluster_output_path)
    plt.close(fig)
    print(f"叶片簇 {cluster_idx} 可视化已保存至: {cluster_output_path}")
    area = calculate_leaf_area_and_visualize(cluster_points, cluster_idx, output_dir)
    print(f"叶片簇 {cluster_idx} 的面积: {area:.8f}")
    with open(areas_output_path, "a") as f:
        f.write(f"Leaf {cluster_idx}: {area:.8f}\n")

print(f"叶片簇面积已保存至: {areas_output_path}")
