import open3d as o3d
import numpy as np
import os
import glob
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA


def compute_angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    angle_rad = np.arccos(dot_product / (norm_v1 * norm_v2))
    return angle_rad


def fit_line_with_pca(points):
    points = np.array(points)
    pca = PCA(n_components=1)
    pca.fit(points)
    line_direction = pca.components_[0]
    return line_direction

def compute_branch_inclination_angle(points, trunk_points, cls_labels):
    trunk_direction = fit_line_with_pca(trunk_points)

    branch_points = [point for point, cls in zip(points, cls_labels) if cls != 0]
    branch_direction = fit_line_with_pca(branch_points)

    angle_rad = compute_angle_between_vectors(trunk_direction, branch_direction)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def has_significant_inclination(angle_deg, threshold=5):
    return angle_deg > threshold

def load_point_cloud_from_txt(file_path):
    points = []
    normals = []
    colors = []
    cls_labels = []

    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split()
            x, y, z = float(data[0]), float(data[1]), float(data[2])
            nx, ny, nz = float(data[3]), float(data[4]), float(data[5])
            cls = float(data[6])

            points.append([x, y, z])
            normals.append([nx, ny, nz])
            colors.append([0.5, 0.5, 0.5])
            cls_labels.append(cls)

    points = np.array(points)
    normals = np.array(normals)
    colors = np.array(colors)

    return points, normals, colors, cls_labels


def classify_branches_by_level_hierarchical(points, trunk_points, distance_threshold=5.0):
    trunk_centroid = np.mean(trunk_points, axis=0)
    distances = np.linalg.norm(points - trunk_centroid, axis=1)
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)  # 设置最大距离
    labels = clustering.fit_predict(distances.reshape(-1, 1))
    return labels



def load_all_point_clouds(directory):
    txt_files = glob.glob(os.path.join(directory, "*.txt"))
    all_data = []
    for file_path in txt_files:
        points, normals, colors, cls_labels = load_point_cloud_from_txt(file_path)
        all_data.append((file_path, points, normals, colors, cls_labels))
    return all_data



def save_results_to_csv(results, output_dir="result"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df = pd.DataFrame(results,
                      columns=["File Path", "Branch Inclination Angle (degrees)", "Significant Inclination Detected",
                               "Branch Level"])
    output_file = os.path.join(output_dir, "branch_inclination_results.csv")
    df.to_csv(output_file, index=False)
    print(f"结果已保存到：{output_file}")


if __name__ == "__main__":
    all_point_cloud_data = load_all_point_clouds("Test data")
    results = []
    for file_path, points, normals, colors, cls_labels in all_point_cloud_data:
        trunk_points = points[np.array(cls_labels) == 0]
        inclination_angle = compute_branch_inclination_angle(points, trunk_points, cls_labels)
        print(f"文件：{file_path}, 计算得到的枝倾角：{inclination_angle} 度")
        significant_inclination = has_significant_inclination(inclination_angle)
        print("检测到显著的枝倾角。" if significant_inclination else "未检测到显著的枝倾角。")
        branch_levels = classify_branches_by_level_hierarchical(points, trunk_points)
        level = np.max(branch_levels)  # 获取层级中最大的值，表示最远的枝
        print(f"枝的级别：{level}")
        results.append([file_path, inclination_angle, "Yes" if significant_inclination else "No", level])
    save_results_to_csv(results)
