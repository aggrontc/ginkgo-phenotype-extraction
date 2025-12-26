import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def visualize_tree_point_cloud(txt_file):
    if not os.path.exists(txt_file):
        print(f"错误：文件 {txt_file} 不存在！")
        return

    try:
        data = np.loadtxt(txt_file)
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        point_size = 2 if len(x) > 50000 else 5

        z_normalized = (z - np.min(z)) / (np.max(z) - np.min(z)) if np.max(z) != np.min(z) else np.zeros_like(z)
        scatter = ax.scatter(
            x, y, z,
            c=z_normalized,
            cmap='viridis_r',
            s=point_size,
            alpha=0.9,
            edgecolors='none'
        )

        ax.view_init(elev=20, azim=70)

        max_range = np.array([
            x.max() - x.min(),
            y.max() - y.min(),
            z.max() - z.min()
        ]).max()
        mid_x = (x.max() + x.min()) / 2
        mid_y = (y.max() + y.min()) / 2
        mid_z = (z.max() + z.min()) / 2
        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)


        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('高度（低→高）', rotation=270, labelpad=20)
        ax.set_title(f'银杏点云可视化 (共 {len(x)} 个点)')

        # 简化坐标轴标签（避免杂乱）
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"处理文件时出错：{str(e)}")


if __name__ == "__main__":
    tree_point_cloud = "normal vector/populus1_normals.txt"  # 替换为你的文件路径
    visualize_tree_point_cloud(tree_point_cloud)