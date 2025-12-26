import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from PIL import Image
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QGroupBox, QFormLayout, QMessageBox, QProgressDialog,
                             QTabWidget, QToolBar, QAction, QStatusBar, QSpacerItem,
                             QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap


plt.rcParams['toolbar'] = 'None'


class TrunkFittingWorker(QThread):
    finished = pyqtSignal(np.ndarray, np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, full_point_cloud, k=12, voxel_size=0.005, fallback=True):
        super().__init__()
        self.full_point_cloud = full_point_cloud
        self.k = k
        self.voxel_size = voxel_size
        self.fallback = fallback

    def run(self):
        try:
            branch_points = self.full_point_cloud[self.full_point_cloud[:, 6] == 0][:, :3]
            if len(branch_points) == 0:
                self.error.emit("No branch points found (label=0), cannot fit trunk")
                return

            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(branch_points))
            pcd = pcd.voxel_down_sample(self.voxel_size)
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            processed_points = np.asarray(pcd.points)
            skeleton_pts = self.extract_skeleton_points(pcd, self.k, self.fallback)
            self.finished.emit(processed_points, skeleton_pts)

        except Exception as e:
            self.error.emit(f"Trunk fitting failed: {str(e)}")

    def extract_skeleton_points(self, pcd, k=10, slice_fallback=False):
        pts = np.asarray(pcd.points)
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        G = nx.Graph()
        for i, p in enumerate(pts):
            _, idx, dist = kdtree.search_knn_vector_3d(p, k + 1)
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
            for i in range(num_slices - 1):
                mask = (z_vals >= slices[i]) & (z_vals < slices[i + 1])
                if np.any(mask):
                    centers.append(pts[mask].mean(axis=0))
            return np.array(centers)


class LeafSegmentationWorker(QThread):
    finished = pyqtSignal(np.ndarray, np.ndarray, str)
    error = pyqtSignal(str)

    def __init__(self, full_point_cloud, input_path, output_dir="result"):
        super().__init__()
        self.full_point_cloud = full_point_cloud
        self.input_path = input_path
        self.output_dir = output_dir
        self.m_init = None
        self.k_nn = 30
        self.tol = 1e-4
        self.max_iter = 100
        self.merge_ratio = 5
        self.seed = 42
        self.sphere_r = 0.002

    def run(self):
        try:
            data = self.full_point_cloud.copy()
            xyz = data[:, :3]
            normals = data[:, 3:6]
            orig_lbl = data[:, 6].astype(int)

            leaf_idx = np.where(orig_lbl == 1)[0]
            if len(leaf_idx) == 0:
                self.error.emit("No leaf points found (label=1), cannot segment leaves")
                return
            leaf_points = xyz[leaf_idx]

            centers = self.fps(leaf_points, self.m_init, self.seed)
            m = centers.shape[0]

            nbrs_k = NearestNeighbors(n_neighbors=self.k_nn + 1).fit(leaf_points)
            dists_k, idxs_k = nbrs_k.kneighbors(centers)
            B_local = dists_k[:, 1:].mean(axis=1)

            for it in range(self.max_iter):
                prev_centers = centers.copy()
                diff = leaf_points[None, :, :] - centers[:, None, :]
                d2 = np.sum(diff ** 2, axis=2)
                W = np.exp(-d2 / (B_local[:, None] ** 2))
                shifts = (W[:, :, None] * diff).sum(axis=1) / (W.sum(axis=1)[:, None] + 1e-8)
                centers += shifts
                max_shift = np.max(np.linalg.norm(centers - prev_centers, axis=1))
                if max_shift < self.tol:
                    break

            nbrs_cent = NearestNeighbors(n_neighbors=2).fit(centers)
            dists_cent, idxs_cent = nbrs_cent.kneighbors(centers)
            keep = []
            for i, (d, j) in enumerate(zip(dists_cent[:, 1], idxs_cent[:, 1])):
                B_pair = (B_local[i] + B_local[j]) / 2
                if d > (B_pair * self.merge_ratio):
                    keep.append(i)
            centers = centers[keep]
            k_final = centers.shape[0]

            nbrs_final = NearestNeighbors(n_neighbors=1).fit(centers)
            _, nn_idx = nbrs_final.kneighbors(leaf_points)
            cluster_of_leaf = nn_idx.flatten() + 1

            new_lbls = np.zeros(data.shape[0], dtype=int)
            new_lbls[leaf_idx] = cluster_of_leaf
            out = np.hstack([xyz, normals, orig_lbl.reshape(-1, 1), new_lbls.reshape(-1, 1)])
            fname_base = os.path.basename(self.input_path).replace('.txt', '')
            os.makedirs(self.output_dir, exist_ok=True)
            out_txt = os.path.join(self.output_dir, f"{fname_base}_segmented.txt")
            np.savetxt(out_txt, out, fmt="%.6f")

            colors = np.zeros_like(xyz)
            colors[orig_lbl == 0] = [0.7, 0.7, 0.7]
            rng = np.random.default_rng(self.seed)
            leaf_colors = rng.random((k_final, 3))
            for cid in range(1, k_final + 1):
                colors[new_lbls == cid] = leaf_colors[cid - 1]

            self.finished.emit(xyz, colors, out_txt)

        except Exception as e:
            self.error.emit(f"Leaf segmentation failed: {str(e)}")

    def fps(self, points, m, seed=None):
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


class PointCloudCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=6, dpi=100, view_elev=20, view_azim=70):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111, projection='3d')
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()
        self._hide_axis()
        self.view_elev = view_elev
        self.view_azim = view_azim

    def _hide_axis(self):
        self.axes.set_axis_off()

    def clear(self):
        self.fig.clear()
        self.axes = self.fig.add_subplot(111, projection='3d')
        self._hide_axis()
        self.axes.view_init(elev=self.view_elev, azim=self.view_azim)
        self.draw()

    def set_view(self, elev, azim):
        self.view_elev = elev
        self.view_azim = azim
        self.axes.view_init(elev=elev, azim=azim)
        self.draw()

    def plot_point_cloud(self, x, y, z, cmap='jet'):
        self.clear()
        z_normalized = (z - np.min(z)) / (np.max(z) - np.min(z)) if np.max(z) != np.min(z) else np.zeros_like(z)
        scatter = self.axes.scatter(
            x, y, z, c=z_normalized, cmap=cmap, s=2, alpha=0.8, edgecolors='none'
        )
        cbar = self.fig.colorbar(scatter, ax=self.axes, pad=0.05)
        cbar.set_label('Relative Height (High → Low)', rotation=270, labelpad=20)
        self._set_scale(x, y, z)
        self.draw()

    def plot_branch_leaf_separation(self, x, y, z, labels):
        self.clear()
        branch_mask = labels == 0
        leaf_mask = labels == 1
        branch_count = np.sum(branch_mask)
        leaf_count = np.sum(leaf_mask)
        point_size = 2 if len(x) > 50000 else 5

        self.axes.scatter(
            x[branch_mask], y[branch_mask], z[branch_mask],
            c='blue', s=point_size, alpha=0.8, edgecolors='none',
            label=f'Branches ({branch_count} points)'
        )
        self.axes.scatter(
            x[leaf_mask], y[leaf_mask], z[leaf_mask],
            c='green', s=point_size, alpha=0.8, edgecolors='none',
            label=f'Leaves ({leaf_count} points)'
        )
        self.axes.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        self._set_scale(x, y, z)
        self.axes.set_title(f'Poplar Point Cloud Branch-Leaf Separation Visualization (Total {len(x)} points)')
        self.draw()

    def plot_trunk_skeleton(self, processed_points, skeleton_points):
        self.clear()
        self.axes.scatter(
            processed_points[:, 0], processed_points[:, 1], processed_points[:, 2],
            c='lightgray', s=1, alpha=0.6, edgecolors='none', label='Preprocessed Branch Points'
        )
        self.axes.plot(
            skeleton_points[:, 0], skeleton_points[:, 1], skeleton_points[:, 2],
            c='red', linewidth=2, label=f'Trunk Skeleton (Length: {len(skeleton_points)})'
        )
        self.axes.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        self.axes.set_title('Poplar Trunk Skeleton Fitting Result')
        all_points = np.vstack([processed_points, skeleton_points])
        self._set_scale(all_points[:, 0], all_points[:, 1], all_points[:, 2])
        self.draw()

    def plot_leaf_segmentation(self, xyz, colors):
        self.clear()
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        self.axes.scatter(x, y, z, c=colors, s=2, alpha=0.8, edgecolors='none')
        self.axes.set_title('Poplar Leaf Segmentation Result')
        self.axes.text2D(0.05, 0.95, "Gray: Branches | Colored: Leaves (Different colors = different leaves)",
                         transform=self.axes.transAxes, bbox=dict(facecolor='white', alpha=0.8))
        self._set_scale(x, y, z)
        self.draw()

    def _set_scale(self, x, y, z):
        max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
        mid_x = (x.max() + x.min()) / 2
        mid_y = (y.max() + y.min()) / 2
        mid_z = (z.max() + z.min()) / 2
        self.axes.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        self.axes.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        self.axes.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)


class TreeVisualizationSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.selected_file = None
        self.full_point_cloud = None
        self.worker = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Poplar Seedling Point Cloud Structural Phenotype Extraction and Processing System")
        self.setGeometry(100, 100, 1400, 800)

        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        open_action = QAction("Open File", self)
        open_action.triggered.connect(self.select_file)
        open_action.setStatusTip("Select point cloud Test data file (.txt)")
        toolbar.addAction(open_action)

        save_action = QAction("Save Result", self)
        save_action.triggered.connect(self.save_result)
        save_action.setStatusTip("Save current visualization result as image")
        save_action.setEnabled(False)
        toolbar.addAction(save_action)
        self.save_action = save_action

        toolbar.addSeparator()

        reset_view_action = QAction("Reset View", self)
        reset_view_action.triggered.connect(self.reset_view)
        reset_view_action.setStatusTip("Reset 3D view angle")
        toolbar.addAction(reset_view_action)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget, spacing=10)

        left_panel = QWidget()
        left_panel.setFixedWidth(220)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.addWidget(left_panel)

        file_group = QGroupBox("File Operations")
        file_group.setCheckable(True)
        file_group.setChecked(True)
        file_layout = QVBoxLayout()
        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setWordWrap(True)
        self.file_path_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        file_layout.addWidget(self.file_path_label)
        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)


        func_group = QGroupBox("Function Operations")
        func_group.setCheckable(True)
        func_group.setChecked(True)
        func_layout = QVBoxLayout()
        btn_style = "QPushButton {padding: 8px; margin: 2px;}"

        self.visual_btn = QPushButton("Visualization")
        self.visual_btn.setStyleSheet(btn_style)
        self.visual_btn.clicked.connect(self.visualize_point_cloud)
        self.visual_btn.setEnabled(False)

        self.separate_btn = QPushButton("Separation")
        self.separate_btn.setStyleSheet(btn_style)
        self.separate_btn.clicked.connect(self.separate_branch_leaf)
        self.separate_btn.setEnabled(False)

        self.trunk_btn = QPushButton("Trunk Fitting")
        self.trunk_btn.setStyleSheet(btn_style)
        self.trunk_btn.clicked.connect(self.fit_trunk_skeleton)
        self.trunk_btn.setEnabled(False)

        self.leaf_btn = QPushButton("Leaf Segmentation")
        self.leaf_btn.setStyleSheet(btn_style)
        self.leaf_btn.clicked.connect(self.segment_leaves)
        self.leaf_btn.setEnabled(False)
        func_layout.addWidget(self.visual_btn)
        func_layout.addWidget(self.separate_btn)
        func_layout.addWidget(self.trunk_btn)
        func_layout.addWidget(self.leaf_btn)
        func_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        func_group.setLayout(func_layout)
        left_layout.addWidget(func_group)
        status_group = QGroupBox("Status Information")
        status_group.setCheckable(True)
        status_group.setChecked(True)
        status_layout = QFormLayout()
        status_layout.setRowWrapPolicy(QFormLayout.DontWrapRows)
        self.file_info_label = QLabel("None")
        self.point_count_label = QLabel("0")
        self.branch_count_label = QLabel("0")
        self.leaf_count_label = QLabel("0")
        status_layout.addRow(QLabel("Current File:"), self.file_info_label)
        status_layout.addRow(QLabel("Total Points:"), self.point_count_label)
        status_layout.addRow(QLabel("Branch Points:"), self.branch_count_label)
        status_layout.addRow(QLabel("Leaf Points:"), self.leaf_count_label)
        status_group.setLayout(status_layout)
        left_layout.addWidget(status_group)
        left_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        main_layout.addWidget(right_panel, stretch=1)
        self.view_tabs = QTabWidget()
        self.canvas_3d = PointCloudCanvas(width=10, height=6, dpi=100)
        self.canvas_side = PointCloudCanvas(width=10, height=6, dpi=100, view_elev=0, view_azim=90)
        self.canvas_top = PointCloudCanvas(width=10, height=6, dpi=100, view_elev=90, view_azim=0)
        self.view_tabs.addTab(self.canvas_3d, "3D View")
        self.view_tabs.addTab(self.canvas_side, "Side View")
        self.view_tabs.addTab(self.canvas_top, "Top View")
        right_layout.addWidget(self.view_tabs)
        control_bar = QHBoxLayout()
        self.rotate_btn = QPushButton("Rotate View")
        self.rotate_btn.setCheckable(True)
        self.rotate_btn.setStatusTip("Click and drag mouse to rotate view")
        self.zoom_btn = QPushButton("Zoom View")
        self.zoom_btn.setCheckable(True)
        self.zoom_btn.setStatusTip("Click and drag mouse to zoom view")
        control_bar.addWidget(self.rotate_btn)
        control_bar.addWidget(self.zoom_btn)
        control_bar.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        right_layout.addLayout(control_bar)
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")

    def select_file(self):
        default_dir = os.path.join(os.getcwd(), "normal vector")
        if not os.path.exists(default_dir):
            default_dir = os.getcwd()

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Point Cloud File", default_dir, "Point Cloud Files (*.txt);;All Files (*)"
        )

        if file_path:
            self.selected_file = file_path
            self.file_path_label.setText(f"Selected:\n{os.path.basename(file_path)}")
            self.file_info_label.setText(os.path.basename(file_path))
            self.statusBar.showMessage(f"File loaded: {os.path.basename(file_path)}")

            try:
                self.full_point_cloud = np.loadtxt(file_path)
                if self.full_point_cloud.shape[1] != 7:
                    raise ValueError(f"Data format error, expected 7 columns, got {self.full_point_cloud.shape[1]} columns")

                self.point_count = len(self.full_point_cloud)
                self.point_count_label.setText(str(self.point_count))

                labels = self.full_point_cloud[:, 6]
                branch_count = np.sum(labels == 0)
                leaf_count = np.sum(labels == 1)
                self.branch_count_label.setText(str(branch_count))
                self.leaf_count_label.setText(str(leaf_count))

                self.visual_btn.setEnabled(True)
                self.separate_btn.setEnabled(True)
                self.trunk_btn.setEnabled(True)
                self.leaf_btn.setEnabled(True)
                self.save_action.setEnabled(True)

            except Exception as e:
                self.statusBar.showMessage(f"File reading error: {str(e)}")
                self.point_count_label.setText(f"Error: {str(e)}")
                self.branch_count_label.setText("0")
                self.leaf_count_label.setText("0")
                self.visual_btn.setEnabled(False)
                self.separate_btn.setEnabled(False)
                self.trunk_btn.setEnabled(False)
                self.leaf_btn.setEnabled(False)
                self.save_action.setEnabled(False)

    def visualize_point_cloud(self):
        if self.full_point_cloud is not None:
            try:
                x = self.full_point_cloud[:, 0]
                y = self.full_point_cloud[:, 1]
                z = self.full_point_cloud[:, 2]
                self.canvas_3d.plot_point_cloud(x, y, z)
                self.canvas_side.plot_point_cloud(x, y, z)
                self.canvas_top.plot_point_cloud(x, y, z)
                self.statusBar.showMessage("Point cloud visualization completed")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Visualization failed: {str(e)}")
                self.statusBar.showMessage(f"Visualization failed: {str(e)}")

    def separate_branch_leaf(self):
        if self.full_point_cloud is None:
            QMessageBox.warning(self, "Warning", "Please select a point cloud file first")
            return

        try:
            x = self.full_point_cloud[:, 0]
            y = self.full_point_cloud[:, 1]
            z = self.full_point_cloud[:, 2]
            labels = self.full_point_cloud[:, 6]
            self.canvas_3d.plot_branch_leaf_separation(x, y, z, labels)
            self.canvas_side.plot_branch_leaf_separation(x, y, z, labels)
            self.canvas_top.plot_branch_leaf_separation(x, y, z, labels)
            self.statusBar.showMessage("Branch-leaf separation visualization completed")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Branch-leaf separation failed: {str(e)}")
            self.statusBar.showMessage(f"Branch-leaf separation failed: {str(e)}")

    def fit_trunk_skeleton(self):
        if self.full_point_cloud is None:
            QMessageBox.warning(self, "Warning", "Please select a point cloud file first")
            return

        progress = QProgressDialog("Performing trunk fitting...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Processing")
        progress.setWindowModality(Qt.WindowModal)
        progress.setValue(10)

        self.worker = TrunkFittingWorker(
            self.full_point_cloud, k=12, voxel_size=0.005, fallback=True
        )
        self.worker.finished.connect(lambda p, s: self.on_trunk_fitted(p, s, progress))
        self.worker.error.connect(lambda e: self.on_worker_error(e, progress))
        self.worker.start()

    def segment_leaves(self):
        if self.full_point_cloud is None:
            QMessageBox.warning(self, "Warning", "Please select a point cloud file first")
            return

        progress = QProgressDialog("Performing leaf segmentation...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Processing")
        progress.setWindowModality(Qt.WindowModal)
        progress.setValue(10)

        self.worker = LeafSegmentationWorker(
            self.full_point_cloud, self.selected_file, output_dir="result"
        )
        self.worker.finished.connect(lambda xyz, colors, path: self.on_leaf_segmented(xyz, colors, path, progress))
        self.worker.error.connect(lambda e: self.on_worker_error(e, progress))
        self.worker.start()

    def on_trunk_fitted(self, processed_points, skeleton_points, progress):
        progress.setValue(100)
        self.canvas_3d.plot_trunk_skeleton(processed_points, skeleton_points)
        self.canvas_side.plot_trunk_skeleton(processed_points, skeleton_points)
        self.canvas_top.plot_trunk_skeleton(processed_points, skeleton_points)
        if not os.path.exists("result"):
            os.makedirs("result")
        output_path = os.path.join("result", f"skeleton_{os.path.basename(self.selected_file).split('.')[0]}.npy")
        np.save(output_path, skeleton_points)
        QMessageBox.information(self, "Completed", f"Trunk fitting succeeded, skeleton points saved to:\n{output_path}")
        self.statusBar.showMessage(f"Trunk fitting completed, results saved to: {output_path}")

    def on_leaf_segmented(self, xyz, colors, output_path, progress):
        progress.setValue(100)
        self.canvas_3d.plot_leaf_segmentation(xyz, colors)
        self.canvas_side.plot_leaf_segmentation(xyz, colors)
        self.canvas_top.plot_leaf_segmentation(xyz, colors)
        QMessageBox.information(self, "Completed", f"Leaf segmentation succeeded, results saved to:\n{output_path}")
        self.statusBar.showMessage(f"Leaf segmentation completed, results saved to: {output_path}")

    def on_worker_error(self, error_msg, progress):
        progress.close()
        QMessageBox.critical(self, "Error", error_msg)
        self.statusBar.showMessage(error_msg)

    def reset_view(self):
        self.canvas_3d.set_view(20, 70)
        self.canvas_side.set_view(0, 90)
        self.canvas_top.set_view(90, 0)
        self.statusBar.showMessage("View reset")

    def save_result(self):
        current_tab = self.view_tabs.currentWidget()
        if not current_tab:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", os.getcwd(), "PNG Images (*.png);;All Files (*)"
        )
        if file_path:
            current_tab.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            self.statusBar.showMessage(f"Image saved to: {file_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TreeVisualizationSystem()
    window.show()
    sys.exit(app.exec_())