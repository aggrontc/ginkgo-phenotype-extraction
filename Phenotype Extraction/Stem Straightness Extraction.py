import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def calculate_straightness(trunk_points):
    X = trunk_points[['x', 'y']].values  # 只考虑横截面 (x, y)
    Z = trunk_points['z'].values.reshape(-1, 1)  # 高度作为预测目标
    model = LinearRegression()
    model.fit(X, Z)
    a, b = model.coef_[0]
    c = model.intercept_[0]
    Z_pred = model.predict(X)
    RSS = np.sum((Z - Z_pred) ** 2)
    TSS = np.sum((Z - np.mean(Z)) ** 2)
    straightness = 1 - (RSS / TSS)
    return straightness


def process_files_in_folder(folder_path):
    results = []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith('.txt'):
            print(f"正在处理文件: {file_name}")
            try:
                data = pd.read_csv(file_path, header=None, delimiter=' ')
                data.columns = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'label']
                trunk_points = data[data['label'] == 0]
                if len(trunk_points) > 10:
                    straightness = calculate_straightness(trunk_points)
                    results.append((file_name, straightness))
                else:
                    print(f"文件 {file_name} 树干点过少，跳过。")
                    results.append((file_name, None))
            except Exception as e:
                print(f"文件 {file_name} 处理失败: {e}")
                results.append((file_name, None))

    output_file = os.path.join(folder_path, 'straightness_results.csv')
    results_df = pd.DataFrame(results, columns=['File Name', 'Straightness'])
    results_df.to_csv(output_file, index=False)
    print(f"通直度计算完成，结果保存在: {output_file}")

data_folder = "Test data"
process_files_in_folder(data_folder)
