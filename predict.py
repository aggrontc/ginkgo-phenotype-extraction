import os
import argparse
import numpy as np
import torch
from model import PointNeXt

def load_and_sample(file_path, num_points):

    data = np.loadtxt(file_path)
    if data.shape[1] < 6:
        raise ValueError("输入文件需至少包含 6 列: x,y,z,nx,ny,nz")
    N = data.shape[0]
    if N >= num_points:
        idxs = np.random.choice(N, num_points, replace=False)
    else:
        idxs = np.random.choice(N, num_points, replace=True)
    sampled = data[idxs, :6]
    return sampled, idxs


def predict(file_path, model_path, num_points, device, output_path):

    model = PointNeXt(num_classes=2, use_normal=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    points, idxs = load_and_sample(file_path, num_points)

    pts = torch.from_numpy(points.astype(np.float32)).unsqueeze(0).to(device)


    with torch.no_grad():
        outputs = model(pts)
        preds = outputs.squeeze(0).argmax(dim=1).cpu().numpy()


    data = np.loadtxt(file_path)
    N = data.shape[0]
    labels = np.zeros(N, dtype=np.int64)
    labels[idxs] = preds

    result = np.concatenate([data[:, :6], labels.reshape(-1, 1)], axis=1)
    header = 'x y z nx ny nz pred_label'
    np.savetxt(output_path, result, fmt='%.6f %.6f %.6f %.6f %.6f %.6f %d', header=header)
    print(f"Prediction saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point cloud prediction using PointNeXt')
    parser.add_argument('--input_file',  type=str, required=True, help='输入点云 txt 文件，列格式：x y z nx ny nz [可选额外列]')
    parser.add_argument('--model_path',  type=str, required=True, help='训练好的模型权重路径')
    parser.add_argument('--num_points',  type=int, default=2048, help='模型输入点数')
    parser.add_argument('--output_file', type=str, default='prediction.txt', help='预测结果保存路径')
    parser.add_argument('--use_gpu',     action='store_true',    help='是否使用 GPU')
    args = parser.parse_args()

    device = torch.device('cuda' if (args.use_gpu and torch.cuda.is_available()) else 'cpu')
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    predict(args.input_file, args.model_path, args.num_points, device, args.output_file)

