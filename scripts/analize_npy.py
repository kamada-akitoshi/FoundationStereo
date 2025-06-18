import numpy as np
import argparse
import os
import cv2
import imageio.v2 as imageio
import open3d as o3d

def vis_disparity(disp):
    disp = np.copy(disp)
    disp[np.isinf(disp)] = 0
    disp[np.isnan(disp)] = 0
    disp_norm = (disp - np.min(disp)) / (np.max(disp) - np.min(disp) + 1e-6)
    disp_colored = cv2.applyColorMap((disp_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return disp_colored

def depth2xyzmap(depth, K):
    h, w = depth.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (i - cx) * depth / fx
    y = (j - cy) * depth / fy
    xyz = np.stack((x, y, depth), axis=-1)
    return xyz

def main(filename):
    # 固定パス
    out_dir = './test_outputs'
    intrinsic_file = './assets/K.txt'
    image_file = './assets/left.png'
    curr_dir='./'
    z_far = 10.0

    # disparity 読み込み
    npy_path = os.path.join(curr_dir, filename)
    disp = np.load(npy_path).squeeze()

    # カメラ行列とベースライン
    with open(intrinsic_file, 'r') as f:
        lines = f.readlines()
        K = np.array(list(map(float, lines[0].split()))).reshape(3, 3)
        baseline = float(lines[1])

    # 視差 → depth
    depth = K[0,0] * baseline / (disp + 1e-6)
    depth[disp <= 0] = 0  # 無効値

    # 震度マップ保存
    vis = vis_disparity(disp)
    vis_path = os.path.join(curr_dir, filename.replace('.npy', '_disp_vis.png'))
    imageio.imwrite(vis_path, vis)
    print(f"[INFO] Disparity map saved to: {vis_path}")

    # 点群生成
    xyz_map = depth2xyzmap(depth, K)
    color = imageio.imread(image_file)
    if color.ndim == 2:
        color = np.stack([color]*3, axis=-1)
    color = cv2.resize(color, (disp.shape[1], disp.shape[0]))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_map.reshape(-1,3))
    pcd.colors = o3d.utility.Vector3dVector((color.reshape(-1,3).astype(np.float32)) / 255.0)

    # zフィルタ
    z = np.asarray(pcd.points)[:, 2]
    mask = (z > 0) & (z < z_far)
    pcd = pcd.select_by_index(np.where(mask)[0])

    # 出力
    ply_path = os.path.join(curr_dir, filename.replace('.npy', '_cloud.ply'))
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"[INFO] Point cloud saved to: {ply_path}")

    # 表示
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='disparity .npy ファイル名（相対パス不要）')
    args = parser.parse_args()
    main(args.filename)
