print("[DEBUG] スクリプト開始")

import argparse
print("[DEBUG] argparse インポート成功")

import os
print("[DEBUG] os インポート成功")

import sys
print("[DEBUG] sys インポート成功")

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(code_dir, ".."))
print("[DEBUG] sys.path に上位ディレクトリ追加")

import cv2
print("[DEBUG] cv2 インポート成功")

# import imageio
import imageio.v2 as imageio 

print("[DEBUG] imageio インポート成功")

import onnxruntime as ort
print("[DEBUG] onnxruntime インポート成功")
providers = ort.get_available_providers()
print(f"[DEBUG] 利用可能な ONNX Runtime プロバイダ: {providers}")

from core.utils.utils import InputPadder
print("[DEBUG] InputPadder インポート成功")

from Utils import *
print("[DEBUG] Utils モジュール インポート成功")

import logging
print("[DEBUG] logging インポート成功")

import torch
print("[DEBUG] torch インポート成功")

import torch.nn.functional as F
print("[DEBUG] torch.nn.functional インポート成功")

import open3d as o3d
print("[DEBUG] open3d インポート成功")

# ここまで来たらimportは成功しているはず
print("[DEBUG] すべてのimport完了")



code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(code_dir, ".."))
print("[DEBUG] すべてのimport完了")


if __name__ == "__main__":
    print("[DEBUG] __main__ 開始")
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_file', type=str, default='./assets/left.png')
    parser.add_argument('--right_file', type=str, default='./assets/right.png')
    parser.add_argument('--onnx_model', type=str, default='./pretrained_models/model_best.onnx')
    parser.add_argument('--out_dir', type=str, default='./test_outputs/')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    print("[DEBUG] 出力ディレクトリ作成済み or 存在")

    print(f"[DEBUG] 左画像読み込み: {args.left_file}")
    img0 = imageio.imread(args.left_file)
    print(f"[DEBUG] 右画像読み込み: {args.right_file}")
    img1 = imageio.imread(args.right_file)

    H, W = img0.shape[:2]
    print(f"[DEBUG] 画像サイズ: H={H}, W={W}")
    img0_ori = img0.copy()

    # print("[DEBUG] Tensor変換開始")
    # img0 = torch.as_tensor(img0).float()[None].permute(0, 3, 1, 2)
    # img1 = torch.as_tensor(img1).float()[None].permute(0, 3, 1, 2)
    # print(f"[DEBUG] img0 shape: {img0.shape}, img1 shape: {img1.shape}")


    # print("[DEBUG] パディング開始")
    # padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    # img0_pad, img1_pad = padder.pad(img0, img1)
    # print(f"[DEBUG] パディング後 img0_pad shape: {img0_pad.shape}, img1_pad shape: {img1_pad.shape}")

    # print("[DEBUG] numpy変換開始")
    # input0 = img0_pad.contiguous().cpu().numpy().astype(np.float32)
    # input1 = img1_pad.contiguous().cpu().numpy().astype(np.float32)





        # torch tensor化後
    img0 = torch.as_tensor(img0).float()[None].permute(0, 3, 1, 2)
    img1 = torch.as_tensor(img1).float()[None].permute(0, 3, 1, 2)

    print(f"[DEBUG] img0 shape: {img0.shape}, img1 shape: {img1.shape}")

    # モデルが固定で期待しているサイズにリサイズ
    target_size = (480, 640)  # H, W
    img0 = F.interpolate(img0, size=target_size, mode='bilinear', align_corners=False)
    img1 = F.interpolate(img1, size=target_size, mode='bilinear', align_corners=False)

    print(f"[DEBUG] リサイズ後 img0 shape: {img0.shape}, img1 shape: {img1.shape}")

    input0 = img0.contiguous().cpu().numpy().astype(np.float32).copy()
    input1 = img1.contiguous().cpu().numpy().astype(np.float32).copy()


    print(f"[DEBUG] input0 shape: {input0.shape}, dtype: {input0.dtype}")
    print(f"[DEBUG] input1 shape: {input1.shape}, dtype: {input1.dtype}")

    print("[DEBUG] ONNX Runtime セッション作成")
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    sess = ort.InferenceSession(args.onnx_model, sess_options,providers=['CUDAExecutionProvider'])
    print("[DEBUG] セッション作成完了")

    input_names = [i.name for i in sess.get_inputs()]
    output_name = sess.get_outputs()[0].name
    print(f"[DEBUG] 入力名: {input_names}")
    print(f"[DEBUG] 出力名: {output_name}")

    print("[DEBUG] 推論実行開始")
    outputs = sess.run([output_name], {input_names[0]: input0, input_names[1]: input1})
    print("[DEBUG] 推論実行完了")
    print("[DEBUG] 推論結果取得・Tensor変換開始")
    # disp = torch.tensor(outputs[0]).squeeze().cpu().numpy()
    # if disp.ndim == 3:
    #     disp = disp[0]  # (1, H, W) -> (H, W)
    disp = np.array(outputs[0]).squeeze()
    if disp.ndim == 3:
        disp = disp[0]
    disp = disp.astype(np.float32)  # 明示的に型変換して所有権を明確化

    print(f"[DEBUG] 推論結果 disp shape: {disp.shape}")

    print("[DEBUG] 元画像サイズにリサイズ")
    disp_resized = cv2.resize(disp, (W, H), interpolation=cv2.INTER_LINEAR).copy()
    
    print(f"[DEBUG] disp_resized shape: {disp_resized.shape}")

    # 保存
    np.save(f"{args.out_dir}/disp_resized.npy", disp_resized)
    np.savetxt(f"{args.out_dir}/disp_resized.txt", disp_resized, fmt="%.4f")
    print(f"[DEBUG] 視差マップ保存完了: {args.out_dir}/disp_resized.npy (.txt)")

    del sess
    del input0
    del input1
    del outputs
    del img0
    del img1
    del disp
    del disp_resized

    import gc
    gc.collect()
    print("[DEBUG] メモリ解放完了")