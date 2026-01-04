import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from skimage import io
from skimage.metrics import structural_similarity as ssim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from model import DenoiseNet

torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor()])


class Logger(object):
    def __init__(self, filename="log.txt"):
        self.console = sys.stdout
        self.file = open(filename, "a")

    def write(self, message):
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        self.console.flush()
        self.file.flush()

def get_local_topk_pixel_bank(feature_map: torch.Tensor,
                              raw_image: torch.Tensor,
                              k: int = 20,
                              window: int = 40,
                              patch_size: int = 5) -> torch.Tensor:

    H, W, D = feature_map.shape
    pixel_bank = []
    cached_edge_feat = None
    cached_edge_rgb = None

    # 保证类型一致
    if isinstance(feature_map, np.ndarray):
        feature_map = torch.from_numpy(feature_map)
    if isinstance(raw_image, np.ndarray):
        raw_image = torch.from_numpy(raw_image)

    feature_map = feature_map.to(torch.float32).contiguous().to(device)
    raw_image = raw_image.to(torch.float32).contiguous().to(device)

    offset_left = window // 2
    offset_right = window - offset_left  #兼容奇数和偶数的window
    C = raw_image.shape[-1]  # 自动适配通道数

    for i in range(H):
        for j in range(W):
            center_feat = feature_map[i, j]

            if i < patch_size // 2 or i >= H - patch_size // 2 or j < patch_size // 2 or j >= W - patch_size // 2:
                if cached_edge_feat is None:
                    # 第一次遇到边缘像素，构建缓存
                    top = feature_map[0:patch_size, patch_size:-patch_size, :].reshape(-1, D)
                    bottom = feature_map[-patch_size:, patch_size:-patch_size, :].reshape(-1, D)
                    left = feature_map[patch_size:-patch_size, 0:patch_size, :].reshape(-1, D)
                    right = feature_map[patch_size:-patch_size, -patch_size:, :].reshape(-1, D)
                    tl = feature_map[0:patch_size, 0:patch_size, :].reshape(-1, D)
                    tr = feature_map[0:patch_size, -patch_size:, :].reshape(-1, D)
                    bl = feature_map[-patch_size:, 0:patch_size, :].reshape(-1, D)
                    br = feature_map[-patch_size:, -patch_size:, :].reshape(-1, D)

                    cached_edge_feat = torch.cat([top, bottom, left, right, tl, tr, bl, br], dim=0)

                    top_rgb = raw_image[0:patch_size, patch_size:-patch_size, :].reshape(-1, C)
                    bottom_rgb = raw_image[-patch_size:, patch_size:-patch_size, :].reshape(-1, C)
                    left_rgb = raw_image[patch_size:-patch_size, 0:patch_size, :].reshape(-1, C)
                    right_rgb = raw_image[patch_size:-patch_size, -patch_size:, :].reshape(-1, C)
                    tl_rgb = raw_image[0:patch_size, 0:patch_size, :].reshape(-1, C)
                    tr_rgb = raw_image[0:patch_size, -patch_size:, :].reshape(-1, C)
                    bl_rgb = raw_image[-patch_size:, 0:patch_size, :].reshape(-1, C)
                    br_rgb = raw_image[-patch_size:, -patch_size:, :].reshape(-1, C)

                    cached_edge_rgb = torch.cat([top_rgb, bottom_rgb, left_rgb, right_rgb, tl_rgb, tr_rgb, bl_rgb, br_rgb], dim=0)

                # 使用缓存
                local_feat = cached_edge_feat
                local_rgb = cached_edge_rgb

            else:
                # 使用正常 window 策略
                i_start = i - offset_left
                i_end = i + offset_right
                j_start = j - offset_left
                j_end = j + offset_right

                # 偏移 window 以保持窗口大小
                if i_start < 0:
                    i_end += -i_start
                    i_start = 0
                if i_end > H:
                    i_start -= i_end - H
                    i_end = H

                if j_start < 0:
                    j_end += -j_start
                    j_start = 0
                if j_end > W:
                    j_start -= j_end - W
                    j_end = W

                local_feat = feature_map[i_start:i_end, j_start:j_end].reshape(-1, D)
                local_rgb = raw_image[i_start:i_end, j_start:j_end].reshape(-1, C)

            dists = torch.norm(local_feat - center_feat, dim=1)
            topk_idx = torch.topk(-dists, k=min(k, dists.size(0)))[1]
            topk_rgb = local_rgb[topk_idx]  # (k, 3)
            pixel_bank.append(topk_rgb.unsqueeze(0))

    return torch.cat(pixel_bank, dim=0).view(H, W, k, C)


def construct_pixel_bank_real(args = None):
    bank_dir = os.path.join(args.save, args.dataset, '_'.join(str(i) for i in [args.ws, args.ps, args.nn, args.loss]))
    os.makedirs(bank_dir, exist_ok=True)

    noisy_folder = os.path.join(args.data_path, args.dataset, args.Noisy)
    image_files = sorted(os.listdir(noisy_folder))

    avg_elapsed = 0

    # for image_file in image_files:
    for image_file in tqdm(image_files, desc="construct Pixel bank Processing"):
        image_path = os.path.join(noisy_folder, image_file)
        start_time = time.time()

        # 读取图像并转为张量
        img = Image.open(image_path)
        if img.mode == 'L': # 兼容灰度图
            img_tensor = transform(img).unsqueeze(0).cuda()  # 保持灰度
        else:
            img = img.convert('RGB')
            img_tensor = transform(img).unsqueeze(0).cuda()  # RGB

        # 提取 PCA 特征图
        from PixelFeatureMap import extract_feature_map_conv_ae
        img_chw = img_tensor.squeeze(0) if img_tensor.dim() == 4 else img_tensor  # (C,H,W)
        feature_map = extract_feature_map_conv_ae(img_chw, patch_size=args.ps, n_components=args.feature_dim)


        # 原图像素 (H, W, 3)，img_tensor 是 (1, 3, H, W)
        if img_tensor.shape[1] == 1:  # 兼容灰度图
            raw_image = img_tensor.squeeze(0).squeeze(0).unsqueeze(-1).contiguous()  # (H, 1)
        else:
            raw_image = img_tensor.squeeze(0).permute(1, 2, 0).contiguous()  # (H, W, 3)


        # 用局部特征搜索构建 pixel bank
        pixel_bank = get_local_topk_pixel_bank(feature_map, raw_image, k=args.nn, window=args.ws, patch_size=args.ps)

        elapsed = time.time() - start_time
        # print(f"Processed {image_file} in {elapsed:.2f} seconds. Pixel bank shape: {pixel_bank.shape}")

        file_name_without_ext = os.path.splitext(image_file)[0]
        np.save(os.path.join(bank_dir, file_name_without_ext), pixel_bank.cpu().numpy())

        avg_elapsed += elapsed

    avg_elapsed /= len(image_files)
    print(f"Pixel bank construction completed for all images. avg_elapsed: {avg_elapsed:.2f}")


def mse_loss(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return nn.MSELoss()(gt, pred)

def loss_func(img1, img2, loss_f=nn.MSELoss()):
    pred1 = model(img1)
    loss = loss_f(img2, pred1)
    return loss


# -------------------------------
def train(optimizer, img_bank,args):
    N, H, W, C = img_bank.shape

    index1 = torch.randint(0, N, size=(H, W), device=device)
    index1_exp = index1.unsqueeze(0).unsqueeze(-1).expand(1, H, W, C)
    img1 = torch.gather(img_bank, 0, index1_exp)  # Shape: (1, H, W, C)
    img1 = img1.permute(0, 3, 1, 2)  # (1, C, H, W)

    index2 = torch.randint(0, N, size=(H, W), device=device)
    eq_mask = (index2 == index1)
    if eq_mask.any():
        index2[eq_mask] = (index2[eq_mask] + 1) % N
    index2_exp = index2.unsqueeze(0).unsqueeze(-1).expand(1, H, W, C)
    img2 = torch.gather(img_bank, 0, index2_exp)
    img2 = img2.permute(0, 3, 1, 2)

    loss_f = nn.L1Loss() if args.loss == 'L1' else nn.MSELoss()
    loss = loss_func(img1, img2, loss_f)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model, noisy_img, clean_img):
    with torch.no_grad():
        pred = torch.clamp(model(noisy_img), 0, 1)
        mse_val = mse_loss(clean_img, pred).item()
        psnr = 10 * np.log10(1 / mse_val)
    return psnr, pred


# -------------------------------
# Denoising using the Constructed Pixel Bank
# -------------------------------
def denoise_images(args, max_epoch = 3000):
    # The pixel bank directory should match the one used in construction
    if hasattr(args, 'GT'):
        bank_dir = os.path.join(args.save, args.dataset, '_'.join(str(i) for i in [args.ws, args.ps, args.nn, args.loss]))
        gt_folder = os.path.join(args.data_path, args.dataset, args.GT)
        gt_files = sorted(os.listdir(gt_folder))
        is_syn = False
    else:
        bank_dir = os.path.join(args.save, '_'.join(str(i) for i in [args.dataset, args.nt, args.nl, args.ws, args.ps, args.nn, args.loss]))
        gt_folder = os.path.join(args.data_path, args.dataset)
        gt_files = sorted(os.listdir(gt_folder))
        is_syn = True

    os.makedirs(args.out_image, exist_ok=True)

    lr = 0.001
    avg_PSNR = 0
    avg_SSIM = 0
    avg_elapsed = 0

    for image_file in gt_files:
    # for image_file in tqdm(gt_files, desc="denoising"):
        image_start_time = time.time()  # 开始计时

        image_path = os.path.join(gt_folder, image_file)
        clean_img = Image.open(image_path)
        clean_img_tensor = transform(clean_img).unsqueeze(0).to(device)
        clean_img_np = io.imread(image_path)

        bank_path = os.path.join(bank_dir, os.path.splitext(image_file)[0])
        if not os.path.exists(bank_path + '.npy'):
            print(f"Pixel bank for {image_file} not found, skipping denoising.")
            continue

        img_bank_arr = np.load(bank_path + '.npy')
        if img_bank_arr.ndim == 3:
            img_bank_arr = np.expand_dims(img_bank_arr, axis=1)
        # Transpose to (k, H, W, c)
        img_bank = img_bank_arr.astype(np.float32).transpose((2, 0, 1, 3))
        # Use only the first mm banks for training
        img_bank = img_bank[:args.mm]
        img_bank = torch.from_numpy(img_bank).to(device)

        # Use the first bank as the noisy input (reshaped to (1, C, H, W))
        noisy_img = img_bank[0].unsqueeze(0).permute(0, 3, 1, 2)

        n_chan = clean_img_tensor.shape[1]
        global model
        model = DenoiseNet(n_chan ,is_syn = is_syn).to(device)
        # print(f"Number of parameters for {image_file}: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        optimizer = optim.AdamW(model.parameters(), lr=lr)
        scheduler = MultiStepLR(optimizer, milestones=[1500, 2000, 2500], gamma=0.5)

        for epoch in range(max_epoch):
            train(optimizer, img_bank, args)
            scheduler.step()

        PSNR, out_img = test(model, noisy_img, clean_img_tensor)

        image_elapsed = time.time() - image_start_time
        # print(f"Total time for {image_file}: {image_elapsed:.2f} seconds")

        out_img_pil = to_pil_image(out_img.squeeze(0))
        out_img_save_path = os.path.join(args.out_image, os.path.splitext(image_file)[0] + '.png')
        out_img_pil.save(out_img_save_path)

        # noisy_img_pil = to_pil_image(noisy_img.squeeze(0))
        # noisy_img_save_path = os.path.join(args.out_image, os.path.splitext(image_file)[0] + '_noisy.png')
        # noisy_img_pil.save(noisy_img_save_path)

        out_img_loaded = io.imread(out_img_save_path)

        SSIM, _ = ssim(clean_img_np, out_img_loaded, full=True, win_size=5, channel_axis=-1)
        print(f"Image: {image_file} | PSNR: {PSNR:.2f} dB | SSIM: {SSIM:.4f} | elapsed: {image_elapsed:.2f} s")

        avg_PSNR += PSNR
        avg_SSIM += SSIM
        avg_elapsed += image_elapsed

    avg_PSNR /= len(gt_files)
    avg_SSIM /= len(gt_files)
    avg_elapsed /= len(gt_files)
    print(f"Average PSNR: {avg_PSNR:.2f} dB, Average SSIM: {avg_SSIM:.4f}, Average elapsed: {avg_elapsed:.2f} seconds")
    return avg_PSNR, avg_SSIM, avg_elapsed
