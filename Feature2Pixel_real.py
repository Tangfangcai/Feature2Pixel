import argparse
import datetime
import os
import sys

from utils import Logger, construct_pixel_bank_real, denoise_images

# -------------------------------
parser = argparse.ArgumentParser('Feature2Pixel')
parser.add_argument('--data_path', default='./data', type=str, help='Path to the data')
parser.add_argument('--dataset', default='PolyU', type=str, help='Dataset name')
parser.add_argument('--GT', default='GT', type=str, help='Folder name for ground truth images')
parser.add_argument('--Noisy', default='Noisy', type=str, help='Folder name for noisy images')
parser.add_argument('--save', default='./results_bank', type=str, help='Directory to save pixel bank results')
parser.add_argument('--out_image', default='./results_image', type=str, help='Directory to save denoised images')
parser.add_argument('--ws', default=48, type=int, help='Window size')
parser.add_argument('--ps', default=5, type=int, help='Patch size')
parser.add_argument('--nn', default=100, type=int, help='Number of nearest neighbors to search')
parser.add_argument('--mm', default=16, type=int, help='Number of pixel banks to use for training')
parser.add_argument('--loss', default='L2', type=str, help='Loss function type')
parser.add_argument('--feature_dim', default=7, type=int, help='Number of components for feature extraction')
args = parser.parse_args()


if __name__ == "__main__":
    if "sidd" in args.dataset.lower():
        args.feature_dim = 1
    elif "polyu" in args.dataset.lower():
        args.feature_dim = 8
    else:
        args.feature_dim = 7

    args.out_image = f"./results_image/{args.dataset}"
    os.makedirs(f"{args.out_image}", exist_ok=True)
    os.makedirs("log", exist_ok=True)
    sys.stdout = Logger(f"log/log_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt")

    print("Start Time:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(args)
    print("Constructing pixel banks ...")
    construct_pixel_bank_real(args=args)
    print("Starting denoising ...")
    denoise_images(args)



