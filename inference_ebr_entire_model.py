import argparse
import os
import random
import time
from os.path import isfile, join, split

import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import numpy as np
import tqdm
import yaml
import cv2

from torch.optim import lr_scheduler
from logger import Logger

from dataloader import get_loader
from model.network import Net
from skimage.measure import label, regionprops
from utils import reverse_mapping, visulize_mapping, edge_align, get_boundary_point

parser = argparse.ArgumentParser(description='PyTorch Semantic-Line Testing')
# arguments from command line
parser.add_argument('--config', default="./config.yml", help="path to config file")
parser.add_argument('--model', required=True, help='path to the saved model')
parser.add_argument('--align', default=False, action='store_true')
parser.add_argument('--tmp', default="", help='tmp')
args = parser.parse_args()

assert os.path.isfile(args.config)
CONFIGS = yaml.safe_load(open(args.config))

# merge configs
if args.tmp != "" and args.tmp != CONFIGS["MISC"]["TMP"]:
    CONFIGS["MISC"]["TMP"] = args.tmp

os.makedirs(CONFIGS["MISC"]["TMP"], exist_ok=True)
logger = Logger(os.path.join(CONFIGS["MISC"]["TMP"], "log.txt"))


def main():
    logger.info(args)

    # Load the saved model directly.
    if isfile(args.model):
        logger.info("=> loading saved model '{}'".format(args.model))
        # Set device according to availability and config.
        device = torch.device("cuda:{}".format(CONFIGS["TRAIN"]["GPU_ID"]) if torch.cuda.is_available() else "cpu")
        model = torch.load(args.model, map_location=device, weights_only=False)
        model = model.to(device)
        logger.info("=> loaded saved model '{}'".format(args.model))
    else:
        logger.info("=> no saved model found at '{}'".format(args.model))
        exit(1)

    # Dataloader for testing.
    test_loader = get_loader(CONFIGS["DATA"]["TEST_DIR"], CONFIGS["DATA"]["TEST_LABEL_FILE"],
                             batch_size=1, num_thread=CONFIGS["DATA"]["WORKERS"], test=True)
    logger.info("Data loading done.")
    logger.info("Start testing.")
    total_time = test(test_loader, model, args)

    logger.info("Test done! Total %d imgs at %.4f secs without image io, fps: %.3f" %
                (len(test_loader.dataset), total_time, len(test_loader.dataset) / total_time))


def test(test_loader, model, args):
    # Switch to evaluation mode.
    model.eval()
    with torch.no_grad():
        bar = tqdm.tqdm(test_loader)
        ftime = 0
        ntime = 0
        for i, data in enumerate(bar):
            t = time.time()
            images, names, size = data
            print(f"Model input size: {size}")
            images = images.cuda(device=CONFIGS["TRAIN"]["GPU_ID"])
            key_points = model(images)
            key_points = torch.sigmoid(key_points)
            ftime += (time.time() - t)
            t = time.time()

            visualize_save_path = os.path.join(CONFIGS["MISC"]["TMP"], 'visualize_test')
            os.makedirs(visualize_save_path, exist_ok=True)

            # --- Save the raw Hough space result ---
            hough_space = key_points.squeeze().cpu().numpy()
            print(hough_space.shape)
            hough_space_norm = np.uint8(255 * (hough_space - hough_space.min()) /
                                        (hough_space.max() - hough_space.min() + 1e-6))
            hough_save_filename = names[0].split('/')[-1].split('.')[0] + '_hough.jpg'
            cv2.imwrite(join(visualize_save_path, hough_save_filename), hough_space_norm)
            # --- End Hough space saving ---

            binary_kmap = key_points.squeeze().cpu().numpy() > CONFIGS['MODEL']['THRESHOLD']
            kmap_label = label(binary_kmap, connectivity=1)
            props = regionprops(kmap_label)
            plist = [prop.centroid for prop in props]

            size_tuple = (size[0][0], size[0][1])
            print(f"[Imp] size: {size_tuple}")
            b_points = reverse_mapping_ebr(plist, numAngle=CONFIGS["MODEL"]["NUMANGLE"],
                                       numRho=CONFIGS["MODEL"]["NUMRHO"], size=(640, 640))
            scale_w = size_tuple[1] / 640
            scale_h = size_tuple[0] / 640
            print(f"[Imp] scale_w: {scale_w} , scale_h: {scale_h}")

            for j in range(len(b_points)):
                y1 = int(np.round(b_points[j][0] * scale_h))
                x1 = int(np.round(b_points[j][1] * scale_w))
                y2 = int(np.round(b_points[j][2] * scale_h))
                x2 = int(np.round(b_points[j][3] * scale_w))
                if x1 == x2:
                    angle = -np.pi / 2
                else:
                    angle = np.arctan((y1 - y2) / (x1 - x2))
                (x1, y1), (x2, y2) = get_boundary_point(y1, x1, angle, size_tuple[0], size_tuple[1])
                b_points[j] = (y1, x1, y2, x2)

            vis = visulize_mapping(b_points, size_tuple[::-1], names[0])
            print(f"> visualize_save_path: {visualize_save_path}")
            cv2.imwrite(join(visualize_save_path, names[0].split('/')[-1]), vis)

            if CONFIGS["MODEL"]["EDGE_ALIGN"] and args.align:
                for j in range(len(b_points)):
                    b_points[j] = edge_align(b_points[j], names[0], size_tuple, division=5)
                vis = visulize_mapping(b_points, size_tuple, names[0])
                cv2.imwrite(join(visualize_save_path, names[0].split('/')[-1].split('.')[0] + '_align.png'), vis)
            ntime += (time.time() - t)
    print('Forward time for total images: %.6f' % ftime)
    print('Post-processing time for total images: %.6f' % ntime)
    return ftime + ntime


if __name__ == '__main__':
    main()
