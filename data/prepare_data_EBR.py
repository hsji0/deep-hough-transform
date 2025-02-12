import numpy as np
import cv2
from PIL import Image
import argparse
import os, sys
from os.path import join, splitext, abspath, isfile

# Add parent and current directories to Python path to allow importing custom modules.
sys.path.insert(0, abspath(".."))
sys.path.insert(0, abspath("."))

from utils import Line, LineAnnotation, line2hough, line2hough_ebr
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

# Set up command-line arguments.
parser = argparse.ArgumentParser(description="Prepare semantic line data format.")
parser.add_argument('--root', type=str, required=True, help='The data root directory containing images.')
parser.add_argument('--label', type=str, required=True,
                    help='The label root directory containing .txt annotation files.')
parser.add_argument('--num_directions', type=int, default=12,
                    help='The number of angular divisions for line orientation.')
parser.add_argument('--list', type=str, required=True, help='List file (for recording processed names, if needed).')
parser.add_argument('--save-dir', type=str, required=True, help='Directory to save processed outputs.')
parser.add_argument('--prefix', type=str, default="", help="Prefix in list file (if needed).")
parser.add_argument('--fixsize', type=int, default=None,
                    help='Fixed size (square) for resizing images and annotations.')
parser.add_argument('--numangle', type=int, default=80, required=True, help='Number of bins for angle in Hough space.')
parser.add_argument('--numrho', type=int, default=80, required=True, help='Number of bins for rho in Hough space.')
args = parser.parse_args()

# Get absolute paths.
label_path = abspath(args.label)
print(f"label_path {label_path}")
image_dir = abspath(args.root)
save_dir = abspath(args.save_dir)
os.makedirs(save_dir, exist_ok=True)


def nearest8(x):
    """Round x to the nearest multiple of 8."""
    return int(np.round(x / 8) * 8)


def vis_anno(image, annotation):
    """
    Generate a visualization of the annotation overlaid on the image.
    Uses the annotation's oriental_mask() method to create a binary mask.
    """
    mask = annotation.oriental_mask()
    mask_sum = mask.sum(axis=0).astype(bool)
    image_cp = image.copy()
    image_cp[mask_sum, ...] = [0, 255, 0]  # Overlay green on annotated pixels.
    vis_mask = np.zeros((image.shape[0], image.shape[1]))
    vis_mask[mask_sum] = 1
    return image_cp, vis_mask


# List all .txt annotation files in the label directory.
labels_files = [f for f in os.listdir(label_path) if f.endswith(".txt")]
num_samples = len(labels_files)

# Open the list file (if needed for recording processed names).
filelist = open(args.list, "w")
stastic = np.zeros(10)  # Array to keep some statistics (e.g. count of annotations per file).

for idx, label_file in enumerate(labels_files):
    filename, _ = splitext(label_file)
    print("Processing %s [%d/%d]..." % (filename, idx + 1, len(labels_files)))

    image_path = join(image_dir, filename + ".jpg")
    if not isfile(image_path):
        print("Warning: image %s doesn't exist!" % image_path)
        continue

    # Read the image.
    im = cv2.imread(image_path)
    H, W = im.shape[:2]

    # Read annotation file.
    lines = []
    with open(join(label_path, label_file)) as f:
        data = f.readlines()
        nums = len(data)
        if nums < len(stastic):
            stastic[nums] += 1
        for line in data:
            data1 = line.strip().split(',')
            if len(data1) < 4:
                continue
            # Convert the first four entries to integers.
            coords = [int(float(x)) for x in data1[:4]]
            # Skip degenerate line segments (where both endpoints are identical).
            if coords[0] == coords[2] and coords[1] == coords[3]:
                continue
            # Here the annotation format is assumed to be: [x1, y1, x2, y2]
            lines.append(Line(coords))

    # Create a LineAnnotation object.
    annotation = LineAnnotation(size=[H, W], divisions=args.num_directions, lines=lines)

    # Resize image and annotation if fixsize is provided.
    if args.fixsize is not None:
        newH = nearest8(args.fixsize)
        newW = nearest8(args.fixsize)
    else:
        newH = nearest8(H)
        newW = nearest8(W)

    print(f"** newW :{newW}, newH:{newH}")
    im = cv2.resize(im, (newW, newH))
    annotation.resize(size=[newH, newW])

    # Create visualizations.
    vis, mask = vis_anno(im, annotation)

    # Build the Hough space label.
    hough_space_label = np.zeros((args.numangle, args.numrho))
    for l in annotation.lines:
        # theta, r = line2hough(l, numAngle=args.numangle, numRho=args.numrho, size=(newH, newW))
        theta, r = line2hough_ebr(l, numAngle=args.numangle, numRho=args.numrho, size=(newH, newW))
        hough_space_label[theta, r] += 1

    # Apply Gaussian blur and normalize the Hough space label.
    hough_space_label = cv2.GaussianBlur(hough_space_label, (5, 5), 0)
    if hough_space_label.max() > 0:
        hough_space_label = hough_space_label / hough_space_label.max()

    # Collect ground truth coordinates.
    gt_coords = np.array([l.coord for l in annotation.lines])
    data = {
        "hough_space_label8": hough_space_label,
        "coords": gt_coords
    }

    # Prepare save file name.
    save_name = join(save_dir, filename)
    np.save(save_name, data)
    cv2.imwrite(save_name + '.jpg', im)
    cv2.imwrite(save_name + '_p_label.jpg', hough_space_label * 255)
    cv2.imwrite(save_name + '_vis.jpg', vis)
    cv2.imwrite(save_name + '_mask.jpg', mask * 255)
    # Optionally, you can record the processed filename in the list file.
    filelist.write(filename + "\n")

filelist.close()
# print(stastic)
