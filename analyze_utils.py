import os
import cv2
import numpy as np
from PIL import Image


# --- Directories and Reference ---
IMAGE_DIRPATH = r"C:\Users\hsji\Downloads\ebr_test_\image"
INTERMEDIATE_DIRPATH = r"C:\Users\hsji\Downloads\ebr_test_\intermediate"
CONCAT_DIRPATH = r"C:\Users\hsji\Downloads\ebr_test_\concat"

REFERENCE_HOUGH_IMG_FULLPATH = r"C:\Users\hsji\Downloads\ebr_test_\image\6560x50000_5_crop_4_hough.jpg"

def visulize_mapping(b_points, size, filename):
    """
    Reads the image from IMAGE_DIRPATH using the given filename,
    resizes it to the given size, and then draws lines on it based on b_points.
    """

    # Use IMAGE_DIRPATH to locate the image.
    img_path = os.path.join(IMAGE_DIRPATH, rf"{filename}.jpg")
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not load image from {img_path}")
        return None
    img = cv2.resize(img, size)
    # Draw each line from b_points.
    for (y1, x1, y2, x2) in b_points:
        thickness = int(0.01 * max(size[0], size[1]))
        img = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=thickness)
    return img

def get_size():
    ref_img = Image.open(REFERENCE_HOUGH_IMG_FULLPATH)
    ref_img =  np.array(ref_img)
    print(ref_img.shape)
    return ref_img.shape[:2]

def get_boundary_point(y, x, angle, H, W):
    '''
    Given point y,x with angle, return a two point in image boundary with shape [H, W]
    return point:[x, y]
    '''
    point1 = None
    point2 = None

    if angle == -np.pi / 2:
        point1 = (x, 0)
        point2 = (x, H - 1)
    elif angle == 0.0:
        point1 = (0, y)
        point2 = (W - 1, y)
    else:
        k = np.tan(angle)
        if y - k * x >= 0 and y - k * x < H:  # left
            if point1 == None:
                point1 = (0, int(y - k * x))
            elif point2 == None:
                point2 = (0, int(y - k * x))
                if point2 == point1: point2 = None
        # print(point1, point2)
        if k * (W - 1) + y - k * x >= 0 and k * (W - 1) + y - k * x < H:  # right
            if point1 == None:
                point1 = (W - 1, int(k * (W - 1) + y - k * x))
            elif point2 == None:
                point2 = (W - 1, int(k * (W - 1) + y - k * x))
                if point2 == point1: point2 = None
        # print(point1, point2)
        if x - y / k >= 0 and x - y / k < W:  # top
            if point1 == None:
                point1 = (int(x - y / k), 0)
            elif point2 == None:
                point2 = (int(x - y / k), 0)
                if point2 == point1: point2 = None
        # print(point1, point2)
        if x - y / k + (H - 1) / k >= 0 and x - y / k + (H - 1) / k < W:  # bottom
            if point1 == None:
                point1 = (int(x - y / k + (H - 1) / k), H - 1)
            elif point2 == None:
                point2 = (int(x - y / k + (H - 1) / k), H - 1)
                if point2 == point1: point2 = None
        # print(int(x-y/k+(H-1)/k), H-1)
        if point2 == None: point2 = point1
    return point1, point2

# --- Dummy Reverse Mapping Function ---
def reverse_mapping(point_list, numAngle, numRho, size=(32, 32)):
    H, W = size
    irho = int(np.sqrt(H*H + W*W) + 1) / ((numRho - 1))
    itheta = np.pi / numAngle
    b_points = []

    for (ri, thetai) in point_list:
        theta = thetai * itheta
        r = ri - numRho // 2
        cosi = np.cos(theta) / irho
        sini = np.sin(theta) / irho
        if sini == 0:
            x = np.round(r / cosi + W / 2)
            b_points.append((0, int(x), H-1, int(x)))
        else:
            angle = np.arctan(- cosi / sini)
            y = np.round(r / sini + W * cosi / sini / 2 + H / 2)
            p1, p2 = get_boundary_point(int(y), 0, angle, H, W)
            if p1 is not None and p2 is not None:
                b_points.append((p1[1], p1[0], p2[1], p2[0]))
    return b_points
