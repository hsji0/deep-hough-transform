import torch
import numpy as np
import math
import cv2 
import os
#import torchvision
from PIL import Image
from basic_ops import *

def draw_line(y, x, angle, image, color=(0,0,255), num_directions=24):
    '''
    Draw a line with point y, x, angle in image with color.
    '''
    cv2.circle(image, (x, y), 2, color, 2)
    H, W = image.shape[:2]
    angle = int2arc(angle, num_directions)
    point1, point2 = get_boundary_point(y, x, angle, H, W)
    cv2.line(image, point1, point2, color, 2)
    return image

def convert_line_to_hough(line, size=(32, 32)):
    H, W = size
    theta = line.angle()
    alpha = theta + np.pi / 2
    if theta == -np.pi / 2:
        r = line.coord[1] - W/2
    else:
        k = np.tan(theta)
        y1 = line.coord[0] - H/2
        x1 = line.coord[1] - W/2
        r = (y1 - k*x1) / np.sqrt(1 + k**2)
    return alpha, r

def line2hough(line, numAngle, numRho, size=(320, 320)):
    H, W = size
    print(f"line info : {line.coord} {line.length} {line.angle}")
    alpha, r = convert_line_to_hough(line, size)

    irho = int(np.sqrt(H*H + W*W) + 1) / ((numRho - 1))
    itheta = np.pi / numAngle

    r = int(np.round(r / irho)) + int((numRho) / 2)
    alpha = int(np.round(alpha / itheta))
    if alpha >= numAngle:
        alpha = numAngle - 1
    return alpha, r

def convert_line_to_hough_ebr(line, size=(32, 32)):
    """
    Converts a line to its Hough representation (theta, r) using the coordinate system
    where the origin is at the bottom-center of the image: (W/2, H).

    The new coordinates are defined as:
        x' = x - (W/2)
        y' = H - y
    so that the origin (0,0) is at bottom-center (center horizontally, bottom vertically).

    The line's endpoints are assumed to be stored in line.coord as [x1, y1, x2, y2].
    We compute the line angle (in the new coordinate system) and then define the normal
    angle theta = (line_angle + pi/2) mod pi. The distance r is computed from the midpoint.

    This function ensures r >= 0 and theta in [0, pi).
    """
    H, W = size
    # Unpack endpoints (assumes line.coord = [x1, y1, x2, y2])
    y1, x1, y2, x2 = [float(v) for v in line.coord]

    # Transform coordinates to new system with origin at (W/2, H)
    x1p = x1 - (W / 2)
    y1p = H - y1
    x2p = x2 - (W / 2)
    y2p = H - y2

    # Compute the angle of the line (in the new coordinate system) using arctan2
    angle_line = np.arctan2(y2p - y1p, x2p - x1p)
    # The normal's angle for the Hough transform:
    theta = (angle_line + np.pi / 2) % np.pi  # ensures theta in [0, pi)

    # Compute the midpoint of the line segment in the new coordinate system.
    xm = (x1p + x2p) / 2.0
    ym = (y1p + y2p) / 2.0

    # Compute the distance from the origin to the line using the Hough formula.
    r = xm * np.cos(theta) + ym * np.sin(theta)

    # Enforce r >= 0: if negative, flip both r and theta.
    if r < 0:
        r = -r
        theta = (theta + np.pi) % np.pi

    return theta, r


def line2hough_ebr(line, numAngle, numRho, size=(32, 32)):
    """
    Quantizes the continuous Hough parameters (theta, r) for a given line into discrete bins.

    The theta is assumed to be in [0, pi) and r in [0, max_r], where max_r is the maximum
    distance in the new coordinate system. In the new system:
      - x' ranges in [-W/2, W/2]
      - y' ranges in [0, H]
    so the maximum distance from the origin iyjr eotf;fd
    dns:
      max_r = sqrt((W/2)^2 + (H)^2)show_interest(value

    Returns:
        theta_bin, r_bin : integer bin indices for theta and r.
    """
    H, W = size
    # Maximum possible r in our new coordinate system (bottom-center as origin)
    max_r = np.sqrt((W / 2) ** 2 + H ** 2)

    # Convert line to continuous Hough parameters in the new coordinate system.
    theta, r = convert_line_to_hough_ebr(line, size)

    # Quantize theta: divide [0, pi) into numAngle bins.
    itheta = np.pi / numAngle
    theta_bin = int(np.round(theta / itheta))
    if theta_bin >= numAngle:
        theta_bin = numAngle - 1

    # Quantize r: divide [0, max_r] into (numRho - 1) intervals.
    irho = max_r / (numRho - 1)
    r_bin = int(np.round(r / irho))
    if r_bin >= numRho:
        r_bin = numRho - 1

    return theta_bin, r_bin



def line2hough_float(line, numAngle, numRho, size=(32, 32)):
    H, W = size
    alpha, r = convert_line_to_hough(line, size)

    irho = int(np.sqrt(H*H + W*W) + 1) / ((numRho - 1))
    itheta = np.pi / numAngle

    r = r / irho + numRho / 2
    alpha = alpha / itheta
    if alpha >= numAngle:
        alpha = numAngle - 1
    return alpha, r

def reverse_mapping(point_list, numAngle, numRho, size=(32, 32)):
    H, W = size
    irho = int(np.sqrt(H*H + W*W) + 1) / ((numRho - 1))
    itheta = np.pi / numAngle
    b_points = []

    for (thetai, ri) in point_list:
        theta = thetai * itheta
        r = ri - numRho // 2
        cosi = np.cos(theta) / irho
        sini = np.sin(theta) / irho
        if sini == 0:
            x = np.round(r / cosi + W / 2)
            b_points.append((0, int(x), H-1, int(x)))
        else:
            # print('k = %.4f', - cosi / sini)
            # print('b = %.2f', np.round(r / sini + W * cosi / sini / 2 + H / 2))
            angle = np.arctan(- cosi / sini)
            y = np.round(r / sini + W * cosi / sini / 2 + H / 2)
            p1, p2 = get_boundary_point(int(y), 0, angle, H, W)
            if p1 is not None and p2 is not None:
                b_points.append((p1[1], p1[0], p2[1], p2[0]))
    return b_points

# def reverse_mapping_ebr(hough_points, numAngle, numRho, size):
#     H, W = size
#     max_r = np.sqrt((W / 2) ** 2 + H ** 2)
#     itheta = np.pi / numAngle
#     irho = max_r / (numRho - 1)
#
#     lines = []
#     for hp in hough_points:
#         # Use hp[0] for theta and hp[1] for r, as regionprops returns (row, col).
#         theta = hp[0] * itheta
#         r = hp[1] * irho
#
#         pts = []
#         # Intersection with left boundary (x' = -W/2)
#         x_val = -W / 2
#         if np.abs(np.cos(theta)) > 1e-6:
#             y_val = (r - x_val * np.cos(theta)) / np.sin(theta) if np.abs(np.sin(theta)) > 1e-6 else None
#             if y_val is not None and 0 <= y_val <= H:
#                 pts.append((x_val, y_val))
#         # Intersection with right boundary (x' = W/2)
#         x_val = W / 2
#         if np.abs(np.cos(theta)) > 1e-6:
#             y_val = (r - x_val * np.cos(theta)) / np.sin(theta) if np.abs(np.sin(theta)) > 1e-6 else None
#             if y_val is not None and 0 <= y_val <= H:
#                 pts.append((x_val, y_val))
#         # Intersection with bottom boundary (y' = 0)
#         y_val = 0
#         if np.abs(np.sin(theta)) > 1e-6:
#             x_val = (r - y_val * np.sin(theta)) / np.cos(theta) if np.abs(np.cos(theta)) > 1e-6 else None
#             if x_val is not None and -W / 2 <= x_val <= W / 2:
#                 pts.append((x_val, y_val))
#         # Intersection with top boundary (y' = H)
#         y_val = H
#         if np.abs(np.sin(theta)) > 1e-6:
#             x_val = (r - y_val * np.sin(theta)) / np.cos(theta) if np.abs(np.cos(theta)) > 1e-6 else None
#             if x_val is not None and -W / 2 <= x_val <= W / 2:
#                 pts.append((x_val, y_val))
#
#         pts = list(set(pts))
#         if len(pts) >= 2:
#             pt1, pt2 = pts[0], pts[1]
#             pt1_orig = (pt1[0] + W / 2, H - pt1[1])
#             pt2_orig = (pt2[0] + W / 2, H - pt2[1])
#             lines.append((pt1_orig[1], pt1_orig[0], pt2_orig[1], pt2_orig[0]))
#     return lines


def visulize_mapping(b_points, size, filename):
    if len(b_points) <= 0:
        return None
    # print(f"b_points :{b_points}")
    # print(f"size :{size}")
    # print(f"filename :{filename}")
    img = cv2.imread(os.path.join('./data/training/ebr_test', filename)) #change the path when using other dataset.
    img = cv2.resize(img, size)
    for (y1, x1, y2, x2) in b_points:
        img = cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), thickness=int(0.01*max(size[0], size[1])))
    return img

def caculate_precision(b_points, gt_coords, thresh=0.90):
    N = len(b_points)
    if N == 0:
        return 0, 0
    ea = np.zeros(N, dtype=np.float32)
    for i, coord_p in enumerate(b_points):
        if coord_p[0]==coord_p[2] and coord_p[1]==coord_p[3]:
            continue
        l_pred = Line(list(coord_p))
        for coord_g in gt_coords:
            l_gt = Line(list(coord_g))
            ea[i] = max(ea[i], EA_metric(l_pred, l_gt))
    return (ea >= thresh).sum(), N

def caculate_recall(b_points, gt_coords, thresh=0.90):
    N = len(gt_coords)
    if N == 0:
        return 1.0, 0
    ea = np.zeros(N, dtype=np.float32)
    for i, coord_g in enumerate(gt_coords):
        l_gt = Line(list(coord_g))
        for coord_p in b_points:
            if coord_p[0]==coord_p[2] and coord_p[1]==coord_p[3]:
                continue
            l_pred = Line(list(coord_p))
            ea[i] = max(ea[i], EA_metric(l_pred, l_gt))
    return (ea >= thresh).sum(), N

def coords_sort(coords):
    y1, x1, y2, x2 = coords
    if x1 > x2 or (x1 == x2 and y1 > y2):
        yy1, xx1, yy2, xx2 = y2, x2, y1, x1
    else:
        yy1, xx1, yy2, xx2 = y1, x1, y2, x2
    return yy1, xx1, yy2, xx2

def get_density(filename, x1, y1, x2, y2):
    hed_path = '/home/hanqi/JTLEE_code/pytorch-hed/hed_results/'
    filename = filename.split('_')[0]
    hed_file_path = os.path.join(hed_path, filename + '.png')
    hed = np.array(Image.open(hed_file_path).convert('L')) / 255

    mask = np.zeros_like(hed)
    mask = cv2.line(mask, (x1, y1), (x2, y2), color=1.0, thickness=7)

    density = (mask * hed).sum() / mask.sum()
    return density

def local_search(coords, coords_ring, d=1):

    y1, x1 = coords
    
    length = len(coords_ring)
    idx = coords_ring.index((x1, y1))
    new_x1, new_y1 = coords_ring[(idx + d) % length]

    return new_y1, new_x1 

def overflow(x, size=400):
    return x < 0 or x >= size

def edge_align(coords, filename, size, division=9):
    y1, x1, y2, x2 = coords
    ry1, rx1, ry2, rx2 = y1, x1, y2, x2
    if overflow(y1, size[0]) or overflow(x1, size[1]) or overflow(y2, size[0]) or overflow(x2, size[1]):
        return [ry1, rx1, ry2, rx2]
    density = 0
    hed_path = './data/sl6500_hed_results/'
    # hed_path = '/home/hanqi/JTLEE_code/pytorch-hed/hed_results/'
    filename = filename.split('.')[0]
    hed_file_path = os.path.join(hed_path, filename + '.png')
    hed = np.array(Image.open(hed_file_path).convert('L')) / 255
    
    coords_ring = [] #(x, y)
    #size = (400, 400)
    for i in range(0, size[1]):
        coords_ring.append((i, 0))
    for i in range(1, size[0]):
        coords_ring.append((size[1]-1, i))
    for i in range(size[1]-2, 0, -1):
        coords_ring.append((i, size[0]-1))
    for i in range(size[0]-1, 0, -1):
        coords_ring.append((0, i))


    for d1 in range(-division, division+1):
        for d2 in range(-division, division+1):
            ny1, nx1 = local_search([y1, x1], coords_ring, d=d1)
            ny2, nx2 = local_search([y2, x2], coords_ring, d=d2)

            mask = np.zeros_like(hed)
            mask = cv2.line(mask, (nx1, ny1), (nx2, ny2), color=1.0, thickness=3)
            dens = (mask * hed).sum() / mask.sum()
            if dens > density:
                density = dens
                ry1, rx1, ry2, rx2 = ny1, nx1, ny2, nx2

    return [ry1, rx1, ry2, rx2]
