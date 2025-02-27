
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import cv2

global H_RATIO
global W_RATIO
# --- Directories and Reference ---
IMAGE_DIRPATH = r"C:\Users\hsji\Downloads\ebr_test_\image"
INTERMEDIATE_DIRPATH = r"C:\Users\hsji\Downloads\ebr_test_\intermediate"
CONCAT_DIRPATH = r"C:\Users\hsji\Downloads\ebr_test_\concat"

REFERENCE_IMG_FULLPATH = r"C:\Users\hsji\Downloads\ebr_test_\image\6560x50000_5_crop_4.jpg"
REFERENCE_HOUGH_IMG_FULLPATH = r"C:\Users\hsji\Downloads\ebr_test_\image\6560x50000_5_crop_4_hough.jpg"
REFERENCE_FILENAME = os.path.splitext(os.path.basename(REFERENCE_IMG_FULLPATH))[0]

OUTPUT_DIRPATH = r"C:\Users\hsji\Downloads\ebr_test_\Output"
os.makedirs(OUTPUT_DIRPATH, exist_ok=True)


def find_hough_lines(file_fullpath, method="max"):
    """
    Given an intermediate file path, derive the corresponding image path.
    Here we assume that the image file is named as:
       <base>_hough.jpg
    where <base> is obtained from the intermediate file's base name.
    """
    print("[find_hough_lines]")
    # Get the base name (without extension) from the intermediate file.
    filename = os.path.splitext(os.path.basename(file_fullpath))[0]
    print("Intermediate base filename:", filename)
    # Construct the image file name by appending "_hough.jpg"
    img_fullpath = os.path.join(IMAGE_DIRPATH, f"{filename}_hough.jpg")
    print("Image path:", img_fullpath)

    # Open image using PIL and convert to numpy array.
    img_ = Image.open(img_fullpath)
    img_ = np.array(img_)

    # Get dimensions.
    if len(img_.shape) == 2:
        img_height, img_width = img_.shape
    else:
        img_height, img_width = img_.shape[:2]

    flattened_sorted_ = img_.flatten()

    if method == "max":  # find hough line in hough space (represent as point (r, theta))
        sorted_indices = np.argsort(flattened_sorted_)
        top3_indices = sorted_indices[-3:]
        result = []
        for top_index in top3_indices[::-1]:
            # Extract (x, y) coordinates of line in hough space.
            x_ = top_index % img_width
            y_ = top_index // img_width
            result.append((x_, y_))
        return result



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
    print(f"- point_list :{point_list}")
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

def get_feat_ratio(layer):
    # Load the original reference image.
    ref_img = Image.open(REFERENCE_HOUGH_IMG_FULLPATH)
    ref_img = np.array(ref_img)
    size = ref_img.shape[-2:]

    intermediate_filename = os.path.splitext(os.path.basename(REFERENCE_HOUGH_IMG_FULLPATH))[0]
    intermediate_npz = os.path.join(INTERMEDIATE_DIRPATH, rf"{intermediate_filename[:-6]}.npz")
    data = np.load(intermediate_npz, allow_pickle=True)
    tensor = data[layer]  # e.g., shape: (1, channels, height, width)

    h_ratio =  size[0] / tensor.shape[-2]
    w_ratio = size[1] / tensor.shape[-1]
    return (w_ratio, h_ratio)

# --- Modified Feature Extraction ---
def  extract_intermediate_feature(intermediate_fullpath, feat_xy_list, layer, size):
    """
    Loads the intermediate feature file (assumed saved as an npz file) from the given path.
    It then extracts features from the specified layer (e.g. "p1").
    Returns a list of dictionaries, each with:
         {"coords": (x, y), "feature": feature_vector}
    """
    # Use the given path directly.
    filename = os.path.splitext(os.path.basename(intermediate_fullpath))[0]
    npz_filename = f"{filename}.npz"
    full_path = os.path.join(INTERMEDIATE_DIRPATH, npz_filename)
    data = np.load(full_path, allow_pickle=True)
    tensor = data[layer]  # e.g., shape: (1, channels, height, width)

    results = []
    for (x_, y_) in feat_xy_list:
        try:
            x_ = int(np.round(x_ / W_RATIO))
            feat = tensor[0, :, y_, x_]
            results.append({"coords": (x_, y_), "feature": feat})
        except IndexError:
            results.append({"coords": (x_, y_), "feature": None})
    print(f"Extracted {len(results)} feature entries from {os.path.basename(full_path)}")
    return results
def extract_concat_feature(concat_fullpath, feat_xy_list, layer, size):
    """
    Loads the intermediate feature file (assumed saved as an npz file) from the given path.
    It then extracts features from the specified layer (e.g. "p1").
    Returns a list of dictionaries, each with:
         {"coords": (x, y), "feature": feature_vector}
    """
    # Use the given path directly.
    filename = os.path.splitext(os.path.basename(concat_fullpath))[0]
    npy_filename = f"{filename}.npy"
    full_path = os.path.join(CONCAT_DIRPATH, npy_filename)
    data = np.load(full_path, allow_pickle=True)
    print(data.shape)
    results = []
    for (x_, y_) in feat_xy_list:
        try:
            feat = data[0, :, y_, x_]
            results.append({"coords": (x_, y_), "feature": feat})
        except IndexError:
            results.append({"coords": (x_, y_), "feature": None})
    print(f"Extracted {len(results)} feature entries from {os.path.basename(full_path)}")
    return results
def calc_cosine_similarity(feature1, feature2):
    """Calculate cosine similarity between two 1D feature vectors."""
    dot_product = np.dot(feature1, feature2)
    norm1 = np.linalg.norm(feature1)
    norm2 = np.linalg.norm(feature2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def compute_match_dict_reference(reference_file, target_files, method_="max", layer="p1", size=(320,320)):
    """
    Processes the reference image and each target image.
    For the reference image, it extracts features and uses the first feature as the representative.
    Then, for each target image, for each extracted feature (up to 3),
    it computes cosine similarity with the reference representative.
    Returns a dictionary in the form:
       { target_filename: [ {"coords": (x,y), "similarity": sim_value}, ... ] }
    """
    # Process reference image.
    ref_feat_xy_list = find_hough_lines(reference_file, method=method_)
    ref_feat_xy_list.pop(-1)
    ref_feat_xy_list.pop(-1)
    # ref_entries = extract_intermediate_feature(reference_file, ref_feat_xy_list, layer, size)
    ref_entries = extract_concat_feature(reference_file, ref_feat_xy_list, layer, size)
    rep_ref = ref_entries[0]["feature"] if ref_entries and ref_entries[0]["feature"] is not None else None
    ref_key = os.path.splitext(os.path.basename(reference_file))[0]

    match_dict = {}  # Will store matches for each target.

    # Process each target image.
    for file_path in target_files:
        feat_xy_list = find_hough_lines(file_path, method=method_)
        # target_entries = extract_intermediate_feature(file_path, feat_xy_list, layer, size)
        target_entries = extract_concat_feature(file_path, feat_xy_list, layer, size)
        key = os.path.splitext(os.path.basename(file_path))[0]
        matches = []
        if rep_ref is None:
            # If no representative feature from reference, store empty matches.
            for entry in target_entries:
                matches.append({"coords": entry["coords"], "similarity": None})
        else:
            # For each entry in target (up to 3)
            for entry in target_entries[:3]:
                feat = entry["feature"]
                sim_val = calc_cosine_similarity(rep_ref, feat) if feat is not None else None
                matches.append({"coords": entry["coords"], "similarity": sim_val})
        match_dict[key] = matches
    return {ref_key: match_dict}  # Outer dict keyed by reference filename.


# --- Visualization of Matched Lines ---
def visualize_matches(reference_file, match_dict, numAngle=320, numRho=320, orig_size=(640, 640), layer="p1"):
    """
    For each target image in match_dict, pick the best match (the one with highest cosine similarity)
    from the reference image. Use reverse_mapping to convert its hough-space coordinate to line endpoints,
    then overlay the line on a copy of the reference image and save the result.
    """
    # Load the original reference image.
    ref_img = Image.open(reference_file)
    ref_img = np.array(ref_img)
    if len(ref_img.shape) == 2:
        ref_img_vis = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
    else:
        ref_img_vis = ref_img.copy()

    # Get the reference key and the target match dictionary.
    ref_key = list(match_dict.keys())[0]
    target_matches = match_dict[ref_key]

    # Iterate through each target.
    for target, matches in target_matches.items():
        # Choose the match with the highest cosine similarity.
        best_match = None
        best_sim = -1.0
        for m in matches:
            if m["similarity"] is not None and m["similarity"] > best_sim:
                best_sim = m["similarity"]
                best_match = m
        # If no valid match found, skip this target.
        if best_match is None:
            continue

        # Reverse-map the hough coordinate for this best match.
        # reverse_mapping expects a list, so we pass [best_match["coords"]].
        b_points = reverse_mapping([best_match["coords"]], numAngle, numRho, size=orig_size)
        if not b_points or b_points[0] is None:
            continue

        # Optionally, apply scaling if needed.
        # (Here we assume a scale from orig_size (640x640) to the desired output size, e.g., 2400x640.)
        scale_w = 2400 / 640
        scale_h = 640 / 640
        for i in range(len(b_points)):
            y1 = int(np.round(b_points[i][0] * scale_h))
            x1 = int(np.round(b_points[i][1] * scale_w))
            y2 = int(np.round(b_points[i][2] * scale_h))
            x2 = int(np.round(b_points[i][3] * scale_w))
            if x1 == x2:
                angle = -np.pi / 2
            else:
                angle = np.arctan((y1 - y2) / (x1 - x2))
            (x1, y1), (x2, y2) = get_boundary_point(y1, x1, angle, 640, 2400)
            b_points[i] = (y1, x1, y2, x2)
        img_vis = visulize_mapping(b_points, (2400, 640), target)
        cv2.imwrite(os.path.join(OUTPUT_DIRPATH, rf"match_{layer}_ref_{target}.jpg"), img_vis)

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


# --- Main processing ---
if __name__ == "__main__":
    # Gather target intermediate files ending with '.npz'
    # target_files = [os.path.join(INTERMEDIATE_DIRPATH, filename)
    #                 for filename in os.listdir(INTERMEDIATE_DIRPATH)
    #                 if filename.endswith("npz")]
    target_files = [os.path.join(CONCAT_DIRPATH, filename)
                    for filename in os.listdir(CONCAT_DIRPATH)
                    if filename.endswith("npy")]
    reference_file = target_files.pop(0)
    print(f"=> reference :{reference_file}, target_files :{target_files}")

    size = get_size()
    layer = "concat"
    print(f"size :{size}, layer :{layer}")
    # (W_RATIO, H_RATIO) = get_feat_ratio(layer)
    # print(f"get_feat_ratio :{W_RATIO, H_RATIO}")
    # Compute the match dictionary: reference vs. targets.
    match_dict = compute_match_dict_reference(REFERENCE_IMG_FULLPATH, target_files, method_="max", layer=layer)
    print("Match Dictionary:")
    print(match_dict)

    # Optionally, save the match dictionary to an Excel file.
    # (For simplicity, we convert the dictionary to a DataFrame.)
    # Here we create a DataFrame with columns: Target, Coords, Similarity.
    ref_key = list(match_dict.keys())[0]
    rows = []
    for target, matches in match_dict[ref_key].items():
        for m in matches:
            rows.append({"Target": target, "Coords": m["coords"], "Similarity": m["similarity"]})
    df = pd.DataFrame(rows)
    excel_path = os.path.join(INTERMEDIATE_DIRPATH, "match_dictionary.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"Saved match dictionary to Excel: {excel_path}")

    # Visualize the matched lines on the reference image.
    # Adjust numAngle, numRho, and orig_size as needed.
    visualize_matches(REFERENCE_IMG_FULLPATH, match_dict, numAngle=320, numRho=320, orig_size=(640, 640), layer=layer)
