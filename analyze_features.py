
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from analyze_utils import *
from analyze_utils import *

"""
Deep hough transform 조희연님 제안 방식 테스트
 - descriptor vector 를 cosine similarity 계산하여 유사도 측정
 - network 에서 concat 결과 기반 (return concat : True) 
 - reference line 의 feature vector 와 가장 유사도 큰 하나만 추출한 결과를 시각화하여 저장

"""
# --- Directories and Reference ---
IMAGE_DIRPATH = r"C:\Users\hsji\Downloads\ebr_test_\image"
INTERMEDIATE_DIRPATH = r"C:\Users\hsji\Downloads\ebr_test_\intermediate"
CONCAT_DIRPATH = r"C:\Users\hsji\Downloads\ebr_test_\concat"

REFERENCE_IMG_FULLPATH = r"C:\Users\hsji\Downloads\ebr_test_\image\6560x50000_5_crop_4.jpg"
REFERENCE_HOUGH_IMG_FULLPATH = r"C:\Users\hsji\Downloads\ebr_test_\image\6560x50000_5_crop_4_hough.jpg"
REFERENCE_FILENAME = os.path.splitext(os.path.basename(REFERENCE_IMG_FULLPATH))[0]

OUTPUT_DIRPATH = r"C:\Users\hsji\Downloads\ebr_test_\Output"
os.makedirs(OUTPUT_DIRPATH, exist_ok=True)

NUM_RHO = 320
NUM_ANGLE = 320
IMAGE_WIDTH = 2400
IMAGE_HEIGHT = 640
MODEL_WIDTH = 640
MODEL_HEIGHT = 640

def find_hough_lines(file_fullpath, method="max"):
    """
    hough jpg 이미지에서 max 값 (가장 밝은) 위치의 x,y 좌표 추출
    """
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
def find_hough_lines_reference(file_fullpath, method="max"):
    """
    hough jpg 이미지에서 max 값 (가장 밝은) 위치의 x,y 좌표 추출
    """
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
        top_index = sorted_indices[-1]
        result = []
        x_ = top_index % img_width
        y_ = top_index // img_width
        result.append((x_, y_))
        return result

def extract_intermediate_feature(intermediate_fullpath, feat_xy_list, layer, size):
    """
    When using intermediate npz files (currently not used).

    For each (x, y) coordinate in feat_xy_list, extract and concatenate
    features from each intermediate level ('p1', 'p2', 'p3', 'p4').
    """
    # Build the npz filename from the provided fullpath.
    filename = os.path.splitext(os.path.basename(intermediate_fullpath))[0]
    npz_filename = f"{filename}.npz"
    full_path = os.path.join(INTERMEDIATE_DIRPATH, npz_filename)

    data = np.load(full_path, allow_pickle=True)
    results = []

    for (x, y) in feat_xy_list:
        feature_list = []
        for pn in ['p1', 'p2', 'p3', 'p4']:
            tensor_pn = data[pn]
            # tensor_pn shape assumed to be (batch, channels, height, width)
            # Since the model does not resize angle, h_ratio is always 1.
            h, w = tensor_pn.shape[-2:]
            # Calculate scaling ratios (adjust according to your model's resizing behavior)
            h_ratio = h / NUM_ANGLE
            w_ratio = w / NUM_ANGLE

            try:
                x_scaled = int(np.round(x * w_ratio))
                y_scaled = int(np.round(y * h_ratio))
                feat = tensor_pn[0, :, y_scaled, x_scaled]
                feature_list.append(feat)
            except IndexError:
                # In case the scaled coordinate is out of bounds.
                feature_list.append(None)

        # If any feature is None, we assign None as the result for this coordinate.
        if any(f is None for f in feature_list):
            concatenated_feature = None
        else:
            # Concatenate features along the channel dimension.
            concatenated_feature = np.concatenate(feature_list, axis=0)

        results.append({"coords": (x, y), "feature": concatenated_feature})

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
    ref_feat_xy_list = find_hough_lines_reference(reference_file, method=method_)

    if layer == "intermediate":
        ref_entries = extract_intermediate_feature(reference_file, ref_feat_xy_list, layer, size)
    elif layer == "concat":
        ref_entries = extract_concat_feature(reference_file, ref_feat_xy_list, layer, size)
    rep_ref = ref_entries[0]["feature"] if ref_entries and ref_entries[0]["feature"] is not None else None
    ref_key = os.path.splitext(os.path.basename(reference_file))[0]

    match_dict = {}  # Will store matches for each target.

    # Process each target image.
    for file_path in target_files:
        feat_xy_list = find_hough_lines(file_path, method=method_)
        target_entries = extract_intermediate_feature(file_path, feat_xy_list, layer, size)
        # target_entries = extract_concat_feature(file_path, feat_xy_list, layer, size)
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
        b_points = reverse_mapping([best_match["coords"]], numAngle, numRho, size=orig_size)
        if not b_points or b_points[0] is None:
            continue

        # Optionally, apply scaling if needed.
        # (Here we assume a scale from orig_size (640x640) to the desired output size, e.g., 2400x640.)
        scale_w = IMAGE_WIDTH / MODEL_WIDTH
        scale_h = IMAGE_HEIGHT / MODEL_HEIGHT
        for i in range(len(b_points)):
            y1 = int(np.round(b_points[i][0] * scale_h))
            x1 = int(np.round(b_points[i][1] * scale_w))
            y2 = int(np.round(b_points[i][2] * scale_h))
            x2 = int(np.round(b_points[i][3] * scale_w))
            if x1 == x2:
                angle = -np.pi / 2
            else:
                angle = np.arctan((y1 - y2) / (x1 - x2))
            (x1, y1), (x2, y2) = get_boundary_point(y1, x1, angle, IMAGE_HEIGHT, IMAGE_WIDTH)
            b_points[i] = (y1, x1, y2, x2)
        img_vis = visulize_mapping(b_points, (IMAGE_WIDTH, IMAGE_HEIGHT), target)
        cv2.imwrite(os.path.join(OUTPUT_DIRPATH, rf"match_{layer}_ref_{target}.jpg"), img_vis)



# --- Main processing ---
if __name__ == "__main__":
    # layer = "concat"
    layer = "intermediate"

    if layer == "concat":
        target_files = [os.path.join(CONCAT_DIRPATH, filename)
                        for filename in os.listdir(CONCAT_DIRPATH)
                        if filename.endswith("npy")]
    elif layer == "intermediate":
        target_files = [os.path.join(INTERMEDIATE_DIRPATH, filename)
                        for filename in os.listdir(INTERMEDIATE_DIRPATH)
                        if filename.endswith("npz")]
    else:
        assert f"layer name is not valid :{layer}"

    reference_file = target_files.pop(0)
    print(f"=> reference :{reference_file}, target_files :{target_files}")

    size = get_size()
    print(f"size :{size}, layer :{layer}")

    # Compute the match dictionary: reference vs. targets.
    match_dict = compute_match_dict_reference(REFERENCE_IMG_FULLPATH, target_files, method_="max", layer=layer)
    print("Match Dictionary")
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
    visualize_matches(REFERENCE_IMG_FULLPATH, match_dict, numAngle=NUM_ANGLE, numRho=NUM_RHO, orig_size=(MODEL_WIDTH, MODEL_HEIGHT), layer=layer)
