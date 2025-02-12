import os
import numpy as np
import cv2
import glob


START_X = 2950
CROP_SIZE = (2400, 640)


def crop_ebr_img(image_fullpath, save_dir, crop_num=32):
    """
    Load an image, crop it into multiple segments of size CROP_SIZE, and save them.

    The cropping starts at a fixed x-coordinate (START_X) and moves
    downwards in equal intervals.

    Args:
        image_fullpath (str): The full path to the image file.
        save_dir (str): Directory where the cropped images will be saved.
        crop_num (int): Number of crops to extract along the height.

    Returns:
        list of str: A list of saved file paths.
    """
    # Load the image
    image = cv2.imread(image_fullpath, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not load image: {image_fullpath}")

    img_height, img_width = image.shape

    crop_width, crop_height = CROP_SIZE

    if START_X + crop_width > img_width:
        raise ValueError(f"Cropping width {crop_width} exceeds image width {img_width} at START_X {START_X}.")

    # Compute interval step
    if crop_num <= 1:
        raise ValueError("crop_num should be at least 2 to define an interval.")

    interval = max(1, (img_height - crop_height) // (crop_num - 1))

    # Create save directory if not exists
    os.makedirs(save_dir, exist_ok=True)

    saved_files = []

    base_filename = os.path.splitext(os.path.basename(image_fullpath))[0]

    for i in range(crop_num):
        start_y = i * interval

        if start_y + crop_height > img_height:
            break  # Avoid out-of-bounds crop

        cropped_img = image[start_y:start_y + crop_height, START_X:START_X + crop_width]

        # Save cropped image
        save_path = os.path.join(save_dir, f"{base_filename}_crop_{i}.jpg")
        cv2.imwrite(save_path, cropped_img)
        saved_files.append(save_path)

    return saved_files

def convert_file_type(folder_fullpath, prev_type="png", convert_type="jpg"):
    """
    Convert all image files of a specified type in a folder to another format.

    Args:
        folder_fullpath (str): The full path to the folder containing images.
        prev_type (str): The file extension of the source images (e.g., "png").
        convert_type (str): The file extension to convert images to (e.g., "jpg").

    Returns:
        list of str: Paths of successfully converted images.
    """
    if not os.path.isdir(folder_fullpath):
        raise ValueError(f"Directory does not exist: {folder_fullpath}")

    converted_files = []
    for filename in os.listdir(folder_fullpath):
        if filename.lower().endswith(f".{prev_type.lower()}"):
            input_path = os.path.join(folder_fullpath, filename)
            output_filename = f"{os.path.splitext(filename)[0]}.{convert_type}"
            output_path = os.path.join(folder_fullpath, output_filename)

            # Load and save the image
            image = cv2.imread(input_path, cv2.IMREAD_COLOR)
            if image is None:
                print(f"Skipping {input_path}: Unable to read image.")
                continue

            success = cv2.imwrite(output_path, image)
            if success:
                converted_files.append(output_path)
                print(f"Converted: {input_path} -> {output_path}")
            else:
                print(f"Failed to save: {output_path}")

    return converted_files

def analyze_result_npy(npy_fullpath):
    npy_file = np.load(npy_fullpath)
    print(npy_file.shape)
    print(npy_file)


def fix_y2_value(line):
    """
    Given a line in the format:
      x1, y1, x2, y2, image_width, image_height

    We overwrite the 4th entry (y2) with '640' unconditionally.
    If the line doesn't have exactly 6 comma-separated parts, return None.
    """
    parts = [p.strip() for p in line.split(',')]

    # We expect exactly 6 parts: indices [0..5]
    if len(parts) != 6:
        return None

    # parts[3] corresponds to y2 in the original format
    parts[3] = "640"

    # Rejoin with comma (and optional space)
    return ", ".join(parts)


def fix_y2_in_folder(folder_path):
    """
    Processes all .txt files in the given folder, sets y2 to '640' in every line,
    and overwrites the original files.
    """
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))

    for txt_file in txt_files:
        # Read all lines
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Process each line
        new_lines = []
        for line in lines:
            new_line = fix_y2_value(line)
            # If valid, add it to the new lines; otherwise, skip or handle differently
            if new_line is not None:
                new_lines.append(new_line + "\n")

        # Overwrite the file
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

        print(f"Processed: {txt_file}")


# Example usage
if __name__ == "__main__":
    folder = r"D:\4. ADC\5.pyeongtaek WIND2 ADC\1.EBR length\Nanya_EBR\1\CROPPED\train"
    fix_y2_in_folder(folder)
# if __name__ == "__main__":
#     result_npy = r"C:\Users\hsji\Downloads\4001.npy"
#     analyze_result_npy(result_npy)

# if __name__ == '__main__':
#     convert_file_type(r"D:\4. ADC\5.pyeongtaek WIND2 ADC\0. E0 ADC\Nanya_EBR\1\CROPPED")


# if __name__ == '__main__':
#     image_path = r"D:\4. ADC\5.pyeongtaek WIND2 ADC\0. E0 ADC\Nanya_EBR\1\6560x50000_9.bmp"
#     save_directory = r"D:\4. ADC\5.pyeongtaek WIND2 ADC\0. E0 ADC\Nanya_EBR\1\CROPPED"
#
#     crop_max_num = 50000 // 640
#     saved_files = crop_ebr_img(image_path, save_directory, crop_num=crop_max_num)
#
#     print("Saved cropped images:")
#     for file in saved_files:
#         print(file)

