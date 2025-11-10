import os
import cv2
import random
import yaml
from tqdm import tqdm

# --- Configuration ---
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# --- Helper Functions ---

def load_class_names_from_yaml(yaml_path: str, classes_to_pick) -> list:
    """
    Loads class names from a YAML file.
    Adapts to a dictionary format like {0: 'name1', 1: 'name2'}.
    """
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if classes_to_pick in data and isinstance(data[classes_to_pick], dict):
            class_dict = data[classes_to_pick]
            sorted_items = sorted(class_dict.items())
            class_names = [value for key, value in sorted_items]
            print(f"[*] Successfully loaded {len(class_names)} class names from '{yaml_path}'.")
            return class_names
        else:
            print(f"[!] Error: YAML file '{yaml_path}' is missing the classes_to_pick key or it's not a dictionary.")
            return []
            
    except FileNotFoundError:
        print(f"[!] Error: YAML file not found at '{yaml_path}'.")
        return []
    except Exception as e:
        print(f"[!] An unexpected error occurred while reading the YAML file: {e}")
        return []

def find_image_label_pairs(root_dir: str):
    """
    Recursively finds all image files and their corresponding label files,
    even if they are in different subdirectories. It matches them based on filename.
    """
    image_map = {}
    label_map = {}
    
    print(f"[*] Scanning '{root_dir}' for all image and label files...")
    
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            name, ext = os.path.splitext(filename)
            if ext.lower() in SUPPORTED_IMAGE_FORMATS:
                image_map[name] = os.path.join(dirpath, filename)
            elif ext.lower() == '.txt':
                label_map[name] = os.path.join(dirpath, filename)

    image_label_pairs = []
    print("[*] Matching images to labels...")
    for name, image_path in image_map.items():
        if name in label_map:
            label_path = label_map[name]
            image_label_pairs.append((image_path, label_path))
    
    if not image_label_pairs:
        print(f"[!] Warning: Found {len(image_map)} images and {len(label_map)} labels, but no pairs matched by filename.")
    else:
        print(f"[+] Found {len(image_label_pairs)} matching image-label pairs.")
        
    return image_label_pairs


# --- Main Plotting Function ---

def plot_annotations(
    root_dir: str,
    output_dir: str,
    yaml_path: str,
    random_mode: bool = False,
    num_samples: int = 10,
    keyword_filter: str = None,  # <-- NEW PARAMETER FOR KEYWORD FILTERING
    box_color: tuple = (0, 255, 0),
    text_color: tuple = (255, 255, 255),
    thickness: int = 2,
    classes_to_pick='combined_classes',
):
    """
    Finds image-label pairs, draws bounding boxes and labels, and saves the results.
    Can filter files based on a keyword in the filename.
    """
    class_names = load_class_names_from_yaml(yaml_path, classes_to_pick)
    if not class_names:
        print("[!] No class names loaded. Cannot proceed with plotting. Exiting.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    all_pairs = find_image_label_pairs(root_dir)

    # --- NEW: Filter pairs based on the keyword if provided ---
    if keyword_filter and keyword_filter.strip():
        keyword = keyword_filter.strip().lower()
        print(f"[*] Applying keyword filter: '{keyword}'")
        
        filtered_pairs = [
            (img_path, lbl_path) for img_path, lbl_path in all_pairs
            if keyword in os.path.basename(img_path).lower()
        ]
        
        if not filtered_pairs:
            print(f"[!] No image files found containing the keyword '{keyword}'. Exiting.")
            return
            
        print(f"[+] Filtered down to {len(filtered_pairs)} matching pairs.")
        all_pairs = filtered_pairs  # Overwrite the list with the filtered results
    # --- END OF NEW BLOCK ---

    if not all_pairs:
        print("[!] No image-label pairs were found to plot. Exiting.")
        return

    if random_mode:
        print(f"[*] Random mode is ON. Selecting up to {num_samples} random samples.")
        if len(all_pairs) < num_samples:
            print(f"[!] Warning: Requested {num_samples} samples, but only found {len(all_pairs)}. Using all found pairs.")
            pairs_to_process = all_pairs
        else:
            pairs_to_process = random.sample(all_pairs, num_samples)
    else:
        print("[*] Random mode is OFF. Processing all found images.")
        pairs_to_process = tqdm(all_pairs, desc="Plotting Annotations")

    for image_path, label_path in pairs_to_process:
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"\n[!] Warning: Could not read image {image_path}. Skipping.")
                continue
            
            h, w, _ = image.shape

            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5: continue
                        
                    class_id, x_center, y_center, box_width, box_height = map(float, parts)
                    class_id = int(class_id)

                    abs_box_w = int(box_width * w)
                    abs_box_h = int(box_height * h)
                    x1 = int((x_center * w) - (abs_box_w / 2))
                    y1 = int((y_center * h) - (abs_box_h / 2))
                    x2 = x1 + abs_box_w
                    y2 = y1 + abs_box_h

                    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, thickness)
                    
                    try:
                        label_text = class_names[class_id]
                    except IndexError:
                        label_text = f"ID:{class_id}"

                    font_scale = max(0.4, min(2.0, abs_box_w / 250.0))
                    (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                    text_x1 = x1
                    text_y1 = y1
                    text_x2 = x1 + text_w + 4
                    text_y2 = y1 + text_h + baseline + 4

                    cv2.rectangle(image, (text_x1, text_y1), (text_x2, text_y2), box_color, -1)
                    cv2.putText(image, label_text, (x1 + 2, y1 + text_h + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
            
            output_filename = os.path.basename(image_path)
            output_save_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_save_path, image)

        except Exception as e:
            print(f"\n[!] An error occurred while processing {image_path}: {e}")
            
    print(f"\n[+] Processing complete. Plotted images are saved in '{output_dir}'.")


if __name__ == '__main__':
    # --- MAIN CONFIGURATION ---
    YAML_FILE_PATH = "classes.yaml"
    INPUT_DATA_DIRECTORY = "/media/HDD04/DGX_Backup/AGTECH-AI-ML/AgTech_AI_ML/data_preparation_dfine/compiled_data/soyabean_disease_dataset/"
    OUTPUT_DIRECTORY = "output_plots"
    PICK_CLASS = 'soyabean_disease_classes'

    # --- CONTROL FLAGS ---
    RANDOM_MODE = True      # True for random samples, False to process all (filtered) files
    NUM_SAMPLES = 10        # Number of samples to process if RANDOM_MODE is True

    # --- FILENAME FILTER ---
    # Set to a string to filter by filename (e.g., "corn", "soyabean"). Case-insensitive.
    # Set to "" or None to disable filtering and process all found files.
    FILENAME_KEYWORD = ""
    
    # --- SCRIPT EXECUTION ---
    # The script will now respect both the filter and the random flag.
    # If a keyword is set, it will first filter, then take a random sample from the filtered results.
    
    print(f"--- Starting Plotter ---")
    print(f"Mode: {'Random' if RANDOM_MODE else 'All'}")
    if FILENAME_KEYWORD:
        print(f"Filename Filter: Active ('{FILENAME_KEYWORD}')")
    else:
        print("Filename Filter: Inactive")
    print("-" * 26)

    plot_annotations(
        root_dir=INPUT_DATA_DIRECTORY,
        output_dir=OUTPUT_DIRECTORY,
        yaml_path=YAML_FILE_PATH,
        random_mode=RANDOM_MODE,
        num_samples=NUM_SAMPLES,
        keyword_filter=FILENAME_KEYWORD,  # Pass the keyword to the function
        classes_to_pick= PICK_CLASS,
    )