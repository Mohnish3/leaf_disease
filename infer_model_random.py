import os
import sys
import random
from pathlib import Path
import cv2
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torchvision.ops import nms

# =============================================================================
# --- CORE SETUP ---
# =============================================================================

# Add the project root to the Python path to find the 'Torch_model' class
PROJECT_ROOT_PATH = "/Users/mohnishnair/Documents/Python/leaf_disease_d_fine/data_preparation_dfine/custom_d_fine"
sys.path.append(PROJECT_ROOT_PATH)
try:
    from src.infer.torch_model import Torch_model
except ImportError:
    print(f"Error: Could not import Torch_model. Make sure the PROJECT_ROOT_PATH is correct.")
    print(f"Current Path: {PROJECT_ROOT_PATH}")
    sys.exit(1)

# =============================================================================
# --- USER CONFIGURATION ---
# =============================================================================

# 1. Path to the root folder containing your 'train' and 'val' image folders.
INPUT_IMAGES_ROOT = '/Users/mohnishnair/Documents/Python/leaf_disease_d_fine/data_preparation_dfine/compiled_data/Disease_with_leaves/images'


# 2. Path to the root folder for all outputs.
OUTPUT_ROOT = 'outputs_inference_best'

# 3. Model configuration
MODEL_WEIGHTS = './models/soya_disease/model.pt'
TRAINING_CONFIG_FILE = './models/soya_disease/config.yaml'
INFERENCE_IMAGE_SIZE = 800 # Hardcoded based on your previous script

# 4. Inference parameters
DEVICE = 'cpu'
CONFIDENCE_THRESHOLD = 0.6  # Confidence score above which a detection is considered.
NMS_IOU_THRESHOLD = 0.5     # IoU threshold for Non-Max Suppression.
USE_CLASS_AGNOSTIC_NMS = False # Set to True to suppress overlapping boxes even if they are different classes.
BATCH_SIZE = 64 # Number of images to process at once. Adjust based on your GPU memory.

# 5. Supported image extensions
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']


# =============================================================================
# --- BATCH BEHAVIOR & FILTERING ---
# =============================================================================

# --- Output Control ---
SAVE_YOLO_LABELS = False      # Set to True to save predicted labels in YOLO .txt format.
SAVE_INFERENCE_IMAGES = True # Set to True to save images with bounding boxes drawn on them.

# --- Analysis & Summary Control ---
SUMMARIZE_AND_ANALYZE = True # Set to True to generate a summary of detection confidences.

# --- Filename Filtering ---
FILENAME_PREFIX = ""  # Only process images starting with this prefix. Set to "" to disable.
FILENAME_SUFFIX = ""

# --- Random Sampling ---
USE_RANDOM_SAMPLING = True   # Set to True to process a random subset of images.
NUM_RANDOM_SAMPLES = 10      # Number of images to randomly select if USE_RANDOM_SAMPLING is True.

# =============================================================================
# --- AUTOMATIC SETUP & MODEL INITIALIZATION ---
# =============================================================================

print("--- Setting up environment and loading model ---")

# --- Define Output Paths ---
OUTPUT_LABELS_ROOT = os.path.join(OUTPUT_ROOT, 'labels')
OUTPUT_IMAGES_ROOT = os.path.join(OUTPUT_ROOT, 'annotated_images')
OUTPUT_SUMMARY_FILE = os.path.join(OUTPUT_ROOT, 'detection_summary.txt')

# --- [FIX] Ensure the main output directory exists before starting ---
print(f"Ensuring output directory exists: {OUTPUT_ROOT}")
Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)


# --- Load training config ---
print(f"Loading training config from: {TRAINING_CONFIG_FILE}")
with open(TRAINING_CONFIG_FILE, 'r') as f:
    training_cfg = yaml.safe_load(f)

custom_class_names = training_cfg['train']['label_to_name']
NUM_CUSTOM_CLASSES = len(custom_class_names)
print(f"Found {len(custom_class_names)} classes: {list(custom_class_names.values())}")
print(f"Model expects {INFERENCE_IMAGE_SIZE}x{INFERENCE_IMAGE_SIZE} input.")

# --- Initialize the model ---
print("Initializing D-Fine model...")
model = Torch_model(
    model_name='s',
    model_path=MODEL_WEIGHTS,
    n_outputs=NUM_CUSTOM_CLASSES,
    input_width=INFERENCE_IMAGE_SIZE,
    input_height=INFERENCE_IMAGE_SIZE,
    conf_thresh=0.01,  # Keep this low; we will filter by confidence later
    use_nms=False,     # Disable internal NMS to apply our own with custom IoU
    device=DEVICE,
    keep_ratio=True
)
print("✅ Model loaded successfully!")

# =============================================================================
# --- HELPER FUNCTIONS ---
# =============================================================================

def generate_detection_summary(all_confidences, class_names_map, output_path):
    """
    Calculates min, max, and average confidence for each detected class and saves to a file.
    """
    print("\n--- Generating Detection Summary ---")
    summary_lines = []
    header = f"{'ID':<4} {'Class Name':<20} {'Min Conf':<12} {'Avg Conf':<12} {'Max Conf':<12}"
    print(header)
    print("-" * len(header))

    for class_id, class_name in sorted(class_names_map.items()):
        scores_list = all_confidences.get(class_id, [])
        if scores_list:
            min_conf, avg_conf, max_conf = np.min(scores_list), np.mean(scores_list), np.max(scores_list)
            line = f"{class_id:<4} {class_name:<20} {min_conf:<12.4f} {avg_conf:<12.4f} {max_conf:<12.4f}"
        else:
            min_conf, avg_conf, max_conf = 'NaN', 'NaN', 'NaN'
            line = f"{class_id:<4} {class_name:<20} {min_conf:<12} {avg_conf:<12} {max_conf:<12}"
        summary_lines.append(line)
        print(line)

    try:
        with open(output_path, 'w') as f:
            f.write(header + '\n')
            f.write('\n'.join(summary_lines))
        print(f"\n✅ Summary saved successfully to: {output_path}")
    except Exception as e:
        print(f"\n❌ Error saving summary file: {e}")


def apply_custom_nms(predictions, conf_thresh, iou_thresh, class_agnostic=False):
    conf_mask = predictions["scores"] > conf_thresh
    boxes, scores, labels = predictions["boxes"][conf_mask], predictions["scores"][conf_mask], predictions["labels"][conf_mask]
    if len(boxes) == 0: return {"boxes": np.empty((0, 4)), "scores": np.empty((0,)), "labels": np.empty((0,), dtype=np.int64)}
    if class_agnostic:
        boxes_tensor, scores_tensor = torch.from_numpy(boxes).to(model.device), torch.from_numpy(scores).to(model.device)
        keep_indices = nms(boxes_tensor, scores_tensor, iou_thresh).cpu().numpy()
        return {"boxes": boxes[keep_indices], "scores": scores[keep_indices], "labels": labels[keep_indices]}
    else:
        final_boxes, final_scores, final_labels = [], [], []
        for label in np.unique(labels):
            class_mask = labels == label
            class_boxes, class_scores = torch.from_numpy(boxes[class_mask]).to(model.device), torch.from_numpy(scores[class_mask]).to(model.device)
            keep_indices = nms(class_boxes, class_scores, iou_thresh)
            final_boxes.append(class_boxes[keep_indices].cpu().numpy())
            final_scores.append(class_scores[keep_indices].cpu().numpy())
            final_labels.append(np.full_like(class_scores[keep_indices].cpu().numpy(), fill_value=label, dtype=np.int64))
        return {"boxes": np.concatenate(final_boxes) if final_boxes else np.empty((0, 4)), "scores": np.concatenate(final_scores) if final_scores else np.empty((0,)), "labels": np.concatenate(final_labels) if final_labels else np.empty((0,), dtype=np.int64)}

def convert_to_yolo_format(predictions, img_width, img_height):
    yolo_annotations = []
    for i in range(len(predictions['labels'])):
        class_id, (xmin, ymin, xmax, ymax) = predictions['labels'][i], predictions['boxes'][i]
        box_w, box_h = xmax - xmin, ymax - ymin
        center_x, center_y = xmin + box_w / 2, ymin + box_h / 2
        norm_center_x, norm_center_y = center_x / img_width, center_y / img_height
        norm_w, norm_h = box_w / img_width, box_h / img_height
        yolo_annotations.append(f"{class_id} {norm_center_x:.6f} {norm_center_y:.6f} {norm_w:.6f} {norm_h:.6f}")
    return yolo_annotations

def draw_predictions_on_image(image, predictions, class_names_map):
    img_draw = image.copy()
    colors = {i: (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)) for i in class_names_map.keys()}
    for i in range(len(predictions['labels'])):
        class_id, score, (xmin, ymin, xmax, ymax) = predictions['labels'][i], predictions['scores'][i], predictions['boxes'][i].astype(int)
        label_name = class_names_map.get(class_id, f"Class_{class_id}")
        display_text, color = f"{label_name}: {score:.2f}", colors.get(class_id, (0, 255, 0))

        cv2.rectangle(img_draw, (xmin, ymin), (xmax, ymax), color, 2)
        (text_w, text_h), baseline = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_draw, (xmin, ymin), (xmin + text_w, ymin + text_h + baseline), color, -1)
        cv2.putText(img_draw, display_text, (xmin, ymin + text_h + int(baseline/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
    return img_draw

# =============================================================================
# --- MAIN BATCH INFERENCE SCRIPT ---
# =============================================================================

def run_batch_inference():
    """Main function to run inference on all images and save annotations."""
    print("\n--- Starting Batch Inference ---")
    
    input_root = Path(INPUT_IMAGES_ROOT)
    all_image_paths = [p for p in input_root.rglob('*') if p.suffix.lower() in IMAGE_EXTENSIONS]
    
    if FILENAME_PREFIX or FILENAME_SUFFIX:
        filtered_paths = [p for p in all_image_paths if (p.stem.startswith(FILENAME_PREFIX) if FILENAME_PREFIX else True) and (p.stem.endswith(FILENAME_SUFFIX) if FILENAME_SUFFIX else True)]
        image_paths = filtered_paths
        print(f"Applying filters: Prefix='{FILENAME_PREFIX}', Suffix='{FILENAME_SUFFIX}'. Filtered down to {len(image_paths)} images.")
    else:
        image_paths = all_image_paths

    if USE_RANDOM_SAMPLING and len(image_paths) > NUM_RANDOM_SAMPLES:
        print(f"Randomly sampling {NUM_RANDOM_SAMPLES} images from the filtered list.")
        image_paths = random.sample(image_paths, NUM_RANDOM_SAMPLES)

    if not image_paths:
        print(f"Error: No images found to process in '{input_root}' with the specified criteria.")
        return
        
    print(f"Found {len(image_paths)} final images to process in chunks of {BATCH_SIZE}.")
    
    if SUMMARIZE_AND_ANALYZE:
        all_confidences = {class_id: [] for class_id in custom_class_names.keys()}

    with tqdm(total=len(image_paths), desc="Processing Images") as pbar:
        for i in range(0, len(image_paths), BATCH_SIZE):
            batch_paths = image_paths[i:i + BATCH_SIZE]
            
            for image_path in batch_paths:
                img_bgr = cv2.imread(str(image_path))
                if img_bgr is None:
                    print(f"Warning: Could not read image, skipping: {image_path}")
                    pbar.update(1)
                    continue
                
                img_height, img_width, _ = img_bgr.shape
                
                raw_outputs = model(img_bgr)[0]

                final_predictions = apply_custom_nms(
                    raw_outputs, 
                    CONFIDENCE_THRESHOLD, 
                    NMS_IOU_THRESHOLD, 
                    class_agnostic=USE_CLASS_AGNOSTIC_NMS
                )
                
                if SUMMARIZE_AND_ANALYZE:
                    for j in range(len(final_predictions['labels'])):
                        class_id, score = final_predictions['labels'][j], final_predictions['scores'][j]
                        if class_id in all_confidences:
                            all_confidences[class_id].append(score)

                if len(final_predictions['labels']) == 0:
                    pbar.update(1) 
                    continue

                relative_path = image_path.relative_to(input_root)
                
                if SAVE_YOLO_LABELS:
                    yolo_strings = convert_to_yolo_format(final_predictions, img_width, img_height)
                    output_dir = Path(OUTPUT_LABELS_ROOT) / relative_path.parent
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_txt_path = output_dir / (image_path.stem + '.txt')
                    with open(output_txt_path, 'w') as f: f.write('\n'.join(yolo_strings))

                if SAVE_INFERENCE_IMAGES:
                    annotated_image = draw_predictions_on_image(img_bgr, final_predictions, custom_class_names)
                    output_dir = Path(OUTPUT_IMAGES_ROOT) / relative_path.parent
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_img_path = output_dir / image_path.name
                    cv2.imwrite(str(output_img_path), annotated_image)
                
                pbar.update(1)

    if SUMMARIZE_AND_ANALYZE:
        generate_detection_summary(all_confidences, custom_class_names, OUTPUT_SUMMARY_FILE)

    print("\n" + "="*50 + "\n      BATCH INFERENCE COMPLETE\n" + "="*50)
    if SAVE_YOLO_LABELS: print(f"Generated labels saved in: '{OUTPUT_LABELS_ROOT}'")
    if SAVE_INFERENCE_IMAGES: print(f"Generated annotated images saved in: '{OUTPUT_IMAGES_ROOT}'")
    print("="*50)

if __name__ == '__main__':
    run_batch_inference()