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

# 1. Path to the input video file.
INPUT_VIDEO_PATH = 'path/to/your/input_video.mp4'

# 2. Path for the output annotated video.
OUTPUT_VIDEO_PATH = 'output_video_inference.mp4'

# 3. Model configuration
MODEL_WEIGHTS = './models/soya_disease/model.pt'
TRAINING_CONFIG_FILE = './models/soya_disease/config.yaml'
INFERENCE_IMAGE_SIZE = 800 # Hardcoded based on your previous script

# 4. Inference parameters
DEVICE = 'cuda:1'
CONFIDENCE_THRESHOLD = 0.6  # Confidence score above which a detection is considered.
NMS_IOU_THRESHOLD = 0.5     # IoU threshold for Non-Max Suppression.
USE_CLASS_AGNOSTIC_NMS = True # Set to True to suppress overlapping boxes even if they are different classes.

# 5. Video processing parameters
FRAME_SKIP = 2 # Process 1 frame, then skip the next 2 frames. Set to 0 to process every frame.

# =============================================================================
# --- AUTOMATIC SETUP & MODEL INITIALIZATION ---
# =============================================================================

print("--- Setting up environment and loading model ---")

# --- Ensure output directory exists ---
output_dir = Path(OUTPUT_VIDEO_PATH).parent
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Output will be saved in: {output_dir.resolve()}")


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
# --- HELPER FUNCTIONS (Re-used from original script) ---
# =============================================================================

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

def draw_predictions_on_image(image, predictions, class_names_map):
    # Pro-Tip: For even smoother bounding boxes, consider implementing a simple tracker
    # like SORT or DeepSORT. This would associate object IDs across frames.
    # The current method (holding predictions) is a great, simple starting point.
    img_draw = image.copy()
    colors = {i: (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)) for i in class_names_map.keys()}
    for i in range(len(predictions['labels'])):
        class_id, score, (xmin, ymin, xmax, ymax) = predictions['labels'][i], predictions['scores'][i], predictions['boxes'][i].astype(int)
        label_name = class_names_map.get(class_id, f"Class_{class_id}")
        display_text, color = f"{label_name}: {score:.2f}", colors.get(class_id, (0, 255, 0))

        cv2.rectangle(img_draw, (xmin, ymin), (xmax, ymax), color, 2)
        (text_w, text_h), baseline = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_draw, (xmin, ymin - text_h - baseline), (xmin + text_w, ymin), color, -1)
        cv2.putText(img_draw, display_text, (xmin, ymin - int(baseline/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
    return img_draw

# =============================================================================
# --- MAIN VIDEO INFERENCE SCRIPT ---
# =============================================================================

def run_video_inference():
    """Main function to run inference on a video file and save the output."""
    print("\n--- Starting Video Inference ---")

    # --- Setup Video Capture ---
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {INPUT_VIDEO_PATH}")
        return

    # --- Get Video Properties for Output ---
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input video properties: {frame_width}x{frame_height} @ {fps} FPS, {total_frames} total frames.")
    
    # --- Setup Video Writer ---
    # Using 'mp4v' codec, which is common for .mp4 files.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    # This will store the last valid set of predictions to draw on skipped frames
    last_known_predictions = None

    with tqdm(total=total_frames, desc="Processing Video") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break # End of video

            # --- Frame Skipping Logic ---
            # We process frame 0, then frame (1+SKIP), (2+2*SKIP), etc.
            if frame_count % (FRAME_SKIP + 1) == 0:
                # This is a frame we need to process
                raw_outputs = model(frame)[0]

                final_predictions = apply_custom_nms(
                    raw_outputs,
                    CONFIDENCE_THRESHOLD,
                    NMS_IOU_THRESHOLD,
                    class_agnostic=USE_CLASS_AGNOSTIC_NMS
                )
                last_known_predictions = final_predictions
                
                # Draw the newly computed predictions on the current frame
                annotated_frame = draw_predictions_on_image(frame, final_predictions, custom_class_names)
            
            else:
                # This is a skipped frame. For smoothness, draw the last known predictions.
                if last_known_predictions:
                    annotated_frame = draw_predictions_on_image(frame, last_known_predictions, custom_class_names)
                else:
                    # If no predictions have been made yet, just use the original frame
                    annotated_frame = frame

            # Write the frame (either newly annotated or re-annotated) to the output video
            out.write(annotated_frame)
            pbar.update(1)
            frame_count += 1
    
    # --- Release resources ---
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\n" + "="*50 + "\n      VIDEO INFERENCE COMPLETE\n" + "="*50)
    print(f"Annotated video saved successfully to: '{OUTPUT_VIDEO_PATH}'")
    print("="*50)


if __name__ == '__main__':
    run_video_inference()