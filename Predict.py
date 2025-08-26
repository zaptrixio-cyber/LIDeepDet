import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
from tqdm import tqdm
from scipy.signal import convolve2d
from mtcnn.mtcnn import MTCNN
import timm
import argparse

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
class InferenceConfig:
    # Path to the trained model weights
    MODEL_PATH = "Outputs/best_checkpoint.pth"
    
    # Number of equidistant frames to sample from the video for inference
    NUM_FRAMES = 100 
    
    # Model architecture details (must match the trained model)
    VIT_MODEL_NAME = 'vit_base_patch16_224'
    EMBED_DIM = 768
    
    # Preprocessing details (must match the trained model)
    IMG_SIZE = (224, 224)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# 2. MODEL DEFINITION (Must be identical to the training script)
# ==============================================================================
class LIDeepDet(nn.Module):
    def __init__(self, vit_model_name, embed_dim, pretrained=False): # Pretrained=False for inference
        super().__init__()
        self.backbone_rgb = timm.create_model(vit_model_name, pretrained=pretrained, num_classes=0)
        self.backbone_illum = timm.create_model(vit_model_name, pretrained=pretrained, num_classes=0)
        self.backbone_material = timm.create_model(vit_model_name, pretrained=pretrained, num_classes=0)
        self.cross_attention = nn.MultiheadAttention(embed_dim, 8, batch_first=True)
        self.classifier = nn.Sequential(nn.LayerNorm(embed_dim * 6), nn.Linear(embed_dim * 6, embed_dim), nn.GELU(), nn.Linear(embed_dim, 1))
    def forward(self, rgb, illum, material):
        f_rgb, f_illum, f_mat = [b.forward_features(x)[:, 0].unsqueeze(1) for b, x in zip((self.backbone_rgb, self.backbone_illum, self.backbone_material), (rgb, illum, material))]
        a_ri, _ = self.cross_attention(f_rgb, f_illum, f_illum)
        a_rm, _ = self.cross_attention(f_rgb, f_mat, f_mat)
        a_ir, _ = self.cross_attention(f_illum, f_rgb, f_rgb)
        a_im, _ = self.cross_attention(f_illum, f_mat, f_mat)
        a_mr, _ = self.cross_attention(f_mat, f_rgb, f_rgb)
        a_mi, _ = self.cross_attention(f_mat, f_illum, f_illum)
        fused = torch.cat([a_ri, a_rm, a_ir, a_im, a_mr, a_mi], dim=-1).squeeze(1)
        return self.classifier(fused)

# ==============================================================================
# 3. PREPROCESSING FUNCTIONS (Must be identical to the preprocessing script)
# ==============================================================================
def extract_illumination_map_paper(face_crop):
    if face_crop is None: return None
    m_hat = np.max(face_crop, axis=-1).astype(np.float32) / 255.0
    guide_image = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    radius, epsilon = 32, 0.01
    guided_filter = cv2.ximgproc.createGuidedFilter(guide=guide_image, radius=radius, eps=epsilon)
    M = guided_filter.filter(src=m_hat)
    smoothed_map = np.uint8(cv2.normalize(M, None, 0, 255, cv2.NORM_MINMAX))
    return cv2.cvtColor(smoothed_map, cv2.COLOR_GRAY2BGR)

def extract_face_material_map_paper(face_crop, mask_size=5):
    if face_crop is None: return None
    gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY).astype(np.float64)
    radius = mask_size // 2
    y, x = np.mgrid[-radius:radius+1, -radius:radius+1]
    denominator = x**2 + y**2 + 1e-12
    kernel_tx, kernel_ty = np.cos(np.arctan2(y, x)) / denominator, np.sin(np.arctan2(y, x)) / denominator
    kernel_tx[radius, radius], kernel_ty[radius, radius] = 0, 0
    gx, gy = convolve2d(gray_face, kernel_tx, 'same', 'symm'), convolve2d(gray_face, kernel_ty, 'same', 'symm')
    magnitude = np.sqrt(gx**2 + gy**2)
    material_map = np.uint8(cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX))
    return cv2.cvtColor(material_map, cv2.COLOR_GRAY2BGR)

# ==============================================================================
# 4. CORE INFERENCE LOGIC
# ==============================================================================
@torch.no_grad()
def predict_video(video_path, model, face_detector, transform, config):
    """
    Processes a single video and returns a prediction score.
    """
    model.eval()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        print("Error: Video has no frames.")
        return None

    indices = np.linspace(0, total_frames - 1, config.NUM_FRAMES, dtype=int) if total_frames > config.NUM_FRAMES else np.arange(total_frames)
    
    batch_rgb, batch_illum, batch_material = [], [], []

    print(f"Processing {len(indices)} frames from the video...")
    for frame_idx in tqdm(indices, desc="Preprocessing Frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_detector.detect_faces(frame_rgb)

        if faces:
            best_face = sorted(faces, key=lambda f: f['confidence'], reverse=True)[0]
            x, y, w, h = best_face['box']
            x, y = max(0, x), max(0, y)
            face_crop = frame[y:y+h, x:x+w]
            
            if face_crop.size == 0: continue
            
            face_crop_resized = cv2.resize(face_crop, config.IMG_SIZE)
            
            illum_map = extract_illumination_map_paper(face_crop_resized)
            material_map = extract_face_material_map_paper(face_crop_resized)
            
            # Convert to RGB PIL images for transforms
            rgb_pil = cv2.cvtColor(face_crop_resized, cv2.COLOR_BGR2RGB)
            illum_pil = cv2.cvtColor(illum_map, cv2.COLOR_BGR2RGB)
            material_pil = cv2.cvtColor(material_map, cv2.COLOR_BGR2RGB)

            batch_rgb.append(transform(rgb_pil))
            batch_illum.append(transform(illum_pil))
            batch_material.append(transform(material_pil))

    cap.release()
    
    if not batch_rgb:
        print("Error: No faces were detected in the sampled frames.")
        return None

    # Stack tensors into a batch and move to device
    rgb_tensor = torch.stack(batch_rgb).to(config.DEVICE)
    illum_tensor = torch.stack(batch_illum).to(config.DEVICE)
    material_tensor = torch.stack(batch_material).to(config.DEVICE)

    print("\nRunning model inference...")
    outputs = model(rgb_tensor, illum_tensor, material_tensor)
    preds = torch.sigmoid(outputs).cpu().numpy()
    
    # Average predictions across all processed frames
    final_score = np.mean(preds)
    
    return final_score

# ==============================================================================
# 5. MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deepfake Detection Inference Script")
    parser.add_argument("video_path", type=str, help="Path to the video file to be analyzed.")
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: The file '{args.video_path}' does not exist.")
        exit()

    # --- 1. Initialize Configuration and Device ---
    config = InferenceConfig()
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # --- 2. Load the Trained Model ---
    print("Loading the trained model...")
    model = LIDeepDet(config.VIT_MODEL_NAME, config.EMBED_DIM).to(device)
    if not os.path.exists(config.MODEL_PATH):
        print(f"Error: Model file not found at '{config.MODEL_PATH}'")
        exit()
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    print("Model loaded successfully.")

    # --- 3. Initialize Preprocessing Tools ---
    face_detector = MTCNN()
    # This transform must be identical to the validation/test transform from training
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- 4. Run Prediction ---
    final_score = predict_video(args.video_path, model, face_detector, transform, config)

    # --- 5. Display Results ---
    if final_score is not None:
        prediction = "FAKE" if final_score > 0.5 else "REAL"
        confidence = final_score if prediction == "FAKE" else 1 - final_score
        
        print("\n" + "="*30)
        print("           INFERENCE RESULT")
        print("="*30)
        print(f"  Prediction:      {prediction}")
        print(f"  Confidence:      {confidence * 100:.2f}%")
        print(f"  Raw Score:       {final_score:.4f} (closer to 1.0 is more likely FAKE)")
        print("="*30)