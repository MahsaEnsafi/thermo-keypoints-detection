import os
import sys
import cv2
import torch
import numpy as np
import kornia.feature as KF
from scripts.evaluation.COMPARISON import execution

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)

from ..models.r2d2 import Fast_Quad_L2Net_ConfCFS

def load_r2d2_model(weights_path, device):
    checkpoint = torch.load(weights_path, map_location=device)
    model = Fast_Quad_L2Net_ConfCFS()
    if "net_state_dict" in checkpoint:
        state = checkpoint["net_state_dict"]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        raise RuntimeError("The weight file has an invalid structure.")

    model.load_state_dict(state, strict=False)
    model.eval()
    
    model.to(device)
    return model

def preprocess(img_bgr, device):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb).permute(2,0,1).unsqueeze(0).to(device)
    return tensor

def nms(score_map, nms_size=5, device='cuda'):
    pooled = torch.nn.functional.max_pool2d(
        score_map.unsqueeze(0).unsqueeze(0),
        kernel_size=nms_size,
        stride=1,
        padding=nms_size//2
    )
    keep = (score_map == pooled.squeeze())
    return score_map * keep

def detect_r2d2(thrm_input, model, device='cuda', top_k=10000, nms_size=5):
    if isinstance(thrm_input, np.ndarray):
        if thrm_input.ndim == 2:
            img_tensor = torch.from_numpy(thrm_input).float().unsqueeze(0).unsqueeze(0)
            img_tensor = img_tensor.repeat(1, 3, 1, 1)
        elif thrm_input.ndim == 3:
            img_tensor = torch.from_numpy(thrm_input).float().permute(2, 0, 1).unsqueeze(0)
            if img_tensor.shape[1] == 1:
                img_tensor = img_tensor.repeat(1, 3, 1, 1)
            elif img_tensor.shape[1] != 3:
                raise ValueError(f"Expected input with 1 or 3 channels, but got {img_tensor.shape[1]} channels")
        else:
            raise ValueError(f"Unsupported thrm_input dimensions: {thrm_input.ndim}")
        img_tensor = img_tensor.to(device)
    else:
        img_tensor = thrm_input.to(device).float()
        if img_tensor.ndim == 2:
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        elif img_tensor.ndim == 3:
            if img_tensor.shape[0] == 1:
                img_tensor = img_tensor.unsqueeze(0).repeat(1, 3, 1, 1)
            elif img_tensor.shape[0] == 3:
                img_tensor = img_tensor.unsqueeze(0)
            else:
                raise ValueError(f"Unsupported channel-first tensor shape: {tuple(img_tensor.shape)}")
        elif img_tensor.ndim == 4:
            if img_tensor.shape[1] == 1:
                img_tensor = img_tensor.repeat(1, 3, 1, 1)
            elif img_tensor.shape[1] != 3:
                raise ValueError(f"Expected 3 channels, but got {img_tensor.shape[1]} channels")
        else:
            raise ValueError(f"Unsupported thrm_input tensor ndim: {img_tensor.ndim}")

    with torch.no_grad():
        out = model.forward_one(img_tensor)

    descriptors = out["descriptors"]
    reliability = out["reliability"]
    repeatability = out["repeatability"]

    score = (reliability * repeatability).squeeze(0).squeeze(0)

    score = nms(score, nms_size=nms_size, device=device)

    score_np = score.detach().cpu().numpy()
    h, w = score_np.shape

    flat = score_np.reshape(-1)
    idx = np.argsort(flat)[-top_k:]

    ys = idx // w
    xs = idx % w

    kpts = []
    descs = []
    desc_map = descriptors.squeeze(0).detach().cpu().numpy()

    for x, y in zip(xs, ys):
       kpts.append(cv2.KeyPoint(float(x), float(y), size=1))
       descs.append(desc_map[:, y, x])

    descs = np.array(descs)
    
    return kpts, descs
#-------------------------------------------------------------------------------------


def detect_disk(thrm, model, max_kpts=10000):

    img = thrm.astype("float32") / 255.0

    t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    t = t.repeat(1,3,1,1).to(device)

    with torch.no_grad():
        out = model(t)

    feats = out[0]

    kpts = feats.keypoints.detach().cpu().numpy()
    scores = feats.detection_scores.detach().cpu().numpy()
    desc = feats.descriptors.detach().cpu().numpy()

   # --- Select the strongest keypoints ---
    if len(scores) > max_kpts:
        idx = np.argsort(scores)[-max_kpts:]
        kpts = kpts[idx]
        desc = desc[idx]
        scores = scores[idx]

    # --- Convert to OpenCV KeyPoint ---
    cv_kpts = []
    for (x, y), s in zip(kpts, scores):
        kp = cv2.KeyPoint(float(x), float(y), 3)
        kp.response = float(s)
        cv_kpts.append(kp)

    return cv_kpts, desc

def feature_detector():
    r2d2_weights_path = "r2d2/models/r2d2_WASF_N16.pt"
    r2d2 = load_r2d2_model(r2d2_weights_path, device)
    disk=KF.DISK.from_pretrained("depth").to(device)
    MODELS_DICT={"DISK":disk,"R2D2":r2d2}
    DL_FEATURE_METHODA={"DISK":detect_disk,"R2D2":detect_r2d2}
    for name,fn in DL_FEATURE_METHODA.items():
        model=MODELS_DICT.get(name)
        execution(fn,model,name)
    