import numpy as np
from PIL import Image
import torch

def detect(net, jpeg_path, output_path, resize_val, device=torch.device('cpu')):
    # 1) Load and preprocess
    pil_img = Image.open(jpeg_path).convert("RGB")
    pil_img = pil_img.resize((resize_val, resize_val), resample=Image.NEAREST)
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0  # H×W×3

    # to tensor: 1×3×H×W
    img = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    # 2) Inference
    with torch.no_grad():
        pred = net(img)           # assume output is 1×C×H×W or 1×1×H×W
        pred = pred.squeeze(0)    # C×H×W or 1×H×W

        # 3) Threshold to binary mask
        mask = (pred > 0.03).float()  # same shape as pred

        # 4) If single‑channel, repeat to RGB; if multi‑channel already, clamp to [0,1]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)        # 1×H×W
            mask = mask.repeat(3, 1, 1)      # 3×H×W
        elif mask.shape[0] == 1:
            mask = mask.repeat(3, 1, 1)      # 3×H×W
        else:
            mask = mask.clamp(0, 1)          # C×H×W (C≥3)

    # 5) Convert back to image
    out_arr = (mask.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # H×W×3
    out_img = Image.fromarray(out_arr, mode="RGB")
    out_img.save(output_path)
