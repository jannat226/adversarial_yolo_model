## FGSM attack model
from ultralytics import YOLO
from ultralytics.data.loaders import LoadImagesAndVideos
from ultralytics.utils import IterableSimpleNamespace
from torchattacks import CW
import torch
import torch.nn.functional as F
import cv2
import os
import cv2

from PIL import Image
import numpy as np
import pandas as pd

def load_yolo_labels(label_path):
    if not os.path.exists(label_path):
        return None

    try:
        labels_df = pd.read_csv(label_path, sep=' ', header=None)
        if labels_df.empty:
            return None
        labels = torch.tensor(labels_df.values, dtype=torch.float32)
        return labels
    except pd.errors.EmptyDataError:
        return None

device= "cuda" if torch.cuda.is_available() else "cpu"

def preprocess_image(img_path, size=(640, 640), device="cpu"):
    image = Image.open(img_path).convert("RGB")
    image = image.resize(size)
    image_tensor = torch.from_numpy(np.array(image)).float() / 255.0 
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
    return image_tensor.to(device).requires_grad_(True)

def fgsm_attack(image, model, train_batch, epsilon=0.2):
    image = image.clone().detach().requires_grad_(True)

    pred = model.model(image.to(device))
    loss = model.model.loss(train_batch, pred)[0].sum()

    loss.backward()

    perturbation = epsilon * image.grad.sign()
    adv_image = image + perturbation
    return adv_image.detach().clamp(0, 1)

def cw_attack_yolo(image, model, train_batch, c=0.1, steps=7, lr=0.05):
    image = image.clone().detach().requires_grad_(True)
    delta = torch.zeros_like(image, requires_grad=True, device=image.device)
    optimizer = torch.optim.Adam([delta], lr=lr)

    best_loss = float("inf")
    best_delta = delta.clone().detach()

    for step in range(steps):
        adv_image = torch.clamp(image + delta, 0, 1)
        pred = model.model(adv_image.to(device))  # shape: (batch, num_preds, 85)
        combined_loss_score = model.model.loss(train_batch, pred)[0].sum()

        l2 = F.mse_loss(adv_image, image)
        loss = l2 + c * combined_loss_score

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_delta = delta.clone().detach()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_adv = torch.clamp(image + best_delta, 0, 1)
    return final_adv

# Load YOLO model

model = YOLO('yolo11n.pt').to(device)

model.model.train()
model.model.args = IterableSimpleNamespace(box=1, cls=2, dfl=1)

img_dirs = [
    "datasets/VisDrone/VisDrone2019-DET-val/images",
    "datasets/VisDrone/VisDrone2019-DET-train/images",
    "datasets/VisDrone/VisDrone2019-DET-test-dev/images"
]

for img_dir in img_dirs:
    split = img_dir.rsplit("/")
    label_dir = '/'.join(split[:3]) + '/labels'
    for i, img_file in enumerate(os.listdir(img_dir)):
        if i == 100:
            break
        img_path = os.path.join(img_dir, img_file)
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, base + ".txt")
        label = load_yolo_labels(label_path)
        train_batch = {'cls': label[:, 0], 'bboxes': label[:, 1:], 'batch_idx': torch.zeros(label.shape[0])}
        image = preprocess_image(img_path)
        fgsm_adv_image = fgsm_attack(image, model, train_batch, epsilon=0.1)
        cw_adv_image = cw_attack_yolo(image, model, train_batch, c=0.1)
        adv_imgs = {
            fgsm_adv_image: f"CW_Adv_RobustF_test",
            cw_adv_image: f"CW_Adv_RobustCW_test"
        }
        for adv_img, adv_img_label in adv_imgs.items():
            adv_np = adv_img.squeeze().permute(1, 2, 0).cpu().detach().numpy() * 255
            adv_np = np.clip(adv_np, 0, 255).astype(np.uint8)
            adv_img_dir = f"{split[0]}/{split[1]}_{adv_img_label}/{split[2]}/{split[3]}"
            os.makedirs(adv_img_dir, exist_ok=True)
            adv_img_path = adv_img_dir + '/' + img_file
            cv2.imwrite(
                adv_img_path,
                cv2.cvtColor(adv_np, cv2.COLOR_RGB2BGR)
            )
        print(f"Processed: {img_file}")
    print(f"Processed: {img_dir}")