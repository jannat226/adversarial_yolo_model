# 🛰️ Adversarial Training of CNN for Aerial Imagery Object Detection

By:  Jannat Lnu, Jimmy Le, Luke Unterman,My Nguyen 
Course Project | Spring 2025

## 📌 Overview

This project investigates the vulnerability of deep learning-based object detectors—specifically **YOLOv11n**—to adversarial attacks in the context of aerial imagery (e.g., UAV, satellite). It also explores the use of **adversarial training** to improve the model’s robustness without significantly sacrificing accuracy on clean data.

---

## 🎯 Motivation

Deep neural networks (DNNs) are widely used in aerial surveillance and security applications but are highly susceptible to adversarial attacks. These attacks manipulate input images in imperceptible ways to humans, yet lead to critical misclassifications, such as missing or wrongly detecting vehicles or people. This has dangerous implications in defense, autonomous driving, and intelligence operations.

---

## 🧪 Objectives

- Evaluate the performance degradation of YOLOv11n under **FGSM** and **Carlini-Wagner (C&W)** adversarial attacks.
- Fine-tune YOLOv11n using adversarially perturbed images to improve resilience.
- Compare clean vs. adversarial accuracy and analyze cross-attack generalization.

---

## 🧰 Dataset

### 🗂️ Primary Dataset: [VisDrone-DET2019](https://github.com/VisDrone/VisDrone-Dataset)
- **Images**: 6,471 training, 548 validation, 3,190 test
- **Classes**: 10 object types (vehicles, people, etc.)
- **Annotations**: Bounding boxes, visibility, occlusion

### 🗂️ Secondary Dataset: [DOTA-v2.0](https://captain-whu.github.io/DOTA/dataset.html)
- **Images**: 11,268
- **Annotations**: Oriented bounding boxes
- **Use**: Generalization and transferability assessment

---

## ⚙️ Model & Training Details

- **Model**: YOLOv11n via Ultralytics (v8.3.106)
- **Preprocessing**:
  - Image Size: 640x640  
  - Batch Size: 16  
  - Learning Rate: 0.01  
  - Weight Decay: 0.0005  
  - NMS IoU Threshold: 0.7  
- **Hardware**: Windows 11, AMD Ryzen 5600x, RTX 3070, 32 GB RAM

---

## 💥 Adversarial Attack Strategies

### ➤ FGSM (Fast Gradient Sign Method)
- Perturbation Levels (ε): 0.1, 0.2, 0.3, 0.4
- Fast, but creates visible noise

### ➤ Carlini-Wagner (C&W)
- Trade-off Parameter (c): 0.1 to 0.4
- More complex and subtle
- Modified for untargeted attacks on object detectors

---

## 🛡️ Adversarial Training Strategy

Two robust versions of YOLOv11n were created:

- **FGSM_ADV**: Fine-tuned on FGSM-perturbed images
- **CW_ADV**: Fine-tuned on Carlini-Wagner images

Both models were validated on clean and adversarial samples to assess robustness and generalization.

---

## 📊 Evaluation Metrics

- **Primary**: Mean Average Precision (mAP@0.5)
- **Supporting**: Precision, Recall, F1-Score, Confusion Matrices, IoU

---

## 📁 Model Evaluation Folder Structure

Located in `.../runs/detect/`:

| Folder         | Description |
|----------------|-------------|
| `train/`       | Original YOLOv11n model fine-tuned on clean data |
| `FGSM_adv/`    | Evaluation after adversarial training on FGSM examples |
| `CW_adv/`      | Evaluation after adversarial training on CW examples |
| `val_fgsm_{eps}/` | Validation on FGSM attacks for ε = 0.1 to 0.4 |
| `val_cw_{c}/`     | Validation on CW attacks for c = 0.1 to 0.4 |

---

## 📈 Results Summary

### ➤ Clean Image Performance
- Baseline (YOLOv11n): **mAP@0.5 = 0.34**
- FGSM_ADV: **mAP@0.5 = 0.328** (only 3.5% drop)
- CW_ADV: **mAP@0.5 = 0.330**

### ➤ Adversarial Attack Performance

| Model       | FGSM (ε=0.1) | CW (c=0.1) |
|-------------|--------------|------------|
| Baseline    | 91.8% drop   | 66.4% drop |
| FGSM_ADV    | 32.4% drop   | 63.1% drop |
| CW_ADV      | 70.0% drop   | 28.6% drop |

### ➤ Cross-Attack Robustness
- FGSM_ADV generalizes better to CW than CW_ADV does to FGSM.
- CW_ADV reaches **0.541 mAP** on CW attacks (c=0.1), but performs poorly on FGSM, indicating overfitting to CW.

---

## 🧩 Limitations

- Training time increased by 2.3× due to adversarial training
- Modified CW implementation less effective than original
- Dataset bias (Chinese environments) may reduce generalizability
- Only tested digital adversarial scenarios (not physical-world attacks)

---

## 🚀 Conclusion

- Adversarial training significantly improves YOLOv11n’s robustness to FGSM and CW attacks.
- Performance on clean data remains competitive, showing only minor degradation.
- Robust YOLOv11n models are promising for security-critical aerial imagery applications.

---

## 🔮 Future Work

- Extend to additional attacks (e.g., PGD, DeepFool)
- Evaluate against black-box and physical adversarial examples
- Combine defensive strategies (e.g., ensemble learning, input denoising)
- Generalize to more diverse datasets and domains

---

## 📎 References

- VisDrone-DET2019 Dataset: https://github.com/VisDrone/VisDrone-Dataset
- DOTA Dataset: https://captain-whu.github.io/DOTA/dataset.html
- Carlini & Wagner Attack: https://arxiv.org/abs/1608.04644
- FGSM: https://arxiv.org/abs/1412.6572
