from ultralytics import YOLO
from ultralytics import settings

if __name__ == "__main__":
    # Ensure settings pointing to correct paths
    settings.reset()
    settings.update({"datasets_dir": "datasets"})
    settings.update({"weights_dir": "weights"})
    settings.update({"runs_dir": "runs"})

    # Here loading one of our pretrained models, but could also just use yolo11n.pt
    model = YOLO('runs/detect/train/weights/best.pt') 

    fgsm_results = model.train(data="VisDrone_FGSM_adv.yaml", epochs=5, imgsz=640)
    cw_results = model.train(data="VisDrone_CW_adv.yaml", epochs=5, imgsz=640)
