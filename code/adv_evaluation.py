from ultralytics import YOLO
import pandas as pd

if __name__ == "__main__":
    model = YOLO("runs/detect/FGSM_adv/weights/best.pt")

    results_clean = model.val(data="yamls/VisDrone.yaml", split='val')
    results_fgsm_adv = model.val(data="yamls/VisDrone_FGSM_adv.yaml", split='test')
    results_cw_adv = model.val(data="yamls/VisDrone_CW_adv.yaml", split='test')

    metrics = {
        "Metric": ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall"],
        "Clean": [
            results_clean.box.map50,  # mAP@0.5
            results_clean.box.map,    # mAP@0.5:0.95
            results_clean.box.mp,     # Mean Precision
            results_clean.box.mr      # Mean Recall
        ],
        "FGSM": [
            results_fgsm_adv.box.map50,
            results_fgsm_adv.box.map,
            results_fgsm_adv.box.mp,
            results_fgsm_adv.box.mr
        ],
        "CW": [
            results_cw_adv.box.map50,
            results_cw_adv.box.map,
            results_cw_adv.box.mp,
            results_cw_adv.box.mr
        ]
    }

    # Create comparison table btwn each of the 3 results
    df = pd.DataFrame(metrics)
    df["FGSM Drop (%)"] = ((df["Clean"] - df["FGSM"]) / df["Clean"] * 100).round(2)
    df["CW Drop (%)"] = ((df["Clean"] - df["CW"]) / df["Clean"] * 100).round(2)

    print("\n Evaluation Comparison Table:\n")
    print(df.to_string(index=False))