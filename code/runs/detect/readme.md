## Model evaluation folders

Each of the folders in this directory (.../runs/detect/) contain the metrics (confusion matrices, F1 scores, mAP scores) for different training and validation runs:

* CW_adv: model evaluation after training on Carlini-Wagner images
* FGSM_adv: model evaluation after training on FGSM images
* train: Original fine-tuned model, contains best weights to be used as basis for VisDrone-YOLOv11n model
* val_cw_{c}: validation scores for original-fine tuned models performance on various c values, from 0.1 to 0.4
* val_fgsm_{eps}: validation scores for original-fine tuned models performance on various eps values, from 0.1 to 0.4

