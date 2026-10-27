# AgroShield AI — Plant Disease Detection Prototype

A Streamlit prototype for classifying supported plant leaf disease classes from an uploaded image. It preprocesses the image, extracts color and gradient features, predicts a class with a scikit-learn Random Forest, and displays general educational prevention guidance.

## Project contents

- `app.py` — Streamlit interface
- `agroshield/` — image feature extraction and general guidance
- `scripts/train_model.py` — training, holdout evaluation, and model export
- `scripts/download_plantvillage.py` — dataset download and extraction helper
- `docs/AgroShield_AI_Project_Report.docx` — final project report
- `artifacts/metrics.json`, `artifacts/confusion_matrix.csv` — recorded evaluation output

The trained model is not committed because it is about 202 MB. The dataset is also not committed (about 868 MB). Download the dataset and train the model locally using the steps below.

## Run on Windows PowerShell

Use Python 3.10 or newer from this repository folder:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\download_plantvillage.py
python scripts	rain_model.py --data-dir dataaw
python -m streamlit run app.py
```

If the dataset archive has already been downloaded, place/extract it so that `data\raw` contains the class folders, then run the training and Streamlit commands. A local trained model is written to `artifacts/agroshield_model.joblib`.

## Dataset

Training data: [Plant Leaf Diseases Dataset (Mendeley Data)](https://data.mendeley.com/datasets/tywbtsjrjv/1), J. Arun Pandian and Geetharamani Gopal (2019), DOI: 10.17632/tywbtsjrjv.1. The recorded training run used 18,410 images across 39 classes, with a stratified 80/20 split (14,728 train, 3,682 holdout). It achieved 89.11% accuracy and 0.8899 macro F1 on that holdout.

## Notes and limitations

The model is a Random Forest over handcrafted HSV color histograms and spatial gradient-orientation histograms. Its displayed class score is not a calibrated probability. PlantVillage-style images may differ from field photographs in lighting, backgrounds, and image quality. Validate on independent field images before relying on results. Severity is not predicted. Guidance is general education, not a diagnosis or pesticide prescription; consult an agricultural expert for decisions.

