# AgroShield AI

AgroShield AI is a Streamlit prototype for plant leaf disease screening. Users create an account or sign in, open a private workspace, upload a leaf image, and describe the crop and symptoms. If a trained model is present, the app returns a predicted dataset class and cautious general guidance.

## Features

- Registration and sign-in with salted PBKDF2 password hashes
- Responsive workspace with image upload and crop/symptom description
- Image validation, RGB preprocessing, and disease classification
- General next steps and downloadable assessment summary
- No image or assessment history is saved; the active session holds the uploaded image and notes

## Run locally on Windows

From PowerShell:

```powershell
cd "E:\MIT\MDM PROJECT\AgroShield_AI"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

Open the Local URL printed by Streamlit. The first account can be created from the **Create account** tab.

## Model and dataset

The trained model is not included in this repository because it is about 202 MB. The PlantVillage archive is also excluded because it is about 868 MB. To create a local model, download the [Plant Leaf Diseases Dataset](https://data.mendeley.com/datasets/tywbtsjrjv/1) and train it:

```powershell
python scripts\download_plantvillage.py
python scripts\train_model.py --data-dir data\raw
```

The recorded baseline used 18,410 images across 39 classes, with a stratified 80/20 split. It achieved 89.11% accuracy and 0.8899 macro F1 on a 3,682-image holdout. See `artifacts/metrics.json` and `artifacts/confusion_matrix.csv` for the detailed report.

The model is a scikit-learn Random Forest over handcrafted HSV color histograms and spatial gradient-orientation histograms. Its class score is not a calibrated probability. The benchmark does not establish accuracy on field photographs, and severity is not predicted.

## Account storage and deployment

This prototype stores account records in `data/users.sqlite3`; passwords are salted and hashed. The database is excluded from Git. SQLite on a single app instance is suitable for a classroom/local demonstration, but Streamlit Community Cloud's local filesystem is not a durable account database. For a public multi-user deployment, connect a managed identity provider and persistent database before inviting users to register. Do not commit account databases, passwords, or secrets.

## Limitations

Predictions are model-assisted suggestions, not a confirmed diagnosis. Similar symptoms can have different causes. Guidance is educational and is not a pesticide prescription; confirm disease and locally appropriate controls with an agricultural expert or extension service.
