# Agrosheild

Agrosheild is a Streamlit prototype for plant leaf disease screening. Users create an account or sign in, open a private workspace, upload a leaf image, and describe the crop and symptoms. If a trained model is present, the app returns a predicted dataset class and cautious general guidance.

## Features

- Email verification by six-digit, expiring code before an account can sign in
- Registration and sign-in with salted PBKDF2 password hashes
- Responsive workspace with image upload and crop/symptom description
- Image validation, RGB preprocessing, and ResNet18 CNN disease classification
- Tool-grounded crop assessment agent with CNN result context and safe next steps
- General next steps and downloadable assessment summary
- No image or assessment history is saved; the active session holds the uploaded image and notes

The crop agent combines the CNN result, top classes, and curated local advice. It uses built-in guided replies and does not require a separate generative model or Ollama installation.

## Run locally on Windows

From PowerShell:

```powershell
cd "E:\MIT\MDM PROJECT\AgroShield_AI"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
# Optional: use the NVIDIA GPU for training (RTX 2050 and compatible CUDA 12.8 driver)
python -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m streamlit run app.py
```

Open the Local URL printed by Streamlit. Create an account in the **Create account** tab, then enter the six-digit code sent to your email to verify the address and finish registration.

## Configure email verification

The app sends verification codes through SMTP. For local development, create `.streamlit/secrets.toml` (this file is ignored by Git) and add your mail provider's SMTP credentials:

```toml
[email]
smtp_host = "smtp.gmail.com"
smtp_port = 587
smtp_username = "your-sender@gmail.com"
smtp_password = "your-provider-app-password"
from_email = "your-sender@gmail.com"
```

Use an app password or provider-issued SMTP credential; do not use or commit your normal email account password. For Streamlit Community Cloud, add the same values under **App settings → Secrets**. Without these settings, account registration and verification-code delivery remain unavailable.

## Model and dataset

The current trained CNN checkpoint and evaluation metrics are included in `artifacts/`, so predictions are available immediately after deployment. To retrain the model, download the [Plant Leaf Diseases Dataset](https://data.mendeley.com/datasets/tywbtsjrjv/1), extract it under `data/raw`, and run:

```powershell
python scripts\download_plantvillage.py
python scripts\train_cnn.py --data-dir data\raw --epochs 4
```

Training uses an ImageNet-pretrained ResNet18 CNN, image augmentation, and stratified training/validation/test splits. By default it samples up to 250 images per class (balanced across 39 classes) to keep training practical on a laptop; use `--max-per-class 0` to train on every image. The first run downloads pretrained weights. The trained model is saved as `artifacts/agroshield_cnn.pt`; evaluation metrics and the confusion matrix are saved as `artifacts/cnn_metrics.json` and `artifacts/cnn_confusion_matrix.csv`. The Streamlit app loads this CNN for uploaded-image predictions.

The current trained run used 9,652 images across 39 classes, split into 6,756 training, 1,429 validation, and 1,467 held-out test images. It achieved 97.68% accuracy and 0.9769 macro F1 on the test split. This is a PlantVillage benchmark result, not verified field performance.

Softmax scores shown by the app are not calibrated probabilities. A PlantVillage holdout score does not establish accuracy on field photographs, and severity is not predicted.

## Account storage and deployment

This prototype stores account records and short-lived verification state in `data/users.sqlite3`; passwords are salted and hashed, and verification codes expire after 10 minutes and are limited to five attempts. The database is excluded from Git. SQLite on a single app instance is suitable for a classroom/local demonstration, but Streamlit Community Cloud's local filesystem is not a durable account database. For a public multi-user deployment, connect a managed identity provider and persistent database before inviting users to register. Do not commit account databases, passwords, or secrets.

## Limitations

Predictions are model-assisted suggestions, not a confirmed diagnosis. Similar symptoms can have different causes. Guidance is educational and is not a pesticide prescription; confirm disease and locally appropriate controls with an agricultural expert or extension service.
