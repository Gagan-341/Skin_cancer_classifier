# Skin Cancer Classification (Deep Learning, TensorFlow/Keras)

This is a beginner-friendly, **ready-to-run** starter for classifying skin-lesion images into cancer types using transfer learning (EfficientNetB0).  
It uses a simple folder structure and can train on your GPU/CPU.

---

## 0) Problem & Labels
This template expects your data organized **by class folders**. Typical HAM10000/ISIC-2018 style 7 classes:
- `akiec` – Actinic keratoses / intraepithelial carcinoma
- `bcc` – Basal cell carcinoma
- `bkl` – Benign keratosis-like lesions
- `df` – Dermatofibroma
- `mel` – Melanoma
- `nv` – Melanocytic nevi
- `vasc` – Vascular lesions

> You can use fewer/more/different classes – just name folders accordingly inside `data/train` (and optionally `data/test`).

---

## 1) Folder Structure
```
skin-cancer-classifier/
  data/
    train/
      akiec/ ... images ...
      bcc/   ... images ...
      bkl/   ... images ...
      df/    ... images ...
      mel/   ... images ...
      nv/    ... images ...
      vasc/  ... images ...
    test/            # optional: keep a clean hold-out if you have enough data
  models/            # saved models/checkpoints
  train.py
  infer.py
  app.py             # simple Flask API for predictions
  utils.py
  requirements.txt
```

If you're using HAM10000 raw archives, you may first reorganize them into folders by label (a quick script can do this), then place them under `data/train/<label>/`.

---

## 2) Create & Activate Environment
**Option A: pip**
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

**Option B: conda**
```bash
conda create -n skinclf python=3.10 -y
conda activate skinclf
pip install -r requirements.txt
```

---

## 3) Train
```bash
# Basic
python train.py --data_dir data/train --epochs 15

# Common tweaks
python train.py --img_size 224 --batch_size 32 --lr 1e-4 --val_split 0.2 --epochs 25

# Save to a custom path
python train.py --model_out models/efficientnetb0.keras
```

This will:
- Build a tf.data pipeline from your folder.
- Apply on-the-fly augmentation.
- Use EfficientNetB0 + class weights to reduce imbalance bias.
- Save best model, training curves, and a classification report.

Outputs land in `models/` by default:
- `best_model.keras` – Best val accuracy (checkpoint)
- `final_model.keras` – Final epoch model
- `history.png` – Loss/accuracy curves
- `report.json` – per-class precision/recall/F1
- `confusion_matrix.png` – confusion matrix image

---

## 4) Inference (single image)
```bash
python infer.py --model models/best_model.keras --image /path/to/lesion.jpg
```

This prints the predicted class and class probabilities.

---

## 5) REST API (Flask)
```bash
# Start the server
python app.py --model models/best_model.keras --host 0.0.0.0 --port 8000

# Predict (via curl)
curl -X POST http://localhost:8000/predict   -F file=@/path/to/lesion.jpg
```
Response looks like:
```json
{
  "predicted_class": "mel",
  "probs": {"akiec": 0.01, "bcc": 0.02, "...": 0.80, "...": 0.03}
}
```

---

## 6) Notes & Tips
- **Imbalance** is common. This template computes **class weights** automatically.
- Keep a **patient-level split** when possible to avoid leakage (same patient appearing in train & val).
- Consider **color constancy** and **hair removal** preprocessing if artifacts are heavy.
- For clinical or production use, you must perform **proper validation**, calibration, and bias checks. This code is for **educational purposes**.

---

## 7) Troubleshooting
- If TensorFlow fails to install, try a CPU-only version or ensure compatible CUDA/cuDNN drivers for GPU.
- If you see “OOM” (out-of-memory), reduce `--batch_size` or `--img_size`.
- If classes are wrong, ensure folder names match your intended labels.

Happy building!
