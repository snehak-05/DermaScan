# DermaScan — AI-Powered Skin Analysis (RandomForest)

**DermaScan** is a lightweight, explainable skin-analysis web app built with **Flask**, **OpenCV**, **skimage** (GLCM), and a **RandomForestClassifier** from scikit-learn.  
It accepts up to 5 user images, extracts handcrafted features (color histograms, texture / GLCM, edge density), predicts the most likely skin condition per image, combines those image-based predictions with user form inputs (age, gender, diet score, stress, water intake), and generates a personalized, human-readable skin care report (`analysis.txt`) with recommendations.

---

## 🔍 Key ideas & highlights

- **Model**: `RandomForestClassifier` (ensemble tree model). Chosen for robustness, interpretability, resistance to overfitting on small-medium tabular feature sets, and fast inference.
- **Features**:
  - **Color histograms** (B, G, R channels, 32 bins each) → captures color/tonal cues.
  - **Texture (GLCM)** features: contrast, dissimilarity, homogeneity, energy, correlation → captures texture patterns (pores, roughness).
  - **Edge density** (Canny edges normalized) → captures micro-texture & irregularities useful for acne / wrinkles.
- **Pipeline**: Feature extraction → model inference (`predict_proba`) → aggregate top class per image → combine with form labels → rule-based personalized analysis & recommendations.
- **Explainability**: Using predicted classes + simple rule-based mapping from form inputs to human-friendly guidance improves trust & auditability (no black-box opaque suggestions).

---

## 🧾 Project structure (recommended)

DermaScan/
│
├── app.py # Flask app (routes: /, /submit, /upload, /result)
├── train_model.py # (optional) training script for RandomForest
├── dermascan_rf_model.joblib # saved trained RandomForest model
├── analysis.txt # generated report (overwritten per user)
├── form_data.csv # appended form submissions (optional)
├── requirements.txt
├── README.md
├── .gitignore
│
├── static/
│ └── uploads/ # uploaded images (reset per session)
│
└── templates/
├── form.html # user form
├── upload.html # image upload page
└── result.html # report page (renders analysis.txt)

---

## 🧠 Model & training details (RandomForest)

**Why RandomForest?**
- Handles tabular numeric input (your handcrafted features).
- Robust to noisy features and small class imbalance.
- Provides `predict_proba()` for confidence scores which are used for report text logic.
- No heavy GPU requirement — good for local/edge deployment.

**Training workflow (example — `train_model.py`):**
1. Load dataset CSV (`skin_features.csv`) where each row = extracted features + `labels`.
2. Split data: `train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)`.
3. Train:
   ```py
   from sklearn.ensemble import RandomForestClassifier
   model = RandomForestClassifier(n_estimators=200, random_state=42)
   model.fit(X_train, y_train)
**Feature Engineering:**
Color Features

32-bin histograms of Blue, Green, Red channels

Capture:

redness

paleness

pigmentation

tone inconsistencies

## **🔁 App flow (user experience)**

- **Form page (/)** — user fills age, gender, skin features (yes/no), diet, stress, water intake.

- **Submit (/submit)** — server saves form data (app.last_form_data) and shows /upload.

- **Upload (/upload)** — user uploads up to 5 images (multipart/form-data). Server saves files to static/uploads (clears old uploads first).

- **Result (/result)** — server:

- - extracts features for each image,

- - runs model.predict_proba() → selects highest probability class per image,

- - composes form_conditions from form flags (acne, oiliness, whiteheads/blackheads → pores, dryness, pigmentation, wrinkles, redness, dark spots),

- - runs personalized_analysis() to produce diet/water/stress/age/gender messages,

- - runs skincare_recommendation() combining model classes + form labels,

- - writes a nicely formatted report file analysis.txt (overwritten for each new user),

- - returns result.html which renders contents of analysis.txt.
