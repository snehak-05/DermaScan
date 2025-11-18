# DermaScan — Your Personal AI Skincare Assistant


**DermaScan is an AI-powered skin analysis system that evaluates a user’s skin condition using both form inputs and facial images.**
Many people struggle to understand what specific issues their skin has or what type of skincare routine will suit them. DermaScan solves this problem by detecting skin concerns such as oiliness, dryness, acne, clogged pores, pigmentation, and redness using a trained machine-learning model.

It then combines **ML predictions, user-provided lifestyle details**, and **skin observations** to generate **personalized skincare recommendations** tailored to the user’s unique skin profile.
This helps users understand their skin better and follow a routine that actually suits their skin needs.

DermaScan is designed to be simple, accessible, and helpful for anyone who wants accurate skincare guidance without relying on guesswork.

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

```
DermaScan/
│
├── .idea/
│
├── dataset/
│   ├── acne/
│   ├── pigmentation/
│   ├── milia/
│   ├── oily/
│   ├── dry/
│   ├── wrinkles/
│   ├── dark_spots/
│   ├── pores/
│   └── redness/
│
├── rotated_dataset/
│   ├── acne/
│   ├── pigmentation/
│   ├── milia/
│   ├── oily/
│   ├── dry/
│   ├── wrinkles/
│   ├── dark_spots/
│   ├── pores/
│   └── redness/
│
├── static/
│   └── uploads/      # User images saved here
│
├── templates/
│   ├── form.html
│   ├── result.html
│   └── upload.html
│
├── .gitignore
├── analysis.txt
├── app.py
├── DermaScan.ipynb
├── dermascan_rf_model.joblib
├── form_data.csv
├── README.md
├── requirements.txt
└── skin_features.csv
```

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

---

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

---

## 🚀 Future Enhancements

To make DermaScan more powerful and user-centered, the following improvements are planned for future versions:

### 1. ML-Based Product Recommendation System

A second machine learning model will be introduced to recommend skincare products tailored to the user's skin conditions.

Suggestions will adapt to the user’s budget (budget-friendly skincare).

Option to select Korean skincare, allowing the system to recommend trending K-beauty products.

Product suggestions will match detected issues such as acne, pigmentation, oiliness, dryness, etc.

### 2. Integrated AI Skincare Chatbot

Add an AI-driven chatbot to assist users with:

- basic skincare questions

- ingredient explanations

- routine guidance

- product usage support

This chatbot will improve user interaction and make the platform more helpful.

### 3. Automatic Morning & Night Skincare Routine Generator

A feature that generates a complete personalized skincare routine based on:

- skin type

- detected conditions

- preferred product type

- user's budget

This ensures the user receives a structured and easy-to-follow routine.
