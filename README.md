# 🫁 Explainable Pneumonia Detection System

## 📌 Project Overview

This project is an end-to-end Deep Learning solution for diagnosing Pneumonia from Chest X-rays. Unlike standard "black-box" models, this system utilizes **Explainable AI (XAI)** to help clinicians understand the "why" behind a diagnosis.

The model was trained on pediatric data and rigorously validated against adult populations to measure **cross-domain generalization**.

-----

## 🚀 Key Features

  * **Architecture:** **DenseNet121** with Transfer Learning for superior medical feature extraction.
  * **Explainable AI:** Integrated **Grad-CAM** to generate activation heatmaps, visualizing diagnostic focus areas.
  * **Advanced Preprocessing:** Implemented **Center Cropping (85%)** to remove marginal noise and **DICOM-to-JPG** conversion for clinical data compatibility.
  * **Validation:** Cross-validated on the **RSNA Pneumonia Challenge** dataset to evaluate performance on adult anatomy.
  * **Deployment:** Fully functional **Streamlit Dashboard** for real-time inference and visualization.

-----

## 📊 Performance Metrics

The model was optimized for **Recall (Sensitivity)** to ensure minimal false negatives in a clinical setting.

| Dataset | Type | Accuracy | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Kermany (Internal)** | Pediatric | **91.6%** | **95.1%** | 0.93 |
| **RSNA (External)** | Adult | *[Your Result]* | *[Your Result]* | *[Your Result]* |

-----

## 🧠 Explainability & Error Analysis

A critical part of this project was the **XAI Audit**. Initial Grad-CAM heatmaps revealed that the model was over-indexing on edge artifacts (shoulders and film markers).

**The Solution:** I implemented **Geometric ROI Filtering** (Center Cropping). This forced the model to ignore non-biological noise and focus strictly on the lung parenchyma, leading to a more robust and clinically relevant model.

-----

## 🛠️ Installation & Usage

### 1\. Clone the repository

```bash
git clone https://github.com/yourusername/pneumonia-detection-xai.git
cd pneumonia-detection-xai
```

### 2\. Install dependencies

```bash
pip install -r requirements.txt
```

### 3\. Run the Dashboard

```bash
streamlit run app.py
```

-----

## 📂 Repository Structure

  * `app.py`: Streamlit dashboard code.
  * `src/preprocess.py`: DICOM handling and cropping logic.
  * `notebooks/`: Comprehensive training and validation logs.
  * `models/`: Saved weights for the fine-tuned DenseNet121.

-----

## 🎓 Author

**Jagan Satarla** *Engineering Student specializing in AI/ML* [LinkedIn](https://www.google.com/search?q=YOUR_LINKEDIN_URL) | [Portfolio](https://www.google.com/search?q=YOUR_PORTFOLIO_URL)

-----
