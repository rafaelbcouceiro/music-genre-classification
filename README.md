# 🎶 Music Genre Classification with Machine Learning  

[![Julia](https://img.shields.io/badge/Language-Julia-9558B2?logo=julia&logoColor=white)](https://julialang.org/)  
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/tomigelo/spotify-audio-features)  

Classifying music genres using audio features and **classical machine learning models** 🎧.  
Originally developed as part of the *Fundamentals of Machine Learning* course, the project has been **translated into English** and **extended with statistical significance testing** to ensure more robust conclusions.  

---

## 🌟 Overview  
This project explores how audio features (tempo, energy, danceability, loudness…) can be used to predict music genres.  
We implemented and compared several classical ML models, and validated our findings not only with metrics but also with **statistical significance tests** ✅.  

---

## 🎼 Dataset  
- Source: [Spotify Audio Features dataset](https://www.kaggle.com/datasets/tomigelo/spotify-audio-features)  
- ~100,000 tracks  
- Features: tempo, energy, danceability, loudness, etc.  
- Target: 10 genres (enriched using **Spotify API**)  
- Preprocessing: normalization, one-hot encoding, stratified cross-validation  
- **Instance reduction** applied to balance the dataset:  
  - Random Undersampling  
  - Edited Nearest Neighbors (ENN)  
- **Files in `data/`:**  
  - `spotify_dataset.csv` – main dataset  
  - `cv_indices.jl` – cross-validation indices for reproducibility  

---

## 🛠️ Methods  
Models implemented:  
- ⚡ Support Vector Machine (SVM)  
- 🔎 k-Nearest Neighbors (kNN)  
- 🌳 Decision Trees  
- 🧠 Artificial Neural Networks (ANN)  
- 🎲 DoME (Developement of Mathematical Expressions)  

**Pipeline:**  
1. Data preprocessing  
2. Cross-validation training  
3. Evaluation (accuracy, F1-score, confusion matrices)  
4. Statistical significance testing (Friedman + Wilcoxon post-hoc)  

---

## 📊 Results (Preview)  
- Best model: **SVM with RBF kernel**  
- Accuracy: **XX%**  
- F1-score: **XX**  

Example outputs:  

[![Accuracy Comparison](results/figures/accuracy_comparison.png)](results/figures/accuracy_comparison.png)  
[![Confusion Matrix](results/figures/confusion_matrix_svm.png)](results/figures/confusion_matrix_svm.png)  

---

## ⚖️ Statistical Analysis  
To ensure results were not due to chance, we applied:  
- **Friedman test** to detect overall differences across models  
- **Wilcoxon post-hoc tests** for pairwise comparisons  

✅ SVM was significantly better than kNN and Decision Trees (*p < 0.05*)  
❌ No significant difference was found between SVM and ANN  

---

## 🔹 Repository Structure  

```
music-genre-classification/
│
├── src/
│   ├── main.jl                # Runs the full pipeline from data loading to evaluation
│   ├── eda.jl                 # Performs exploratory data analysis
│   ├── models.jl              # Defines all ML models
    ├── trainer.jl             # Facilitates experimentation and results management 
│   ├── evaluation.jl          # Computes metrics and plots confusion matrices
│   └── stats_tests.jl         # Performs Friedman + Wilcoxon post-hoc tests
│
├── data/
│   ├── spotify_dataset.csv    # Main dataset
│   └── cv_indices.jl          # Cross-validation indices for reproducibility
│
├── results/
│   └── figures/               # Plots and tables generated
│
├── report/
│   └── full_report.pdf        # Detailed report
│
└── README.md                  # This file
```
---


**Brief explanation of each module:**  
- **main.jl** – Runs the full pipeline from loading data to model evaluation and plotting.  
- **preprocessing.jl** – Handles normalization, one-hot encoding, and instance reduction (Random Undersampling + ENN).  
- **models.jl** – Defines and trains all machine learning models used in the project.  
- **evaluation.jl** – Computes metrics (accuracy, F1-score) and plots confusion matrices, comparisons, etc.  
- **stats_tests.jl** – Performs statistical tests (Friedman + Wilcoxon) to validate results.  

---

## 📝 Full Report  
The report includes:  
- Related work  
- Dataset creation and enrichment process  
- Exploratory Data Analysis (EDA)  
- Preprocessing steps and rationale  
- Model evaluation and result discussion  
- Ideas for future work  

Check the `report/full_report.pdf` for the full document 📄.  

---

## 💻 How to Run  
Clone the repository and install dependencies:  

```bash
git clone https://github.com/yourusername/music-genre-classification.git
cd music-genre-classification
julia --project
using Pkg; Pkg.instantiate()
julia src/main.jl
