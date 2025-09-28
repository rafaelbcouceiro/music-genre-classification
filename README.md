# 🎶 Music Genre Classification with Machine Learning  

Classifying music genres using audio features and **classical machine learning models** 🎧.  
Originally developed as part of the *Fundamentals of Machine Learning* course, the project has been **translated into English** and **extended with statistical significance testing** to ensure more robust conclusions.  

---

## 🌟 Overview  
This project explores how audio features (tempo, energy, danceability, loudness…) can be used to predict music genres.  
We implemented and compared several classical ML models, and validated our findings not only with metrics but also with **statistical significance tests** ✅.  

---

## 🎼 Dataset  
- Source: [Spotify Audio Features dataset](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db)  
- ~10,000 tracks  
- Features: tempo, energy, danceability, loudness, etc.  
- Target: 10 genres  
- Preprocessing: normalization, one-hot encoding, stratified cross-validation  

---

## 🛠️ Methods  
Models implemented:  
- ⚡ Support Vector Machine (SVM)  
- 🔎 k-Nearest Neighbors (kNN)  
- 🌳 Decision Trees  
- 🧠 Artificial Neural Networks (ANN)  
- 🎲 DoME (Diversity of Models Ensemble)  

**Pipeline**  
1. Data preprocessing  
2. Cross-validation training  
3. Evaluation (accuracy, F1-score, confusion matrices)  
4. Statistical significance testing (paired t-tests)  

---

## 📊 Results (Preview)  
- Best model: **SVM with RBF kernel**  
- Accuracy: **XX%**  
- F1-score: **XX**  

Example outputs:  

![Accuracy Comparison](results/figures/accuracy_comparison.png)  
![Confusion Matrix](results/figures/confusion_matrix_svm.png)  

---

## ⚖️ Statistical Analysis  
To ensure results were not due to chance, we applied **paired t-tests** comparing SVM against other models.  
- ✅ SVM was significantly better than kNN and Decision Trees (*p < 0.05*).  
- ❌ No significant difference was found between SVM and ANN.  

---

## 🚀 How to Run  
Clone the repository and install dependencies:  

```bash
git clone https://github.com/yourusername/music-genre-classification.git
cd music-genre-classification
julia --project
using Pkg; Pkg.instantiate()
