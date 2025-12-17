# 📘 Zoo Animal Classification

Proyek klasifikasi hewan menggunakan Machine Learning dan Deep Learning untuk memprediksi tipe/kelas hewan berdasarkan karakteristik fisiknya.

## 👤 Informasi
- **Nama:** Arya Dwipa Mukti
- **NIM:** 233307037
- **Repo:** https://github.com/aryacahil/UAS-DATASCIENCE.git
- **Video:** https://youtu.be/Xcr4bI9KoqY?si=HM_0ROXbx4FIdj9t

---

## 1. 🎯 Ringkasan Proyek
Proyek ini bertujuan untuk mengklasifikasikan 101 hewan ke dalam 7 kategori berdasarkan 16 atribut karakteristik fisik. Pendekatan yang digunakan:
- Melakukan Exploratory Data Analysis (EDA)
- Melakukan data preparation dan feature engineering
- Membangun 3 model: **Decision Tree (Baseline)**, **Random Forest (Advanced)**, **Neural Network (Deep Learning)**
- Melakukan evaluasi dan menentukan model terbaik

---

## 2. 📄 Problem & Goals

**Problem Statements:**
1. Bagaimana cara mengklasifikasikan hewan berdasarkan karakteristik fisiknya secara otomatis?
2. Model mana yang paling efektif untuk klasifikasi multi-class pada dataset Zoo?
3. Apakah deep learning memberikan performa lebih baik dibanding model tradisional pada dataset kecil?

**Goals:**
1. Membangun sistem klasifikasi hewan dengan akurasi minimal 80%
2. Membandingkan performa 3 jenis model (baseline, advanced, deep learning)
3. Mengidentifikasi fitur-fitur penting yang menentukan klasifikasi hewan
4. Membuat model yang reproducible dan dapat di-deploy

---

## 📁 Struktur Folder
```
zoo-classification/
│
├── data/                   # Dataset
│   └── zoo.data           # Data hewan (101 instances)
│
├── notebooks/              # Jupyter notebooks
│   └── ML_Project.ipynb   # Notebook utama proyek
│
├── src/                    # Source code 
│
├── models/                 # Saved models
│   ├── model_baseline.pkl # Decision Tree model
│   ├── model_rf.pkl       # Random Forest model
│   └── model_nn.h5        # Neural Network model
│
├── images/                 # Visualizations
│   ├── correlation_heatmap.png
│   ├── class_distribution.png
│   ├── feature_importance.png
│   └── training_history.png
│
├── requirements.txt        # Dependencies
├── .gitignore
└── README.md
```

---

## 3. 📊 Dataset

- **Sumber:** [UCI Machine Learning Repository - Zoo Dataset](https://archive.ics.uci.edu/ml/datasets/zoo)
- **Jumlah Data:** 101 instances
- **Jumlah Fitur:** 17 (16 features + 1 target)
- **Tipe:** Tabular Data (Boolean & Numeric)
- **Target:** 7 kelas (Mammal, Bird, Reptile, Fish, Amphibian, Bug, Invertebrate)

### Fitur Utama
| Fitur | Tipe | Deskripsi |
|-------|------|-----------|
| hair | Boolean | Memiliki rambut/bulu |
| feathers | Boolean | Memiliki bulu burung |
| eggs | Boolean | Bertelur |
| milk | Boolean | Menyusui |
| airborne | Boolean | Dapat terbang |
| aquatic | Boolean | Hidup di air |
| predator | Boolean | Predator |
| toothed | Boolean | Memiliki gigi |
| backbone | Boolean | Memiliki tulang belakang |
| breathes | Boolean | Bernapas |
| venomous | Boolean | Berbisa |
| fins | Boolean | Memiliki sirip |
| legs | Numeric | Jumlah kaki (0,2,4,5,6,8) |
| tail | Boolean | Memiliki ekor |
| domestic | Boolean | Hewan peliharaan |
| catsize | Boolean | Berukuran seperti kucing |
| type | Numeric | Kelas hewan (1-7) |

---

## 4. 🔧 Data Preparation

### 4.1 Data Cleaning
- Tidak ada missing values
- Tidak ada duplicate data
- Dataset sudah bersih dan siap digunakan

### 4.2 Feature Engineering
- Encoding target variable (1-7 → 0-6)
- Normalisasi fitur numerik (StandardScaler)
- Feature selection berdasarkan correlation analysis

### 4.3 Data Splitting
- Training set: 80% (80 samples)
- Test set: 20% (21 samples)
- Stratified split untuk menjaga distribusi kelas

---

## 5. 🤖 Modeling

### Model 1 – Baseline: Decision Tree
- Model sederhana untuk pembanding
- Mudah diinterpretasi
- Tidak memerlukan scaling data

### Model 2 – Advanced: Random Forest
- Ensemble dari multiple decision trees
- Robust terhadap overfitting
- Mampu menangani non-linear relationships

### Model 3 – Deep Learning: Neural Network
- Multilayer Perceptron (MLP)
- Arsitektur: Input (16) → Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dropout(0.3) → Dense(7, Softmax)
- Training dengan early stopping

---

## 6. 🧪 Evaluation

**Metrik:** Accuracy, Precision, Recall, F1-Score

### Hasil Perbandingan Model
| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| Decision Tree | ~90% | ~0.88 | ~0.90 | ~0.89 | <1s |
| Random Forest | ~95% | ~0.94 | ~0.95 | ~0.94 | ~2s |
| Neural Network | ~95% | ~0.93 | ~0.95 | ~0.94 | ~30s |

---

## 7. 🏁 Kesimpulan

### Model Terbaik: Random Forest & Neural Network (Tie)
- **Random Forest**: Akurasi tinggi, training cepat, interpretable
- **Neural Network**: Akurasi setara, lebih flexible untuk data kompleks

### Alasan:
1. Kedua model mencapai akurasi ~95% pada test set
2. Random Forest lebih efisien dalam hal waktu training
3. Neural Network memberikan hasil yang stabil dengan early stopping

### Key Insights:
- Fitur paling penting: `hair`, `milk`, `feathers`, `eggs`, `aquatic`
- Dataset kecil (101 samples) tidak selalu membutuhkan deep learning
- Model tradisional (Random Forest) sangat efektif untuk dataset terstruktur kecil

---

## 8. 🔮 Future Work

- [x] Mengumpulkan lebih banyak data hewan
- [x] Mencoba arsitektur CNN dengan feature extraction
- [x] Hyperparameter tuning dengan Grid/Random Search
- [ ] Deployment ke web application (Streamlit/Gradio)
- [ ] Membuat API dengan Flask/FastAPI
- [ ] Model explainability dengan SHAP/LIME

---

## 9. 🔁 Reproducibility

### Instalasi Dependencies
```bash
pip install -r requirements.txt
```

### Menjalankan Project
```bash
# Clone repository
git clone [URL_REPO_ANDA]
cd zoo-classification

# Install dependencies
pip install -r requirements.txt

# Download dataset (manual) atau jalankan di notebook
# Letakkan zoo.data di folder data/

# Jalankan notebook
jupyter notebook notebooks/ML_Project.ipynb
```

### Google Colab
1. Upload `ML_Project.ipynb` ke Google Colab
2. Upload `zoo.data` ke Colab atau mount Google Drive
3. Run all cells

---

## 📚 Referensi
- UCI Machine Learning Repository: Zoo Dataset
- Forsyth, R. (1990). Zoo Database. UCI Machine Learning Repository.
- Scikit-learn Documentation
- TensorFlow/Keras Documentation

---

## 📧 Contact
[Campgreget2@gmail.com] | [aryacahil]

---

**⭐ Jika project ini bermanfaat, silakan berikan star!**