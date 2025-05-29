# 📊 China Cancer Patient Records Analysis

Proyek ini merupakan analisis data pasien kanker di Tiongkok menggunakan teknik data preprocessing, eksplorasi data, dan algoritma machine learning untuk klasifikasi. Dataset dianalisis dalam format Jupyter Notebook, dengan fokus pada pembersihan data, penanganan missing value, konversi waktu, dan analisis visual.

---

## 📁 Struktur Proyek

- **`catatan-pasien-kanker-tiongkok.ipynb`**: Notebook utama berisi seluruh tahapan analisis.
- **Data**: Berasal dari direktori `/kaggle/input/`, mencakup catatan pasien kanker.

---

## 🔧 Teknologi yang Digunakan

- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn

---

## 🔍 Ringkasan Analisis

1. **Data Preprocessing**
   - Mengecek missing value
   - Konversi kolom waktu ke format datetime
   - Pengisian missing value dengan nilai "Unknown"

2. **Eksplorasi Data**
   Visualisasi distribusi pasien, fitur kesehatan, dan korelasi antar atribut.

3. **Modeling**
   - Penggunaan algoritma klasifikasi (seperti Decision Tree, Random Forest, dll.)
   - Evaluasi model menggunakan akurasi dan confusion matrix.

---

## 📌 Cara Menjalankan

1. Clone repositori ini:
   ```bash
   git clone https://github.com/Ahmad-Indragiri/china-cancer-patient-analysis.git
   cd china-cancer-patient-analysis
