# Analisis Prediktif Status Survival Pasien Kanker di Tiongkok

Proyek ini berfokus pada analisis data pasien kanker sintetis dari Tiongkok dan pengembangan model *machine learning* serta *deep learning* untuk memprediksi status kelangsungan hidup pasien.

## 📝 Deskripsi Dataset

Dataset yang digunakan dalam analisis ini adalah `/kaggle/input/china-cancer-patient-records/china_cancer_patients_synthetic.csv`. Dataset ini terdiri dari 10.000 entri (pasien) dan 20 kolom (fitur).

**Detail Kolom (Fitur):**

Dataset ini mencakup berbagai jenis informasi, termasuk:
* **Identifikasi Pasien**: `PatientID`
* **Demografi**: `Gender`, `Age`, `Province`, `Ethnicity`
* **Detail Tumor**: `TumorType`, `CancerStage`, `TumorSize`, `Metastasis`
* **Informasi Diagnosis & Perawatan**: `DiagnosisDate`, `TreatmentType`, `SurgeryDate`, `ChemotherapySessions`, `RadiationSessions`
* **Status & Tindak Lanjut**: `SurvivalStatus` (variabel target: 'Alive' atau 'Deceased'), `FollowUpMonths`
* **Faktor Gaya Hidup & Genetik**: `SmokingStatus`, `AlcoholUse`, `GeneticMutation`, `Comorbidities`

**Statistik Deskriptif (Contoh untuk Fitur Numerik):**
* **Age**: Rata-rata 51.6 tahun (Min: 18, Max: 85)
* **TumorSize**: Rata-rata 6.34 cm (Min: 0.5, Max: 14.2)
* **ChemotherapySessions**: Rata-rata 4 sesi (Min: 0, Max: 20)
* **RadiationSessions**: Rata-rata 3 sesi (Min: 0, Max: 30)
* **FollowUpMonths**: Rata-rata 30.4 bulan (Min: 1, Max: 60)

**Nilai yang Hilang (Missing Values):**
Beberapa kolom memiliki nilai yang hilang, di antaranya:
* `SurgeryDate`: Hanya 4327 dari 10000 entri yang memiliki data.
* `AlcoholUse`: Hanya 4079 dari 10000 entri yang memiliki data.
* `GeneticMutation`: Hanya 2800 dari 10000 entri yang memiliki data.
* `Comorbidities`: Hanya 6285 dari 10000 entri yang memiliki data.

## ⚙️ Tahapan Proyek

### 1. Pra-pemrosesan Data
* **Pembersihan Data**:
    * Nilai yang hilang pada kolom `AlcoholUse`, `Comorbidities`, dan `GeneticMutation` diisi dengan kategori "Unknown" untuk mempertahankan informasi.
* **Encoding Fitur Kategorikal**: Fitur-fitur kategorikal (seperti `Gender`, `Province`, `TumorType`, dll.) diubah menjadi representasi numerik menggunakan teknik *one-hot encoding* agar dapat diproses oleh model. Ini menghasilkan total 56 fitur setelah encoding.
* **Pembagian Data**: Dataset dibagi menjadi set pelatihan (8000 sampel) dan set pengujian (2000 sampel).

### 2. Pemodelan dan Evaluasi
Beberapa model klasifikasi dieksplorasi dan dievaluasi:

#### a. Regresi Logistik
* **Akurasi**: 0.7885 (sekitar 79%)
* **Confusion Matrix**:
    ```
    [[ 194  248]
     [ 175 1383]]
    ```
* **Laporan Klasifikasi**:
    * Presisi untuk kelas 0 (misalnya, 'Deceased'): 0.53
    * Recall untuk kelas 0: 0.44
    * Presisi untuk kelas 1 (misalnya, 'Alive'): 0.85
    * Recall untuk kelas 1: 0.89

#### b. Support Vector Machine (SVM)
* **Optimasi Hiperparameter**: Menggunakan pencarian kandidat dengan 3-fold cross-validation.
    * Parameter terbaik yang ditemukan: `{'C': 3.8454, 'class_weight': 'balanced', 'kernel': 'linear'}`
* **Akurasi**: 0.769 (sekitar 77%)
* **Confusion Matrix**:
    ```
    [[ 442    0]
     [ 462 1096]]
    ```
* **Laporan Klasifikasi**:
    * Presisi untuk kelas 0: 0.49
    * Recall untuk kelas 0: 1.00
    * Presisi untuk kelas 1: 1.00
    * Recall untuk kelas 1: 0.70

#### c. Random Forest
* **Optimasi Hiperparameter**: Menggunakan pencarian kandidat dengan 3-fold cross-validation.
    * Parameter terbaik yang ditemukan: `{'n_estimators': 300, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_depth': 10, 'class_weight': None}`
* **Akurasi**: 0.773 (sekitar 77%)
* **Confusion Matrix**:
    ```
    [[ 114  328]
     [ 126 1432]]
    ```
* **Laporan Klasifikasi**:
    * Presisi untuk kelas 0: 0.47
    * Recall untuk kelas 0: 0.26
    * Presisi untuk kelas 1: 0.81
    * Recall untuk kelas 1: 0.92

#### d. Artificial Neural Network (ANN) / Deep Learning
* **Arsitektur Model**:
    * Model Sequential Keras.
    * Layer Input: `Dense` dengan 64 unit, fungsi aktivasi `relu`, `input_shape=(56,)`.
    * Layer Tersembunyi: `Dense` dengan 32 unit, fungsi aktivasi `relu`.
    * Layer Output: `Dense` dengan 1 unit, fungsi aktivasi `sigmoid` (untuk klasifikasi biner).
* **Kompilasi Model**: Optimizer `adam`, loss function `binary_crossentropy`, metrik `accuracy`.
* **Pelatihan**: Dilatih selama 50 epoch.
    * **Hasil Pelatihan (Epoch ke-50)**:
        * Akurasi Latih: ~0.9986
        * Loss Latih: ~0.0227
        * Akurasi Validasi: ~0.7862
        * Loss Validasi: ~1.0774
* **Evaluasi pada Data Uji**:
    * **Akurasi**: 0.787 (sekitar 79%)
    * **Laporan Klasifikasi**:
        * Presisi untuk kelas 0: 0.52
        * Recall untuk kelas 0: 0.54
        * Presisi untuk kelas 1: 0.87
        * Recall untuk kelas 1: 0.86
* **Analisis Bobot Model**:
    * Layer 1 (dense): Shape bobot (56, 64), shape bias (64,)
    * Layer 2 (dense_1): Shape bobot (64, 32), shape bias (32,)
    * Layer 3 (dense_2): Shape bobot (32, 1), shape bias (1,)

## 📊 Ringkasan Hasil Model

| Model                     | Akurasi Test | Presisi (Kelas 0) | Recall (Kelas 0) | Presisi (Kelas 1) | Recall (Kelas 1) |
| :------------------------ | :----------- | :---------------- | :--------------- | :---------------- | :--------------- |
| Regresi Logistik          | ~78.9%       | 0.53              | 0.44             | 0.85              | 0.89             |
| SVM (Optimized)           | ~76.9%       | 0.49              | 1.00             | 1.00              | 0.70             |
| Random Forest (Optimized) | ~77.3%       | 0.47              | 0.26             | 0.81              | 0.92             |
| **ANN (Deep Learning)** | **~78.7%** | **0.52** | **0.54** | **0.87** | **0.86** |

## 📈 Analisis Performa dan Observasi

* **Performa Umum**: Model ANN dan Regresi Logistik menunjukkan akurasi tertinggi pada data uji, sekitar 79%.
* **Overfitting pada ANN**: Terdapat indikasi *overfitting* yang kuat pada model ANN. Akurasi pada data latih mencapai hampir 100%, sementara akurasi pada data validasi jauh lebih rendah dan loss validasi meningkat seiring epoch. Ini menunjukkan model terlalu menghafal data latih dan kurang mampu generalisasi ke data baru.
* **Kinerja Kelas**: Semua model menunjukkan kesulitan dalam memprediksi kelas minoritas (kemungkinan kelas '0' atau 'Deceased' berdasarkan recall yang lebih rendah pada beberapa model atau presisi yang lebih rendah pada kelas tersebut). SVM dengan parameter `class_weight='balanced'` dan kernel linear menunjukkan recall sempurna untuk kelas 0 tetapi dengan presisi yang lebih rendah.
* **Peringatan Debugger & CUDA**: Log menunjukkan beberapa peringatan terkait *frozen modules* pada debugger Python dan registrasi factory untuk cuFFT, cuDNN, dan cuBLAS yang mungkin relevan jika menggunakan GPU untuk pelatihan, namun tidak secara langsung mempengaruhi hasil akhir jika pelatihan berhasil diselesaikan.

## 💡 Kesimpulan dan Potensi Pengembangan Lanjutan

Model *machine learning* dan *deep learning* yang dikembangkan mampu memberikan prediksi status survival pasien dengan akurasi moderat. Model ANN, meskipun mencapai akurasi yang kompetitif, memerlukan penanganan lebih lanjut untuk mengatasi *overfitting*.

Beberapa area untuk pengembangan di masa depan:
1.  **Penanganan Overfitting pada ANN**:
    * Menerapkan teknik regularisasi seperti *Dropout*, L1/L2 regularization.
    * Menggunakan *Early Stopping* berdasarkan performa pada set validasi.
    * Menyederhanakan arsitektur model ANN.
2.  **Penanganan Data Tidak Seimbang**: Mengeksplorasi teknik *oversampling* (misalnya SMOTE) atau *undersampling* untuk menangani potensi ketidakseimbangan kelas dalam variabel target.
3.  **Feature Engineering dan Seleksi Fitur**: Melakukan analisis lebih mendalam untuk memilih fitur yang paling relevan atau membuat fitur baru yang dapat meningkatkan daya prediktif model.
4.  **Interpretasi Model**: Menggunakan teknik seperti SHAP atau LIME untuk lebih memahami bagaimana model membuat prediksi, terutama untuk model yang lebih kompleks seperti Random Forest dan ANN.
5.  **Eksplorasi Model Lain**: Mencoba algoritma *boosting* seperti XGBoost, LightGBM, atau CatBoost yang seringkali memberikan performa tinggi pada data tabular.

