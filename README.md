# **Laporan Proyek Machine Learning - Frederikus Felix Bintang Setyawan**

## **Domain Proyek**

Cryptocurrency, khususnya Bitcoin, telah menjadi salah satu aset investasi yang paling volatil dan menarik perhatian dalam dekade terakhir. Sejak diluncurkan pada tahun 2009, Bitcoin telah mengalami fluktuasi harga yang ekstrem—dengan pertumbuhan nilai yang mencapai ribuan persen dalam beberapa periode, diikuti oleh penurunan drastis pada periode lainnya. Volatilitas ini menciptakan peluang dan risiko yang besar bagi investor, trader, dan institusi keuangan.

Prediksi harga Bitcoin merupakan tantangan yang kompleks karena dipengaruhi oleh berbagai faktor, termasuk sentimen pasar, regulasi pemerintah, tingkat adopsi teknologi, kondisi ekonomi global, dan aktivitas spekulasi [1]. Meskipun demikian, kemampuan untuk memprediksi pergerakan harga Bitcoin, bahkan dalam jangka pendek dan dengan tingkat akurasi yang moderat, dapat memberikan keunggulan kompetitif dalam pengambilan keputusan investasi.

Penelitian oleh McNally et al. [2] menunjukkan bahwa pendekatan deep learning seperti Long Short-Term Memory (LSTM) memiliki potensi tinggi dalam memprediksi harga cryptocurrency. Model LSTM terbukti efektif dalam mengenali pola temporal dalam data deret waktu dan menghasilkan prediksi harga Bitcoin yang lebih akurat dibandingkan ARIMA maupun Recurrent Neural Network (RNN) berbasis Bayesian optimization.

Dalam konteks industri keuangan, model prediktif yang akurat dapat membantu investor dalam mengembangkan strategi trading berbasis data, mengelola risiko secara lebih efektif, serta meningkatkan efisiensi alokasi aset [3]. Selain itu, model ini juga dapat digunakan oleh regulator dan pembuat kebijakan sebagai alat bantu untuk memahami dinamika pasar aset digital secara lebih mendalam.

[1] L. Kristoufek, "What are the main drivers of the Bitcoin price? Evidence from wavelet coherence analysis," PLOS ONE, vol. 10, no. 4, 2015. [Online]. Available: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0123923
[2] S. McNally, J. Roche, and S. Caton, "Predicting the price of Bitcoin using Machine Learning," 2018 26th Euromicro International Conference on Parallel, Distributed and Network-based Processing (PDP), pp. 339–343. [Online]. Available: https://ieeexplore.ieee.org/document/8311682
[3] L. Alessandretti, A. ElBahrawy, L. M. Aiello, and A. Baronchelli, "Anticipating cryptocurrency prices using machine learning," Complexity, vol. 2018, Article ID 8983590, 2018. [Online]. Available: https://onlinelibrary.wiley.com/doi/full/10.1155/2018/8983590

## **Business Understanding**

### **Problem Statements**

Beberapa permasalahan yang akan diselesaikan dalam proyek ini:

* Bagaimana mengembangkan model deep learning yang dapat memprediksi harga Bitcoin dengan akurasi yang memadai berdasarkan data historis?
* Seberapa efektif model LSTM dalam menangkap pola temporal dan memprediksi pergerakan harga Bitcoin untuk periode jangka pendek?
* Apakah model prediksi dapat mengidentifikasi tren pergerakan harga Bitcoin untuk 30 hari ke depan dengan tingkat error yang dapat diterima?

### **Goals**

Tujuan dari proyek ini adalah:

* Mengembangkan model deep learning berbasis LSTM yang dapat memprediksi harga penutupan (closing price) Bitcoin dengan error (RMSE) yang minimal.
* Mengevaluasi efektivitas model LSTM dalam memprediksi harga Bitcoin dengan menggunakan berbagai metrik evaluasi termasuk RMSE, MAE, R², dan MAPE.
* Menghasilkan prediksi harga Bitcoin untuk 30 hari ke depan yang dapat digunakan sebagai referensi dalam pengambilan keputusan investasi.

### **Solution Statements**

Untuk mencapai tujuan yang telah ditetapkan, beberapa solusi yang akan diimplementasikan:

1. Melakukan analisis komprehensif terhadap data historis harga Bitcoin untuk memahami pola, tren, dan karakteristik data.

2. Mengembangkan model deep learning menggunakan arsitektur LSTM, yang secara khusus dirancang untuk menangani data deret waktu dan mampu mengingat pola jangka panjang.

3. Mengoptimalkan model dengan menerapkan teknik regularisasi seperti Dropout untuk mencegah overfitting dan meningkatkan generalisasi model.

4. Mengevaluasi performa model menggunakan berbagai metrik evaluasi dan melakukan analisis error untuk memahami keterbatasan model.

5. Menggunakan model terlatih untuk memprediksi harga Bitcoin 30 hari ke depan dan memberikan interpretasi terhadap hasil prediksi.

## **Data Understanding**

Dataset yang digunakan dalam proyek ini adalah data historis harga Bitcoin yang diunduh dari Yahoo Finance menggunakan library yfinance. Dataset mencakup periode dari awal 2014 hingga saat ini, memberikan gambaran komprehensif tentang pergerakan harga Bitcoin sejak tahap awal adopsinya hingga era kematangannya.

### **Variabel-variabel pada House Price Prediction dataset adalah sebagai berikut:**

1. Date: Tanggal observasi harga Bitcoin
2. Open: Harga pembukaan Bitcoin dalam USD pada hari tersebut
3. High: Harga tertinggi Bitcoin dalam USD yang dicapai pada hari tersebut
4. Low: Harga terendah Bitcoin dalam USD yang dicapai pada hari tersebut
5. Close: Harga penutupan Bitcoin dalam USD pada hari tersebut
6. Volume: Jumlah Bitcoin yang diperdagangkan pada hari tersebut

### **Exploratory Data Analysis**

#### **Deskripsi Variabel**

1. Dataset terdiri dari 3,871 baris dan 6 kolom, mencakup data harian harga Bitcoin.
2. Setelah membersihkan data, semua kolom memiliki tipe data yang sesuai:  
    * Date: datetime64[ns]  
    * Close, High, Low, Open: float64  
    * Volume: int64

#### **Statistik Deskriptif**

1. Mean: $21,664.77
2. Std Dev: $24,945.45
3. Min: $178.10
4. Max: $106,146.27
5. Median (50%): $9,686.44
6. Skewness: 1.3309 (positively skewed)
7. Kurtosis: 1.0139 (heavy-tailed)

#### **Missing Value dan Outliers**

1. 1 missing value pada kolom Date telah ditangani.
2. Tidak ada duplikasi data.
3. Outlier pada harga penutupan (metode IQR):  
4. Q1: $1,746.90  
5. Q3: $35,440.41  
6. IQR: $33,693.51  
7. Batas bawah: -$48,793.37  
8. Batas atas: $85,980.68  
9. Jumlah outlier: 123 data (3.18%)  
10. Outlier tidak dihapus karena mencerminkan pergerakan valid di pasar keuangan.

#### **Analisis Univariat**

1. Distribusi harga sangat condong ke kanan, sebagian besar data pada harga rendah.
2. Plot deret waktu menunjukkan tren kenaikan jangka panjang dengan beberapa bull run (2017, 2021, 2024).
3. Return harian sangat volatile, berkisar -40% hingga +25%.  
   Daily volatility (standar deviasi return harian): 3.5991%
4. Boxplot harga penutupan menunjukkan banyak outlier.
5. Autokorelasi harga menunjukkan pola temporal jangka panjang.

#### **Analisis Multivariat**

1. Korelasi sangat tinggi antara Open, High, Low, Close (>0.99).
2. Korelasi harga-volume moderat (sekitar 0.66).
3. Scatter plot harga vs volume memperlihatkan pola positif meski variatif.
4. Moving average (MA7, MA30, MA90) membantu mengidentifikasi tren jangka panjang dan mengurangi noise.

## **Data Preparation**

Beberapa teknik data preparation yang diterapkan dalam proyek ini:

### **1. Normalization**

Harga Bitcoin dinormalisasi menggunakan MinMaxScaler (range 0-1).

### **2. Time Series Data Preparation**

1. Sliding window, time step (lag) = 60 hari.  
2. Data dibagi menjadi training (80%) dan testing (20%).  
3. Format data: [samples, time steps, features].

### **3. Train-Test Split**
1. Training set: 3,096 hari  
2. Testing set: 775 hari  
3. X_train: (3035, 60, 1)  
4. y_train: (3035,)  
5. X_test: (714, 60, 1)  
6. y_test: (714,)

### **4. Feature Engineering**

1. Daily Return  
2. Moving Averages (MA7, MA30, MA90)

## **Modeling**

Pada tahap ini, tiga algoritma machine learning diterapkan untuk memprediksi harga BTC: LSTM (Long Short-Term Memory)

### **1. LSTM (Long Short-Term Memory)**

* LSTM efektif menangkap pola panjang dalam deret waktu dan mengatasi vanishing gradient.
* Arsitektur:
    - LSTM(150, return_sequences=True)
    - Dropout(0.3)
    - LSTM(100, return_sequences=False)
    - Dropout(0.3)
    - Dense(50)
    - Dense(1/output)
    - Optimizer: Adam, learning rate 0.001
    - Loss: Mean Squared Error (MSE)
    - Batch size: 32
    - Epochs: 25
    - Time step: 60 hari

## **Evaluation**

Untuk mengevaluasi performa model, beberapa metrik evaluasi diterapkan:

### **1. Metrik Evaluasi**
* RMSE: 5,957.85 USD
* MAE: 4,996.71 USD
* R²: 0.9384
* MAPE: 8.02%

#### **Interpretasi: **
* R² tinggi (0.9384), model menjelaskan 93.84% variasi harga.
* MAPE 8.02% (rata-rata prediksi menyimpang 8.02% dari aktual).
* RMSE 5,957.85 USD (error moderat untuk volatilitas Bitcoin).

### **2. Analisis Error**
* Mean error: 4,989.61 USD (cenderung under-predict)
* Std error: 3,255.73 USD (variasi error cukup tinggi)
* Error meningkat pada harga tinggi, bias under-prediction pada puncak harga.
* Distribusi error right-skewed (banyak di 2,000–5,000 USD, beberapa di atas 17,500 USD).

## **Future Price Prediction**
Model digunakan untuk prediksi 30 hari ke depan (recursive):

* Harga Bitcoin saat ini: $90,056.65
* Harga prediksi setelah 30 hari: $24,692.70
* Perubahan harga: -$65,363.95 (-72.58%)

### **Tabel Prediksi:**
![alt text](https://res.cloudinary.com/dk2tex4to/image/upload/v1745505347/image_gashxe.png)


  
Prediksi menurun tajam dan konsisten selama 30 hari ke depan. Perlu kehati-hatian: penurunan ekstrem ini bisa jadi akibat bias under-prediction pada harga tinggi.

## **Kesimpulan**
1. Performa Model:  
   R² = 0.9384 dan MAPE = 8.02%. Model mampu menangkap tren, namun cenderung under-predict pada harga tinggi.
2. Insight Data:  
   Volatilitas sangat tinggi (3.6%), distribusi heavy-tailed, korelasi antar harga sangat tinggi, volume berkorelasi moderat.
3. Prediksi Masa Depan:  
   Model memprediksi penurunan drastis 72.58% dalam 30 hari. Hasil ini harus digunakan dengan sangat hati-hati dan dikombinasikan dengan analisis lain.
4. Keterbatasan & Pengembangan:  
   * Hanya menggunakan data harga historis, belum integrasi faktor eksternal.
   * Bisa ditingkatkan dengan data multivariate, ensemble, attention mechanism, dan hyperparameter tuning lebih lanjut.
   * Model statistik atau hybrid dapat digunakan sebagai pembanding.

Model ini memberikan insight dan alat prediksi berbasis data yang dapat membantu investor, trader, dan regulator dalam memahami risiko dan peluang pasar Bitcoin, namun tetap harus digunakan sebagai salah satu referensi, bukan satu-satunya dasar keputusan investasi.