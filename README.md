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

Data dapat diakses melalui Yahoo Finance dengan simbol 'BTC-USD' atau melalui API yfinance: https://finance.yahoo.com/quote/BTC-USD/history

### **Informasi Dataset**
Berdasarkan hasil eksplorasi awal terhadap dataset, ditemukan informasi sebagai berikut:

* Jumlah data: 3,872 baris (observasi harian) dan 6 kolom
* Periode waktu: 29 September 2014 hingga 14 Mei 2024
* Format data: Data deret waktu harian

### **Variabel-variabel pada Bitcoin Price dataset adalah sebagai berikut:**

![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1745988776/Screenshot_2025-04-30_115234_sefc4h.png)

### **Kondisi Data**

1. Missing Values: 
* Tidak ditemukan missing values pada dataset

2. Duplikasi Data:
* Tidak ditemukan data duplikat pada dataset

3. Outliers:
* Menggunakan metode IQR (Interquartile Range) terdeteksi 123 data (3.18%) yang berada di luar batas IQR
* Q1 (25%): $1,746.90
* Q3 (75%): $35,440.41
* IQR: $33,693.51
* Batas bawah: -$48,793.37
* Batas atas: $85,980.68

### **Exploratory Data Analysis**

#### **Deskripsi Variabel**

1. Dataset terdiri dari 3,872 baris dan 6 kolom, mencakup data harian harga Bitcoin.
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

##### **Insight:**

* Distribusi harga Bitcoin sangat condong ke kanan (positively skewed), menunjukkan banyak observasi pada harga rendah dan beberapa observasi pada harga sangat tinggi
* Distribusi memiliki ekor tebal (heavy-tailed), mengindikasikan frekuensi kejadian ekstrem yang lebih tinggi dari distribusi normal
* Range harga sangat lebar, dari 106,146.27, menunjukkan volatilitas ekstrem

#### **Analisis Univariat**
1. Tren Harga Bitcoin Over Time:

![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1745989998/Screenshot_2025-04-30_121241_ztt5l3.png)

##### **Insight:**
* Terlihat beberapa siklus bull market yang jelas (2017, 2021, dan 2024)
* Periode konsolidasi dan penurunan (bear market) setelah setiap bull run
* Tren jangka panjang menunjukkan kenaikan meskipun dengan volatilitas tinggi

2. Distribusi Harga Penutupan:

![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1745990246/Screenshot_2025-04-30_121705_mt2vkn.png)

##### **Insight:**
* Distribusi miring ke kanan
* Mayoritas data terkonsentrasi di bagian kiri (harga rendah hingga menengah)
* Beberapa observasi ekstrem di sisi kanan (harga tinggi)

3. Return Harian:

![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1745990345/Screenshot_2025-04-30_121837_tw7h9b.png)

##### **Insight:**
* Return harian sangat volatil, berkisar dari -40% hingga +25%
* Volatilitas harian (standar deviasi return): 3.5991%
* Distribusi return mendekati simetris tetapi memiliki ekor tebal (leptokurtic)

4. Autokorelasi:

![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1745990747/Screenshot_2025-04-30_122517_wbkbuo.png)

##### **Insight:**
* Terdapat autokorelasi positif yang signifikan pada lag pendek
* Pola autokorelasi menurun perlahan, menunjukkan adanya memori panjang (long memory) dalam data
* Pola ini mendukung penggunaan model LSTM yang dapat menangkap dependensi jangka panjang

#### **Analisis Multivariate**

1. Matrix Korelasi:
   
![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1745990897/matrix_e1hm23.png)

* Korelasi sangat tinggi (>0.99) antara Open, High, Low, dan Close
* Korelasi moderat antara harga dan volume (sekitar 0.66)
* High dan Low memiliki korelasi tertinggi (0.9996)

1. Hubungan Harga dan Volume:
   
![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1745991165/Screenshot_2025-04-30_123220_avhjfn.png)

* Hubungan antara volume dan harga penutupan Bitcoin cenderung positif, tapi tidak linear dan tersebar luas.
* Harga tinggi sering disertai dengan volume besar, tapi volume besar tidak selalu menghasilkan harga tinggi.
* Terdapat banyak variasi harga untuk volume yang sama, menandakan pengaruh faktor lain (misalnya sentimen pasar, berita, dll).
* Outlier volume yang ekstrem bisa mengganggu analisis korelasi dan perlu ditinjau lebih lanjut.
* Analisis kuantitatif lebih lanjut (misalnya korelasi Pearson/Spearman, atau regresi) akan memberikan gambaran yang lebih akurat.

1. Moving Averages:
   
![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1746000185/Screenshot_2025-04-30_150239_ogytvu.png)

* MA7 (rata-rata 7 hari) lebih volatile dan mengikuti pergerakan harga jangka pendek
* MA30 (rata-rata 30 hari) lebih halus dan menangkap tren jangka menengah
* MA90 (rata-rata 90 hari) menunjukkan tren jangka panjang dengan lebih jelas
* Perpotongan MA dapat mengindikasikan perubahan tren

## **Data Preparation**

Tahap persiapan data sangat penting untuk memastikan kualitas input yang diberikan kepada model machine learning. Berikut adalah tahapan-tahapan persiapan data yang dilakukan: 

### **1. Konversi Tipe Data**

Langkah pertama adalah memastikan semua kolom memiliki tipe data yang sesuai:

![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1746000524/Screenshot_2025-04-30_150828_ojwpwu.png)

Hasil: Semua kolom berhasil dikonversi ke tipe data yang sesuai (datetime untuk Date, float64 untuk kolom harga, dan int64 untuk Volume).

### **2. Penanganan Missing Value**

Meskipun analisis awal menunjukkan tidak ada missing value, dilakukan pemeriksaan ulang dan penanganan untuk memastikan integritas data:

![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1746000658/Screenshot_2025-04-30_151033_luevij.png)

Hasil: Tidak ditemukan missing value yang perlu ditangani.

### **3. Penanganan Outliers**

Setelah analisis menggunakan metode IQR, diputuskan untuk mempertahankan outliers karena:

* Outliers merepresentasikan pergerakan pasar yang valid dalam konteks cryptocurrency
* Menghapus outliers dapat menghilangkan informasi penting tentang volatilitas pasar
* Outliers merupakan bagian dari karakteristik alami data keuangan

### **4. Feature Engineering**

Beberapa fitur tambahan dibuat untuk memperkaya analisis:

1. Daily Return: Menghitung persentase perubahan harga harian
   
![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1746001584/Screenshot_2025-04-30_152559_dufo7h.png)
   
   Formula: (Close_today - Close_yesterday) / Close_yesterday * 100

1. Moving Averages: Menghitung rata-rata bergerak untuk mengidentifikasi tren
   
![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1746001660/Screenshot_2025-04-30_152721_lzyc26.png)
   
   Formula MA: Σ(Close_prices_in_window) / window_size

Hasil: Berhasil menambahkan 4 fitur baru yang memberikan perspektif tambahan tentang pergerakan harga.

### **5. Normalisasi Data**

Normalisasi diperlukan karena:

* Neural networks (termasuk LSTM) bekerja lebih baik dengan data yang dinormalisasi
* Mencegah dominasi fitur dengan skala besar
* Mempercepat konvergensi selama proses training
  
![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1746001748/Screenshot_2025-04-30_152848_kdrgrp.png)

Formula MinMaxScaler: X_scaled = (X - X_min) / (X_max - X_min)

Hasil: Data harga Bitcoin berhasil dinormalisasi ke dalam range 0-1.

### **6. Persiapan Data Time Series**

Data deret waktu memerlukan format khusus untuk model LSTM:

1. Pembagian Data: 80% training, 20% testing
   
![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1746001931/Screenshot_2025-04-30_153055_p2mtvt.png)

2. Sliding Window: Menggunakan 60 hari data untuk memprediksi hari berikutnya
   
![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1746002017/Screenshot_2025-04-30_153319_jz6jji.png)

3. Reshaping Data: Format untuk LSTM [samples, time steps, features]
   
![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1746002127/Screenshot_2025-04-30_153508_rkxr6f.png)

Hasil:
* X_train shape: (2,800, 60, 1)
* y_train shape: (2,800,)
* X_test shape: (641, 60, 1)
* y_test shape: (641,)

Alasan pemilihan time step 60 hari:
* Menangkap pola musiman bulanan (lebih dari 2 bulan)
* Menyeimbangkan antara memori jangka pendek dan jangka panjang
* Berdasarkan penelitian sebelumnya yang menunjukkan efektivitas window 1-3 bulan untuk prediksi cryptocurrency

## **Modeling**

Pada proyek ini, model LSTM (Long Short-Term Memory) dipilih untuk memprediksi harga Bitcoin. LSTM adalah jenis arsitektur Recurrent Neural Network (RNN) yang dirancang khusus untuk mengatasi masalah vanishing gradient yang sering terjadi pada RNN standar, sehingga mampu mempelajari dependensi jangka panjang dalam data deret waktu.

### **1. Cara Kerja LSTM**

LSTM bekerja dengan mekanisme sel memori dan tiga gerbang kontrol:

1. Forget Gate : Menentukan informasi mana dari sel memori sebelumnya yang harus dibuang
2. Input Gate : Terdiri dari dua komponen:
3. Cell State Update: Memperbarui sel memori dengan menggabungkan informasi lama dan baru
4. Output Gate : Menentukan bagian mana dari sel memori yang akan dioutputkan

Kelebihan LSTM:

* Mampu mempelajari dependensi jangka panjang dalam data deret waktu
* Mengatasi masalah vanishing gradient dengan mekanisme gerbang
* Selektif dalam mempertahankan informasi yang relevan dan membuang yang tidak relevan
* Sangat cocok untuk prediksi deret waktu dengan pola kompleks seperti harga Bitcoin

### **2. Arsitektur Model LSTM**

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

1. Root Mean Squared Error (RMSE)
* Mengukur akar rata-rata dari kuadrat error
* Memberikan penalti lebih besar untuk error yang besar
* Dalam satuan yang sama dengan data asli (USD)
* Formula: ![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1746002910/Screenshot_2025-04-30_154756_ckoz36.png)
 
2. Mean Absolute Error (MAE)
* Mengukur rata-rata dari nilai absolut error
* Lebih robust terhadap outliers dibandingkan RMSE
* Dalam satuan yang sama dengan data asli (USD)
* Formula: ![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1746003096/Screenshot_2025-04-30_155118_dsjmgs.png)

3. R-squared (R²)
* Mengukur proporsi variasi dalam variabel dependen yang dapat dijelaskan oleh model
* Range 0-1, dimana nilai lebih tinggi menandakan model lebih baik
* Formula: $1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$

4. Mean Absolute Percentage Error (MAPE)
* Mengukur rata-rata persentase error
* Memungkinkan perbandingan error relatif terhadap nilai aktual
* Formula: ![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1746003186/Screenshot_2025-04-30_155252_qa5lp9.png)

### **2. Hasil Evaluasi Model LSTM**

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

![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1746003337/Screenshot_2025-04-30_155507_zgyvjk.png)

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