<p align="center"> ![LOGO NUSAIRA](https://github.com/user-attachments/assets/e4d1d3ad-1846-43a8-a875-9ae8f70d4c88)

    # NUSAIRA-AAI

<h1 align="center">  This is a Machine Learning / AI Project </h1>

<p align="center"> 

</p>

<div align="center">
    <!-- Your badges here -->
    <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
    <img src="https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white">
    <img src="https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white">
    <img src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white">
    <img src="https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white">
    <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white">
    <img src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white">
    <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white">
    <img src="https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB">
    <img src="https://img.shields.io/badge/tailwindcss-%2338B2AC.svg?style=for-the-badge&logo=tailwind-css&logoColor=white">
</div>

## Teams

- Ahmad Ziyat  (Design Researcher)
- indrawan althaf rosyadi (Machine Learning Engineer)
- Karlina Maelani (Data Engineer)
- Iftikhar Rizqullah (Machine Learning Ops)
- Istiqomah (Machine Learning Ops)
- arya pratama putra (Machine Learning Engineer)

### Idea Background

## Theme
Tema : Agraris/Perikanan/Aquaculture

## Problem yang ditimbulkan
- Manajemen penjualan pada tambak lele yang masih tradisional
- Manajemen pakan pada tambak lele yang tidak terukur
- Pengelolaan air di tambak lele yang tidak optimal
- Menurunnya produktivitas yang bisa menyebabkan kerugian

## Solution
Membuat prediksi harga lele menggunakan AI, chatbot interaktif, dan website untuk membantu petambak lele dalam:

Mengoptimalkan manajemen pakan dengan panduan yang lebih terukur.
Memantau dan mengelola kualitas air secara efisien.
Meningkatkan produktivitas melalui prediksi harga yang akurat dan pengambilan keputusan yang lebih cepat.

# NusAIra 🌊🐟

NusAIra adalah aplikasi web yang dirancang untuk membantu peternak lele dalam mengelola tambak mereka dengan lebih efisien. Aplikasi ini menyediakan berbagai fitur seperti manajemen tambak, pemantauan kesehatan ikan, dan akses ke informasi terkini mengenai budidaya lele.

## Fitur 🌟

- **Manajemen Tambak**: Buat dan kelola tambak lele Anda. 🐟
- **Pemantauan Kesehatan**: Informasi tentang penyakit lele dan cara penanganannya. 🩺
- **Berita dan Informasi**: Akses berita terbaru dan informasi penting terkait budidaya lele. 📰
- **E-Learning**: Akses ke kursus dan materi pembelajaran untuk meningkatkan pengetahuan Anda tentang budidaya lele. 📚
- **Prediksi Harga**: Pantau tren harga lele secara real-time. 📈

## Dataset and Algorithm

### 1. Dataset
- Data Collection <br />
Sumber Data 1 (KKP - Harga dan Produksi Ikan Lele):

Menyediakan data harga ikan lele dari 3 provinsi. Memuat data hasil produksi perikanan, mencakup volume dan nilai produksi. Rasio antara volume produksi dan nilai produksi dihitung untuk mengetahui rata-rata biaya produksi ikan.

Sumber Data 2 (BPS - NTPi untuk Petani):

Data NTPi (Nilai Tukar Petani ikan) dari BPS. Menunjukkan indeks harga yang harus dibayar dan yang diterima oleh petani, memberikan gambaran daya beli petani.
```
https://github.com/AryaPratamaPutra-10/NUSAIRA-AI/blob/main/data_lele.csv
```


- Data Cleaning <br />
Kami menggunakan pandas untuk membersihkan data. Berikut tabel contoh data yang sudah dibersihkan : 


| Jenis Budidaya Pembesaran | Provinsi  | Kabupaten/Kota | Jenis Ikan | Tahun | Volume Produksi | Nilai Produksi         |
|---------------------------|-----------|-----------------|------------|-------|------------------|------------------------|
| KOLAM AIR TENANG          | JAWA BARAT | BANDUNG         | LELE       | 2019  | 5,372,785        | Rp85,964,560,000       |
| KOLAM AIR TENANG          | JAWA BARAT | BANDUNG BARAT   | LELE       | 2019  | 1,328,082        | Rp19,921,230,000       |
| KOLAM AIR TENANG          | JAWA BARAT | BEKASI          | LELE       | 2019  | 2,019,121        | Rp34,325,057,000       |
| KOLAM AIR TENANG          | JAWA BARAT | BOGOR           | LELE       | 2019  | 89,814,340       | Rp1,526,843,780,000    |
| KOLAM AIR TENANG          | JAWA BARAT | CIAMIS          | LELE       | 2019  | 18,162,827       | Rp236,116,751,000     |
| KOLAM AIR TENANG          | JAWA BARAT | CIANJUR         | LELE       | 2019  | 14,124,120       | Rp225,985,920,000     |
| KOLAM AIR TENANG          | JAWA BARAT | CIREBON         | LELE       | 2019  | 9,987,423        | Rp199,748,460,000     |
| KOLAM AIR TENANG          | JAWA BARAT | GARUT           | LELE       | 2019  | 3,162,167        | Rp53,756,839,000      |
| KOLAM AIR TENANG          | JAWA BARAT | INDRAMAYU       | LELE       | 2019  | 70,666,948       | Rp1,060,004,220,000   |
| KOLAM AIR TENANG          | JAWA BARAT | KARAWANG        | LELE       | 2019  | 759,058          | Rp9,108,696,000       |
| KOLAM AIR TENANG          | JAWA BARAT | KOTA BANDUNG    | LELE       | 2019  | 51,015           | Rp867,255,000         |

Kami menggunakan matriks korelasi untuk melihat kesinambungan data



### 2. Algoritma

- Framework <br />
Framework yang digunakan adalah TensorFlow.

- Pembangunan Model <br />
Model Machine Learning dibangun menggunakan model LSTM (Long Short Term Memory)
Masukkan kode training dan juga spesifikasi model, seperti epoch, learning rate, batch size, dan lain sebagainya.

- Model Evaluation <br />
Masukkan metrik evaluasi model seperti accuracy, precision, recall, F1-score, dan lain - lain.

## Prototype
Disesuaikan dengan kebutuhan atau bisa ditiru dari laporan dokumentasi massive.

## Deployment
- Teknologi Back End yang digunakan adalah Flask dan Docker 
- Deployment model menggunakan IBM Cloud

## Integration
Integrasi Model AI ke dalam web dengan menyambungkan API dari Model yang sudah di deploy di IBM Cloud ke dalam Back End Web.
Untuk aplikasi Web, silahkan pergi ke repository ini:
- https://github.com/praditus343/Nusaira
- https://github.com/praditus343/NusairaBE

## Result
https://nusaira.vercel.app/

## Conclusion
Disesuaikan dengan kebutuhan atau bisa ditiru dari laporan dokumentasi massive.
