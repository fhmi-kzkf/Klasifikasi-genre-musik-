# 🎵 Aplikasi Klasifikasi Genre Musik

Aplikasi web berbasis Streamlit untuk mengklasifikasikan genre musik dari file audio menggunakan machine learning. Aplikasi ini memungkinkan Anda mengunggah file audio (WAV atau MP3) dan memprediksi genre musiknya menggunakan model Support Vector Machine yang telah dilatih sebelumnya.

## 🚀 Fitur

- **Pemrosesan File Audio**: Unggah dan analisis file audio WAV atau MP3
- **Model SVM Terlatih**: Menggunakan model Support Vector Machine yang dilatih pada dataset musik
- **Ekstraksi Fitur**: Secara otomatis mengekstrak fitur audio (chroma, spectral centroid, bandwidth, dll.)
- **Prediksi Genre**: Memprediksi genre musik dengan skor kepercayaan
- **Visualisasi Audio**: Menampilkan visualisasi waveform dari audio yang diunggah
- **Analisis Real-time**: Prediksi genre instan dengan umpan balik visual
- **Perbandingan Algoritma**: Kemampuan untuk membandingkan berbagai algoritma klasifikasi
- **Upload Dataset**: Kemampuan untuk mengunggah dataset baru untuk pelatihan model

## 🎧 Genre yang Didukung

Aplikasi ini dapat mengklasifikasikan file audio ke dalam 10 genre musik yang berbeda:
- Blues
- Classical
- Country
- Disco
- Hip-Hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

## 📁 Persyaratan

- Python 3.7 atau lebih tinggi
- Streamlit
- Librosa (untuk pemrosesan audio)
- Scikit-learn
- Numpy
- Pandas
- Matplotlib
- Seaborn

## 🛠️ Instalasi

1. Clone atau unduh repositori ini
2. Instal paket yang diperlukan:

```bash
pip install -r requirements.txt
```

Atau instal paket secara individual:
```bash
pip install streamlit librosa scikit-learn pandas numpy matplotlib seaborn
```

## ▶️ Cara Menjalankan

```bash
streamlit run app.py
```

Kemudian buka browser Anda ke URL yang disediakan (biasanya http://localhost:8501)

## 🎯 Penggunaan

1. **Unggah File Audio**: Klik "Browse files" untuk mengunggah file audio WAV atau MP3
2. **Pemutaran Audio**: Dengarkan file audio yang diunggah langsung di aplikasi
3. **Analisis Otomatis**: Aplikasi akan secara otomatis mengekstrak fitur dan memprediksi genre
4. **Lihat Hasil**: Lihat genre yang diprediksi dan skor kepercayaan
5. **Probabilitas Detail**: Lihat distribusi probabilitas di semua genre
6. **Visualisasi Audio**: Lihat visualisasi waveform dari audio Anda

## 📊 Fitur Dataset

Dataset yang digunakan untuk melatih model berisi fitur-fitur audio berikut:
- Chroma STFT
- RMSE (Root Mean Square Energy)
- Spectral Centroid
- Spectral Bandwidth
- Rolloff
- Zero Crossing Rate
- MFCC (Mel-Frequency Cepstral Coefficients) 1-20

## 📈 Performa yang Diharapkan

Model SVM yang telah dilatih sebelumnya mencapai akurasi sekitar 68% pada set pengujian. Aplikasi akan menampilkan tingkat kepercayaan untuk setiap prediksi.

## 🧠 Detail Teknis

- **Model**: Support Vector Machine (SVM) dilatih pada dataset musik
- **Ekstraksi Fitur**: Chroma STFT, RMSE, Spectral Centroid, Spectral Bandwidth, Rolloff, Zero Crossing Rate, dan 20 MFCC
- **Pemrosesan Audio**: Librosa library untuk ekstraksi fitur
- **Preprocessing**: StandardScaler untuk normalisasi fitur
- **Handling Error**: Sistem fallback untuk menangani masalah kompatibilitas model dan scaler

## 📝 Catatan

- Aplikasi bekerja paling baik dengan klip audio 30 detik (durasi yang sama dengan data pelatihan)
- Untuk hasil terbaik, gunakan rekaman audio yang bersih tanpa noise latar belakang
- Model dilatih pada dataset musik, sehingga bekerja paling baik dengan gaya musik serupa
- File MP3 akan dikonversi ke format WAV untuk pemrosesan

## 🤝 Kontribusi

Jangan ragu untuk fork proyek ini dan mengirimkan pull request dengan perbaikan. Saran untuk peningkatan meliputi:
- Fitur audio tambahan
- Model ML yang lebih canggih
- Opsi visualisasi yang lebih baik
- Dukungan untuk file audio yang lebih panjang
- Implementasi algoritma klasifikasi tambahan

## 📄 Lisensi

Proyek ini bersifat open source dan tersedia di bawah Lisensi MIT.