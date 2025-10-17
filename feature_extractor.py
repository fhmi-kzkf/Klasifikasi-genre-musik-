# feature_extractor.py
import numpy as np
import librosa

def extract_features(file_path):
    """
    Mengekstrak fitur dari sebuah file audio.
    Fitur yang diekstrak HARUS SAMA PERSIS dengan yang digunakan saat training.
    """
    try:
        # Muat file audio, durasi 30 detik sesuai dataset GTZAN
        y, sr = librosa.load(file_path, mono=True, duration=30)
        
        # Ekstraksi Fitur (sesuaikan jika dataset Anda berbeda)
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        rmse = np.mean(librosa.feature.rms(y=y))
        spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Ekstrak 20 MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_means = np.mean(mfcc.T, axis=0)
        
        # Gabungkan semua fitur menjadi satu array NumPy
        # Urutannya harus sama persis dengan kolom di dataset training Anda
        features = np.array([
            chroma_stft, rmse, spec_cent, spec_bw, rolloff, zcr,
            *mfcc_means
        ])
        
        return features

    except Exception as e:
        print(f"Error saat memproses file audio: {e}")
        return None