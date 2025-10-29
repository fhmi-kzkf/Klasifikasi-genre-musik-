import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pickle
import joblib
import tempfile
import os
from feature_extractor import extract_features

# Import librosa only when needed to avoid import issues
def load_librosa():
    try:
        import librosa
        return librosa
    except ImportError as e:
        st.error(f"Error importing librosa: {e}")
        return None

# Judul aplikasi
st.set_page_config(page_title="Aplikasi Klasifikasi Genre Musik", layout="wide")
st.title("🎵 Aplikasi Klasifikasi Genre Musik Berbasis Fitur Audio")
st.markdown("Unggah file audio (WAV atau MP3) untuk mengklasifikasikan genre musiknya")

# Daftar genre musik (disimpan sebagai referensi, tapi kita akan pakai encoder)
GENRES = ['blues', 'classical', 'country', 'disco', 'hip-hop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Mengubah nama fungsi dan menambahkan 'encoder'
@st.cache_resource(show_spinner="Memuat semua file model...")
def load_all_files():
    model = None
    scaler = None
    encoder = None # Tambahkan encoder
    
    try:
        # Mengganti nama file model ke model TERBAIK kita
        model_file = 'model_voting_terbaik.pkl' # Menggunakan model Voting 72.0%
        scaler_file = 'scaler.pkl'
        encoder_file = 'label_encoder.pkl' # Nama file encoder
        
        # Try loading the model with multiple approaches
        try:
            model = joblib.load(model_file)
        except Exception as joblib_error:
            st.warning(f"Joblib failed for model: {joblib_error}. Trying pickle...")
            try:
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
            except Exception as pickle_error:
                st.warning(f"Pickle failed for model: {pickle_error}. Trying pickle with fix_imports...")
                try:
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f, fix_imports=True)
                except Exception as fix_imports_error:
                    st.warning(f"Pickle with fix_imports failed for model: {fix_imports_error}. Trying pickle with latin1...")
                    try:
                        with open(model_file, 'rb') as f:
                            model = pickle.load(f, encoding='latin1')
                    except Exception as latin1_error:
                        st.error(f"All methods failed for model: {latin1_error}")
                        return None, None, None
        
        # Try loading the scaler with multiple approaches
        try:
            scaler = joblib.load(scaler_file)
        except Exception as joblib_error:
            st.warning(f"Joblib failed for scaler: {joblib_error}. Trying pickle...")
            try:
                with open(scaler_file, 'rb') as f:
                    scaler = pickle.load(f)
            except Exception as pickle_error:
                st.warning(f"Pickle failed for scaler: {pickle_error}. Trying pickle with fix_imports...")
                try:
                    with open(scaler_file, 'rb') as f:
                        scaler = pickle.load(f, fix_imports=True)
                except Exception as fix_imports_error:
                    st.warning(f"Pickle with fix_imports failed for scaler: {fix_imports_error}. Trying pickle with latin1...")
                    try:
                        with open(scaler_file, 'rb') as f:
                            scaler = pickle.load(f, encoding='latin1')
                    except Exception as latin1_error:
                        try:
                            scaler = create_new_scaler()
                            st.info("Created new scaler from dataset")
                        except Exception as create_error:
                            st.error(f"Gagal membuat scaler baru: {create_error}")
                            return model, None, None
        
        # Try loading the encoder with multiple approaches
        try:
            encoder = joblib.load(encoder_file)
        except Exception as joblib_error:
            st.warning(f"Joblib failed for encoder: {joblib_error}. Trying pickle...")
            try:
                with open(encoder_file, 'rb') as f:
                    encoder = pickle.load(f)
            except Exception as pickle_error:
                st.warning(f"Pickle failed for encoder: {pickle_error}. Trying pickle with fix_imports...")
                try:
                    with open(encoder_file, 'rb') as f:
                        encoder = pickle.load(f, fix_imports=True)
                except Exception as fix_imports_error:
                    st.warning(f"Pickle with fix_imports failed for encoder: {fix_imports_error}. Trying pickle with latin1...")
                    try:
                        with open(encoder_file, 'rb') as f:
                            encoder = pickle.load(f, encoding='latin1')
                    except Exception as latin1_error:
                        st.error(f"Gagal memuat file label encoder: {latin1_error}")
                        return model, scaler, None 

        # Mengembalikan 3 file
        return model, scaler, encoder
        
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat file: {e}")
        return None, None, None

# Fungsi untuk membuat scaler baru dari dataset (Tidak diubah)
def create_new_scaler():
    try:
        df = pd.read_csv('dataset.csv')
        features = df.drop(['filename', 'label'], axis=1)
        new_scaler = StandardScaler()
        new_scaler.fit(features)
        joblib.dump(new_scaler, 'scaler_new.pkl')
        return new_scaler
    except Exception as e:
        st.error(f"Error creating new scaler: {e}")
        raise e

# Memuat 3 file
model, scaler, encoder = load_all_files()

# Fungsi untuk memprediksi genre (Tidak diubah, karena 'prediction' adalah angka)
def predict_genre(features):
    if model is None or scaler is None:
        return None, None
    
    try:
        features = features.reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0] # Ini adalah angka (int)
        
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features_scaled)[0]
        else:
            probabilities = model.decision_function(features_scaled)[0]
            probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities))
            
        return prediction, probabilities
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# Sidebar
st.sidebar.header("ℹ️ Tentang Aplikasi")
st.sidebar.markdown("""
Aplikasi ini mengklasifikasikan genre musik dari file audio menggunakan model **Voting Classifier (Akurasi 72.0%)**.
""")
st.sidebar.markdown("**Genre yang didukung:**")
if encoder is not None:
    for g in encoder.classes_:
        st.sidebar.markdown(f"- {g.title()}")
else:
    for g in GENRES:
        st.sidebar.markdown(f"- {g.title()}")

st.sidebar.markdown("""
**Cara menggunakan:**
1. Unggah file audio (WAV/MP3)
2. Dengarkan file yang diunggah
3. Lihat hasil prediksi genre
4. Analisis distribusi probabilitas
""")

# File uploader
uploaded_file = st.file_uploader("📁 Pilih file audio (WAV atau MP3)", type=["wav", "mp3"])

if uploaded_file is not None:
    # Cek apakah model berhasil dimuat
    if model is None or scaler is None or encoder is None:
        st.error("Model/Scaler/Encoder gagal dimuat. Aplikasi tidak dapat melanjutkan.")
    else:
        # Simpan file sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Tampilkan info dan audio player
            st.subheader("📄 Informasi File")
            st.info(f"File diunggah: {uploaded_file.name}")
            st.subheader("▶️ Dengarkan Audio")
            st.audio(uploaded_file, format='audio/wav')
            
            with st.spinner("🔍 Mengekstrak fitur audio..."):
                features = extract_features(tmp_file_path)
                
            if features is not None:
                with st.spinner("🧠 Memprediksi genre musik..."):
                    prediction, probabilities = predict_genre(features)
                    
                if prediction is not None and probabilities is not None:
                    st.subheader("🎯 Hasil Prediksi")
                    
                    # Menerjemahkan prediksi angka ke teks
                    prediction_text = encoder.inverse_transform([prediction])[0]
                    formatted_genre = str(prediction_text).title()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"**Genre yang Diprediksi: {formatted_genre}**")
                    
                    with col2:
                        confidence = float(np.max(probabilities)) * 100
                        st.info(f"**Tingkat Kepercayaan: {confidence:.2f}%**")
                    
                    if confidence < 50:
                        st.warning("⚠️ Tingkat kepercayaan prediksi rendah. Hasil mungkin tidak akurat.")
                    
                    st.subheader("📊 Distribusi Probabilitas Genre")
                    
                    # Menggunakan 'encoder.classes_' untuk label plot yang AKURAT
                    prob_df = pd.DataFrame({
                        'Genre': [g.title() for g in encoder.classes_], # Menggunakan encoder
                        'Probabilitas': probabilities
                    }).sort_values('Probabilitas', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(data=prob_df, x='Probabilitas', y='Genre', palette='viridis', ax=ax)
                    ax.set_xlabel('Probabilitas')
                    ax.set_title('Distribusi Probabilitas Genre Musik')
                    st.pyplot(fig)
                    
                    # Tampilkan tabel probabilitas
                    st.subheader("📋 Detail Probabilitas")
                    st.dataframe(prob_df.style.format({'Probabilitas': '{:.4f}'}))
                    
                    # --- BLOK VISUALISASI WAVEFORM SUDAH DIHAPUS DARI SINI ---  
                else:
                    st.error("Gagal melakukan prediksi. Silakan coba file lain.")
            else:
                st.error("Gagal mengekstrak fitur dari file audio. Pastikan file tidak rusak dan formatnya didukung.")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
            
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
else:
    # Tampilkan pesan jika file gagal dimuat di awal
    if not (model and scaler and encoder):
        st.error("Aplikasi gagal dimuat karena file model/scaler/encoder tidak ditemukan.")
    else:
        st.info("👆 Silakan unggah file audio (WAV atau MP3) untuk memulai klasifikasi genre musik.")
    
# Footer
st.markdown("---")
st.markdown("🎵 Aplikasi Klasifikasi Genre Musik | Dibuat dengan Streamlit")