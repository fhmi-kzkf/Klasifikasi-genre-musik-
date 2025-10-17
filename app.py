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

# Daftar genre musik
GENRES = ['blues', 'classical', 'country', 'disco', 'hip-hop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Load model SVM and scaler with error handling
@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    model = None
    scaler = None
    
    try:
        # Try loading the model with joblib first
        try:
            model = joblib.load('model_svm_terbaik.pkl')
            # st.info("Model loaded successfully with joblib")
        except Exception as joblib_error:
            # st.info(f"Joblib failed: {str(joblib_error)[:50]}...")
            # Fallback to pickle with compatibility settings
            try:
                with open('model_svm_terbaik.pkl', 'rb') as f:
                    # Try with fix_imports=True for compatibility
                    model = pickle.load(f, fix_imports=True)
                # st.info("Model loaded successfully with pickle (fix_imports=True)")
            except Exception as pickle_error:
                # st.info(f"Pickle failed: {str(pickle_error)[:50]}...")
                # Try with encoding='latin1' for older pickle files
                try:
                    with open('model_svm_terbaik.pkl', 'rb') as f:
                        model = pickle.load(f, encoding='latin1')
                    # st.info("Model loaded successfully with pickle (latin1 encoding)")
                except Exception as latin1_error:
                    st.error(f"Gagal memuat model: {latin1_error}")
                    return None, None
        
        # Try loading the scaler with joblib first
        try:
            scaler = joblib.load('scaler.pkl')
            # st.info("Scaler loaded successfully with joblib")
        except Exception as joblib_error:
            # st.info(f"Scaler joblib failed: {str(joblib_error)[:50]}...")
            # Fallback to pickle with compatibility settings
            try:
                with open('scaler.pkl', 'rb') as f:
                    # Try with fix_imports=True for compatibility
                    scaler = pickle.load(f, fix_imports=True)
                # st.info("Scaler loaded successfully with pickle (fix_imports=True)")
            except Exception as pickle_error:
                # st.info(f"Scaler pickle failed: {str(pickle_error)[:50]}...")
                # Try with encoding='latin1' for older pickle files
                try:
                    with open('scaler.pkl', 'rb') as f:
                        scaler = pickle.load(f, encoding='latin1')
                    # st.info("Scaler loaded successfully with pickle (latin1 encoding)")
                except Exception as latin1_error:
                    # st.info(f"All scaler loading methods failed: {latin1_error}")
                    # Jika semua metode gagal, buat scaler baru dari dataset
                    try:
                        # st.info("Attempting to create new scaler from dataset...")
                        scaler = create_new_scaler()
                        # st.info("Successfully created new scaler from dataset")
                    except Exception as create_error:
                        st.error(f"Gagal membuat scaler baru: {create_error}")
                        return model, None
            
        return model, scaler
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model atau scaler: {e}")
        return None, None

# Fungsi untuk membuat scaler baru dari dataset
def create_new_scaler():
    try:
        # Baca dataset
        df = pd.read_csv('dataset.csv')
        
        # Ambil fitur (semua kolom kecuali 'filename' dan 'label')
        features = df.drop(['filename', 'label'], axis=1)
        
        # Buat dan latih scaler baru
        new_scaler = StandardScaler()
        new_scaler.fit(features)
        
        # Simpan scaler baru
        joblib.dump(new_scaler, 'scaler_new.pkl')
        
        return new_scaler
    except Exception as e:
        st.error(f"Error creating new scaler: {e}")
        raise e

model, scaler = load_model_and_scaler()

# Fungsi untuk memprediksi genre
def predict_genre(features):
    if model is None or scaler is None:
        return None, None
    
    try:
        # Reshape features untuk prediksi
        features = features.reshape(1, -1)
        
        # Normalisasi fitur menggunakan scaler
        features_scaled = scaler.transform(features)
        
        # Prediksi genre
        prediction = model.predict(features_scaled)[0]
        
        # Dapatkan probabilitas prediksi
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features_scaled)[0]
        else:
            # Jika model tidak memiliki predict_proba, gunakan decision function
            probabilities = model.decision_function(features_scaled)[0]
            # Normalisasi ke probabilitas
            probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities))
        
        return prediction, probabilities
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# Sidebar
st.sidebar.header("ℹ️ Tentang Aplikasi")
st.sidebar.markdown("""
Aplikasi ini mengklasifikasikan genre musik dari file audio menggunakan machine learning.

**Genre yang didukung:**
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
""")

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
    # Simpan file sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        # Tampilkan informasi file
        st.subheader("📄 Informasi File")
        st.info(f"File diunggah: {uploaded_file.name}")
        
        # Tampilkan audio player
        st.subheader("▶️ Dengarkan Audio")
        st.audio(uploaded_file, format='audio/wav')
        
        # Ekstrak fitur
        with st.spinner("🔍 Mengekstrak fitur audio..."):
            features = extract_features(tmp_file_path)
            
        if features is not None:
            # Prediksi genre
            with st.spinner("🧠 Memprediksi genre musik..."):
                prediction, probabilities = predict_genre(features)
                
            if prediction is not None and probabilities is not None:
                # Tampilkan hasil prediksi
                st.subheader("🎯 Hasil Prediksi")
                
                # Format nama genre (kapitalisasi)
                # Konversi prediction ke string terlebih dahulu untuk menghindari error numpy.int64
                formatted_genre = str(prediction).title()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**Genre yang Diprediksi: {formatted_genre}**")
                
                with col2:
                    # Hitung confidence score
                    confidence = float(np.max(probabilities)) * 100
                    st.info(f"**Tingkat Kepercayaan: {confidence:.2f}%**")
                
                # Tampilkan peringatan jika confidence rendah
                if confidence < 50:
                    st.warning("⚠️ Tingkat kepercayaan prediksi rendah. Hasil mungkin tidak akurat.")
                
                # Visualisasi distribusi probabilitas
                st.subheader("📊 Distribusi Probabilitas Genre")
                
                # Buat dataframe untuk visualisasi
                prob_df = pd.DataFrame({
                    'Genre': [g.title() for g in GENRES],
                    'Probabilitas': probabilities
                }).sort_values('Probabilitas', ascending=False)
                
                # Plot distribusi probabilitas
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=prob_df, x='Probabilitas', y='Genre', palette='viridis', ax=ax)
                ax.set_xlabel('Probabilitas')
                ax.set_title('Distribusi Probabilitas Genre Musik')
                st.pyplot(fig)
                
                # Tampilkan tabel probabilitas
                st.subheader("📋 Detail Probabilitas")
                st.dataframe(prob_df.style.format({'Probabilitas': '{:.4f}'}))
                
                # Visualisasi waveform
                st.subheader("🎼 Visualisasi Gelombang Audio")
                try:
                    # Load audio untuk visualisasi
                    librosa = load_librosa()
                    if librosa is not None:
                        y, sr = librosa.load(tmp_file_path, duration=30)
                        fig, ax = plt.subplots(figsize=(12, 4))
                        librosa.display.waveshow(y, sr=sr, ax=ax)
                        ax.set_title('Waveform Audio')
                        ax.set_xlabel('Waktu (detik)')
                        ax.set_ylabel('Amplitudo')
                        st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Tidak dapat menampilkan visualisasi waveform: {e}")
            else:
                st.error("Gagal melakukan prediksi. Silakan coba file lain.")
        else:
            st.error("Gagal mengekstrak fitur dari file audio. Pastikan file tidak rusak dan formatnya didukung.")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        
    finally:
        # Hapus file sementara
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
else:
    st.info("👆 Silakan unggah file audio (WAV atau MP3) untuk memulai klasifikasi genre musik.")
    
# Footer
st.markdown("---")
st.markdown("🎵 Aplikasi Klasifikasi Genre Musik | Dibuat dengan Streamlit")