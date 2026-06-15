
import streamlit as st
import pickle
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ─────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────
st.set_page_config(
    page_title="Analisis Sentimen Shopee vs Tokopedia",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────
@st.cache_resource
def load_model():
    with open("svm_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    return model, tfidf

@st.cache_data
def load_data():
    comparative_data = pd.read_csv("comparative_data.csv", index_col=0)
    comparative_pct  = pd.read_csv("comparative_pct.csv",  index_col=0)
    sample_data      = pd.read_csv("sample_data.csv")
    with open("model_metadata.json") as f:
        metadata = json.load(f)
    with open("classification_report.json") as f:
        report   = json.load(f)
    with open("slang_dict.json") as f:
        slang_dict = json.load(f)
    return comparative_data, comparative_pct, sample_data, metadata, report, slang_dict

model, tfidf = load_model()
comparative_data, comparative_pct, sample_data, metadata, report, slang_dict = load_data()

# ─────────────────────────────────────
# PREPROCESSING FUNCTION
# ─────────────────────────────────────
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    factory_stem  = StemmerFactory()
    stemmer       = factory_stem.create_stemmer()
    factory_stop  = StopWordRemoverFactory()
    stopword      = factory_stop.create_stop_word_remover()
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    text  = " ".join([slang_dict.get(w, w) for w in words])
    if SASTRAWI_AVAILABLE:
        text = stopword.remove(text)
        text = stemmer.stem(text)
    return text

def predict_sentiment(text):
    clean  = preprocess_text(text)
    vector = tfidf.transform([clean])
    result = model.predict(vector)[0]
    return result, clean, None


# ─────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shopping-cart.png", width=80)
    st.title(" Navigasi")
    page = st.radio(
        "Pilih Halaman:",
        [" Dashboard", " Prediksi Sentimen", " Analisis Komparatif",
         " Performa Model", " Dataset Preview"]
    )
    
# ─────────────────────────────────────
# HALAMAN: DASHBOARD
# ─────────────────────────────────────
if page == " Dashboard":
    st.title(" Analisis Sentimen: Shopee vs Tokopedia")
    st.markdown("Analisis sentimen ulasan pengguna menggunakan **SVM + TF-IDF**")

    col1, col2, col3, col4 = st.columns(4)
    total = comparative_data.sum().sum()
    col1.metric("Total Ulasan",    f"{int(total):,}")
    col2.metric("Akurasi Test",    f"{metadata['test_accuracy']*100:.2f}%")
    col3.metric("Akurasi Val",     f"{metadata['val_accuracy']*100:.2f}%")
    

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribusi Kelas (Keseluruhan)")
        overall = comparative_data.sum()
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        bars = ax.bar(["Positif", "Netral", "Negatif"],
                      [overall.get('Positif',0), overall.get('Netral',0), overall.get('Negatif',0)],
                      color=colors, edgecolor='white')
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
                    f'{int(bar.get_height()):,}', ha='center', fontweight='bold')
        ax.set_ylabel('Jumlah')
        ax.grid(axis='y', alpha=0.4)
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.subheader("Distribusi per Platform")
        fig, ax = plt.subplots(figsize=(7, 4))
        comparative_pct.plot(kind='bar', ax=ax,
                             color=['#2ecc71', '#f39c12', '#e74c3c'],
                             edgecolor='white')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_ylabel('Persentase (%)'); ax.legend(title='Sentimen')
        ax.grid(axis='y', alpha=0.4)
        st.pyplot(fig, use_container_width=True)

# ─────────────────────────────────────
# HALAMAN: PREDIKSI SENTIMEN
# ─────────────────────────────────────
elif page == " Prediksi Sentimen":
    st.title(" Prediksi Sentimen Ulasan")
    st.markdown("Masukkan ulasan produk, model akan memprediksi sentimennya.")

    platform_sel = st.selectbox("Platform:", ["Shopee", "Tokopedia", "Lainnya"])
    user_input   = st.text_area("Tulis ulasan di sini:",
                                placeholder="Contoh: Barang bagus, pengiriman cepat, seller ramah...",
                                height=150)

    if st.button(" Prediksi", type="primary", use_container_width=True):
        if user_input.strip():
            sentiment, clean_text, proba = predict_sentiment(user_input)
            emoji_map = {'Positif': '', 'Netral': '', 'Negatif': ''}
            color_map = {'Positif': 'green',   'Netral': 'orange', 'Negatif': 'red'}
            st.markdown(f"### Hasil: :{color_map[sentiment]}[{emoji_map[sentiment]} {sentiment}]")
            with st.expander("Detail Preprocessing"):
                col1, col2 = st.columns(2)
                col1.markdown(f"**Teks Asli:**\n> {user_input}")
                col2.markdown(f"**Teks Bersih:**\n> {clean_text}")
        else:
            st.warning("Harap masukkan teks ulasan terlebih dahulu.")

# ─────────────────────────────────────
# HALAMAN: ANALISIS KOMPARATIF
# ─────────────────────────────────────
elif page == " Analisis Komparatif":
    st.title(" Analisis Komparatif: Shopee vs Tokopedia")
    st.dataframe(comparative_data.style.background_gradient(cmap='RdYlGn'), use_container_width=True)
    st.markdown("---")
    st.subheader("Distribusi Sentimen (%)")
    st.dataframe(comparative_pct.style.background_gradient(cmap='RdYlGn'), use_container_width=True)

# ─────────────────────────────────────
# HALAMAN: PERFORMA MODEL
# ─────────────────────────────────────
elif page == " Performa Model":
    st.title(" Performa Model SVM")

    col1, col2 = st.columns(2)
    col1.metric("Akurasi Validation", f"{metadata['val_accuracy']*100:.2f}%")
    col2.metric("Akurasi Test",       f"{metadata['test_accuracy']*100:.2f}%")

    st.subheader("Classification Report (Test Set)")
    report_df = pd.DataFrame(report).T.drop(columns=['support'], errors='ignore')
    st.dataframe(report_df.style.format("{:.3f}").background_gradient(cmap='Blues'),
                 use_container_width=True)

    st.subheader("Gambar dari Notebook")
    import os
    images = {'Confusion Matrix': 'confusion_matrix.png',
              'Learning Curve': 'learning_curve.png'}
    for title, fname in images.items():
        if os.path.exists(fname):
            st.markdown(f"**{title}**")
            st.image(fname, use_column_width=True)

# ─────────────────────────────────────
# HALAMAN: DATASET PREVIEW
# ─────────────────────────────────────
elif page == " Dataset Preview":
    st.title(" Preview Dataset")
    filter_platform  = st.selectbox("Filter Platform:",  ["Semua"] + list(sample_data['platform'].unique()))
    filter_sentiment = st.selectbox("Filter Sentimen:", ["Semua"] + ["Positif", "Netral", "Negatif"])

    filtered = sample_data.copy()
    if filter_platform  != "Semua": filtered = filtered[filtered['platform']  == filter_platform]
    if filter_sentiment != "Semua": filtered = filtered[filtered['predicted_sentiment'] == filter_sentiment]

    st.markdown(f"Menampilkan **{len(filtered):,}** baris")
    st.dataframe(filtered, use_container_width=True, height=400)
