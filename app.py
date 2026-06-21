import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Telemarketing Admin Dashboard", layout="wide")

# ==========================================
# 1. DEFINISI CUSTOM CLASS (Wajib ada)
# ==========================================
class GaussianNaiveBayesManual:
    """
    Implementasi Gaussian Naive Bayes dari nol (manual), tanpa menggunakan
    sklearn.naive_bayes atau library machine learning lain untuk bagian
    algoritma intinya.

    Tahapan algoritma:
    1. fit()  -> menghitung mean, varians, dan prior probability setiap kelas
                 dari data latih
    2. _gaussian_log_pdf() -> menghitung log-likelihood setiap fitur
                 menggunakan rumus fungsi densitas probabilitas (PDF) Gaussian
    3. predict_proba_log() -> menjumlahkan log(prior) + total log-likelihood
                 semua fitur, untuk setiap kelas
    4. predict() -> memilih kelas dengan nilai log-probabilitas tertinggi
                 (argmax), sesuai prinsip Maximum A Posteriori (MAP)
    """

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)

        self.classes_ = np.unique(y)
        self.mean_ = {}
        self.var_ = {}
        self.priors_ = {}

        n_samples = X.shape[0]

        for c in self.classes_:
            X_c = X[y == c]
            # Mean dan varians setiap fitur, dihitung khusus untuk kelas c
            self.mean_[c] = X_c.mean(axis=0)
            # Ditambahkan epsilon kecil (var smoothing) agar tidak terjadi
            # pembagian dengan nol jika ada fitur dengan varians = 0
            self.var_[c]  = X_c.var(axis=0) + 1e-9
            # Prior probability P(C) = proporsi jumlah sampel kelas c terhadap total data
            self.priors_[c] = X_c.shape[0] / n_samples

        return self

    def _gaussian_log_pdf(self, x, mean, var):
        """Menghitung log dari fungsi densitas probabilitas (PDF) Gaussian
        untuk setiap fitur, secara element-wise."""
        return -0.5 * np.log(2 * np.pi * var) - ((x - mean) ** 2) / (2 * var)

    def predict_proba_log(self, X):
        """Menghitung log-probabilitas posterior (belum dinormalisasi)
        untuk setiap kelas."""
        X = np.array(X, dtype=float)
        log_probs = np.zeros((X.shape[0], len(self.classes_)))

        for idx, c in enumerate(self.classes_):
            log_prior = np.log(self.priors_[c])
            # Asumsi independensi antar fitur (ciri khas Naive Bayes):
            # log-likelihood total = penjumlahan log-likelihood setiap fitur
            log_likelihood = self._gaussian_log_pdf(
                X, self.mean_[c], self.var_[c]
            ).sum(axis=1)
            log_probs[:, idx] = log_prior + log_likelihood

        return log_probs

    def predict(self, X):
        """Prediksi kelas akhir berdasarkan log-probabilitas tertinggi
        (Maximum A Posteriori / MAP)."""
        log_probs = self.predict_proba_log(X)
        return self.classes_[np.argmax(log_probs, axis=1)]

@st.cache_resource
def load_model_artifacts():
    # Pastikan file model_artifacts.pkl berada di folder yang sama
    with open('model_artifacts.pkl', 'rb') as f:
        return pickle.load(f)

try:
    artifacts = load_model_artifacts()
    model = artifacts['model']
    scaler = artifacts['scaler']
    selected_features = artifacts['features']
except FileNotFoundError:
    st.error("File 'model_artifacts.pkl' tidak ditemukan. Pastikan Anda sudah mengekspornya dari Colab.")
    st.stop()


@st.cache_data
def load_and_preprocess_all_data():
    # Membaca dataset hasil buatan rekan Anda
    try:
        df = pd.read_csv("dataset_dummy.csv")
    except FileNotFoundError:
        df = pd.read_csv("bank-full.csv") # fallback jika file tidak ditemukan
        
    # Ubah nama kolom 'nama' menjadi 'Nama Klien' agar tidak merusak kode UI Streamlit yang sudah ada
    if 'nama' in df.columns:
        df.rename(columns={'nama': 'Nama Klien'}, inplace=True)
    elif 'Nama Klien' not in df.columns:
        # Fallback darurat jika kolom nama benar-benar tidak ada
        df['Nama Klien'] = [f"Klien (ID: {i+1})" for i in range(len(df))]
    
    # --- PROSES BATCH INFERENCE ---
    X_batch = df.copy()
    
    # Preprocessing
    X_batch['job'] = X_batch['job'].replace('unknown', 'blue-collar')
    X_batch['education'] = X_batch['education'].replace('unknown', 'secondary')
    X_batch['pdays_contacted'] = (X_batch['pdays'] != -1).astype(int)
    X_batch['pdays'] = X_batch['pdays'].replace(-1, 0)
    
    for col in ['default', 'housing', 'loan']:
        X_batch[col] = X_batch[col].map({'no': 0, 'yes': 1})
        
    nominal_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
    X_encoded = pd.get_dummies(X_batch, columns=nominal_cols)
    
    # Reindex otomatis membuang kolom email & telpon yang tidak diperlukan oleh model
    X_final = X_encoded.reindex(columns=selected_features, fill_value=0)
    
    X_scaled = scaler.transform(X_final)
    
    # Hitung Probabilitas dengan Softmax & Temperature Scaling
    log_probs = model.predict_proba_log(X_scaled)
    temperature = 15.0 
    scaled_log_probs = log_probs / temperature
    
    max_log_probs = np.max(scaled_log_probs, axis=1, keepdims=True)
    exp_probs = np.exp(scaled_log_probs - max_log_probs)
    probs = exp_probs / np.sum(exp_probs, axis=1, keepdims=True)
    
    # Masukkan hasil prediksi & persentase ke DataFrame utama
    df['Peluang Sukses (%)'] = np.round(probs[:, 1] * 100, 2)
    df['Rekomendasi'] = df['Peluang Sukses (%)'].apply(lambda x: "🟢 TELEPON" if x >= 50.0 else "🔴 LEWATI")
    
    # Susun kolom agar info kontak (nama, email, telpon) dan prediksi berada di paling depan UI
    front_cols = ['Nama Klien']
    if 'email' in df.columns and 'telpon' in df.columns:
        front_cols.extend(['email', 'telpon'])
        
    front_cols.extend(['Peluang Sukses (%)', 'Rekomendasi', 'age', 'job', 'balance', 'duration'])
    
    # Sisa kolom ditaruh di belakang
    remaining_cols = [col for col in df.columns if col not in front_cols]
    return df[front_cols + remaining_cols]

df_master = load_and_preprocess_all_data()

# ==========================================
# 3. STRUKTUR LAYOUT & SIDEBAR
# ==========================================
st.sidebar.title("🎯 Telemarketing Filter")
st.sidebar.write("Gunakan panel ini untuk menyaring daftar target penelepon.")

# Komponen Filter di Sidebar
search_query = st.sidebar.text_input("🔍 Cari Nama Klien:", "")
filter_rekomendasi = st.sidebar.multiselect(
    "Filter Rekomendasi Model:", 
    options=["🟢 TELEPON", "🔴 LEWATI"], 
    default=["🟢 TELEPON", "🔴 LEWATI"]
)

# Terapkan Filter Data
df_filtered = df_master[df_master['Rekomendasi'].isin(filter_rekomendasi)]
if search_query:
    df_filtered = df_filtered[df_filtered['Nama Klien'].str.contains(search_query, case=False)]

# Ringkasan Statistik di Atas Dashboard
st.title("🏦 Dashboard Manajemen Klien - Telemarketing Deposito")
st.write("Sistem otomatisasi penentuan prioritas menelepon menggunakan Algoritma Custom Naive Bayes.")

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Total Klien di Database", f"{len(df_master):,}")
metric_col2.metric("Klien Siap Ditelepon (Peluang ≥ 50%)", f"{len(df_master[df_master['Peluang Sukses (%)'] >= 50.0]):,}")
metric_col3.metric("Data Terfilter saat ini", f"{len(df_filtered):,}")

st.divider()

# ==========================================
# 4. TAMPILAN UTAMA (TAB UI LIST)
# ==========================================
tab_table, tab_cards = st.tabs(["📊 Tampilan Tabel Ringkas (Batch)", "📇 Tampilan Kartu Detail (Card List)"])

# --- TAB 1: TAMPILAN TABEL DATA ---
with tab_table:
    st.write("### 📋 Daftar Seluruh Data Klien")
    st.write("Klik pada judul kolom untuk mengurutkan data (misal urutkan berdasarkan *Peluang Sukses (%)* tertinggi).")
    
    # Filter list kolom sesuai permintaan spesifik (Rekomendasi ditambahkan di sini)
    kolom_utama = ['Nama Klien', 'Peluang Sukses (%)', 'Rekomendasi', 'job', 'balance', 'email', 'telpon']
    
    # Pastikan program tidak error jika email/telpon belum ter-load sempurna
    kolom_tampil = [col for col in kolom_utama if col in df_filtered.columns]
    
    st.dataframe(
        df_filtered[kolom_tampil], 
        use_container_width=True,
        height=400
    )

# --- TAB 2: TAMPILAN KARTU DETAIL (CARD LIST UI) ---
# --- TAB 2: TAMPILAN KARTU DETAIL (CARD LIST UI) ---
with tab_cards:
    st.write("### 📇 Feed Kartu Profil Klien (Prioritas Tertinggi)")
    st.write("Daftar di bawah ini otomatis diurutkan dari nasabah dengan probabilitas berlangganan paling besar.")
    
    max_cards = 15
    # PERBAIKAN DI SINI: Mengurutkan data berdasarkan Peluang Sukses tertinggi (Descending)
    df_cards_sorted = df_filtered.sort_values(by="Peluang Sukses (%)", ascending=False)
    df_cards_display = df_cards_sorted.head(max_cards)
    
    if df_cards_display.empty:
        st.info("Tidak ada data klien yang cocok dengan filter.")
    else:
        for idx, row in df_cards_display.iterrows():
            with st.container(border=True):
                c_name, c_metrics, c_action = st.columns([2, 4, 2])
                
                with c_name:
                    st.write(f"#### {row['Nama Klien']}")
                    st.caption(f"Pekerjaan: {str(row['job']).title()}")
                
                with c_metrics:
                    # Menampilkan Saldo, Peluang, Email, dan Telepon
                    sub_col1, sub_col2, sub_col3 = st.columns(3)
                    sub_col1.metric("Saldo Klien", f"€{row['balance']:,}")
                    sub_col2.metric("Peluang Sukses", f"{row['Peluang Sukses (%)']}%")
                    
                    with sub_col3:
                        st.markdown("**Info Kontak:**")
                        st.caption(f"📧 {row.get('email', 'Tidak ada data')}")
                        st.caption(f"📞 {row.get('telpon', 'Tidak ada data')}")
                
                with c_action:
                    st.write("Status Analisis:")
                    if row['Peluang Sukses (%)'] >= 50.0:
                        st.success("🟢 TELEPON")
                    else:
                        st.error("🔴 LEWATI")