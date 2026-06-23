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
    # Pastikan file best_naive_bayes_model.pkl berada di folder yang sama
    with open('best_naive_bayes_model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    artifacts = load_model_artifacts()
    model = artifacts['model']
    scaler = artifacts['scaler']
    selected_features = artifacts['features']
except FileNotFoundError:
    st.error("File 'best_naive_bayes_model.pkl' tidak ditemukan. Pastikan Anda sudah mengekspornya dari Colab.")
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
# NAMA TAB SUDAH DISESUAIKAN DENGAN ISTILAH DOSEN
tab_input, tab_table, tab_cards, tab_evaluasi = st.tabs([
    "📝 Input Data Baru (Tanpa Label)", 
    "📊 Tampilan Tabel Ringkas", 
    "📇 Tampilan Kartu Detail",
    "📈 Validasi Model (Data Berlabel)"
])

# --- TAB 1: FORM INPUT & UPLOAD DATA BARU ---
with tab_input:
    st.write("### 📝 Evaluasi Nasabah Baru")
    
    # Berikan opsi kepada user: mau upload banyak sekaligus, atau input satu per satu
    metode_input = st.radio(
        "Pilih Metode Evaluasi:", 
        ["📂 Upload Data Massal (CSV)", "✍️ Input Manual (Satu per Satu)"], 
        horizontal=True
    )
    
    st.divider()

    # =======================================================
    # OPSI 1: UPLOAD DATA MASSAL (CSV)
    # =======================================================
    if metode_input == "📂 Upload Data Massal (CSV)":
        st.info("💡 Unggah file `.csv` berisi daftar nasabah (tanpa target/keputusan). Sistem akan memprediksi peluang berlangganan untuk seluruh nasabah dalam hitungan detik.")
        
        uploaded_file = st.file_uploader("Pilih file CSV Nasabah Baru", type=["csv"])
        
        if uploaded_file is not None:
            df_new_batch = pd.read_csv(uploaded_file)
            st.success(f"Berhasil memuat {len(df_new_batch)} baris data nasabah baru.")
            
            # Tombol untuk mengeksekusi prediksi massal
            if st.button("🚀 Prediksi Seluruh Nasabah", use_container_width=True):
                with st.spinner('Sedang menghitung probabilitas menggunakan Naive Bayes...'):
                    # 1. Preprocessing Massal
                    X_prep = df_new_batch.copy()
                    
                    if 'job' in X_prep.columns: X_prep['job'] = X_prep['job'].replace('unknown', 'blue-collar')
                    if 'education' in X_prep.columns: X_prep['education'] = X_prep['education'].replace('unknown', 'secondary')
                    
                    if 'pdays' in X_prep.columns:
                        X_prep['pdays_contacted'] = (X_prep['pdays'] != -1).astype(int)
                        X_prep['pdays'] = X_prep['pdays'].replace(-1, 0)
                    
                    for col in ['default', 'housing', 'loan']:
                        if col in X_prep.columns:
                            X_prep[col] = X_prep[col].map({'no': 0, 'yes': 1, 0: 0, 1: 1})
                            
                    nominal_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
                    existing_nominal = [c for c in nominal_cols if c in X_prep.columns]
                    df_encoded = pd.get_dummies(X_prep, columns=existing_nominal)
                    
                    X_final = df_encoded.reindex(columns=selected_features, fill_value=0)
                    
                    # 2. Scaling & Prediksi Massal
                    X_scaled = scaler.transform(X_final)
                    
                    log_probs = model.predict_proba_log(X_scaled)
                    temperature = 15.0 
                    scaled_log_probs = log_probs / temperature
                    max_log_probs = np.max(scaled_log_probs, axis=1, keepdims=True)
                    exp_probs = np.exp(scaled_log_probs - max_log_probs)
                    probs = exp_probs / np.sum(exp_probs, axis=1, keepdims=True)
                    
                    peluang_sukses_batch = np.round(probs[:, 1] * 100, 2)
                    
                    # 3. Gabungkan Hasil Prediksi ke DataFrame Asli
                    df_result = df_new_batch.copy()
                    
                    # Sisipkan kolom hasil di paling depan agar mudah dilihat
                    rekomendasi_batch = ["🟢 TELEPON" if p >= 50.0 else "🔴 LEWATI" for p in peluang_sukses_batch]
                    df_result.insert(0, 'Peluang Sukses (%)', peluang_sukses_batch)
                    df_result.insert(1, 'Rekomendasi', rekomendasi_batch)
                    
                    # Urutkan dari peluang tertinggi ke terendah
                    df_result = df_result.sort_values(by="Peluang Sukses (%)", ascending=False).reset_index(drop=True)
                    
                    st.write("### 📊 Hasil Prediksi Target Prioritas")
                    st.dataframe(df_result, use_container_width=True)
                    
                    # 4. Tombol Download untuk diekspor kembali ke Excel oleh Sales
                    csv_output = df_result.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download Hasil Prediksi (CSV)",
                        data=csv_output,
                        file_name="rekomendasi_telemarketing_nasabah.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

    # =======================================================
    # OPSI 2: INPUT MANUAL (SATU PER SATU)
    # =======================================================
    else:
        st.write("Masukkan data demografi dan riwayat nasabah secara manual untuk memprediksi probabilitas berlangganan.")
        with st.form("form_prediksi_baru"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🧑‍💼 Demografi**")
                age = st.number_input("Umur", min_value=18, max_value=100, value=30)
                job = st.selectbox("Pekerjaan", ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"])
                marital = st.selectbox("Status Pernikahan", ["divorced", "married", "single"])
                education = st.selectbox("Pendidikan", ["primary", "secondary", "tertiary", "unknown"])
                
            with col2:
                st.markdown("**💰 Finansial**")
                balance = st.number_input("Saldo (Balance) €", value=1500)
                housing = st.selectbox("KPR / Pinjaman Rumah", ["no", "yes"])
                loan = st.selectbox("Pinjaman Pribadi", ["no", "yes"])
                default = st.selectbox("Riwayat Kredit Macet", ["no", "yes"])
                
            with col3:
                st.markdown("**📞 Riwayat Kampanye**")
                campaign = st.number_input("Jumlah Kontak Kampanye Ini", min_value=1, value=1)
                pdays = st.number_input("Hari sejak kontak terakhir (-1 = belum pernah)", min_value=-1, value=-1)
                poutcome = st.selectbox("Hasil Kampanye Sebelumnya", ["unknown", "other", "failure", "success"])
                month = st.selectbox("Bulan Kontak", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
                contact = st.selectbox("Metode Kontak", ["cellular", "telephone", "unknown"])

            duration = 0 
            previous = 0 if pdays == -1 else 1
            
            submit_button = st.form_submit_button("🎯 Prediksi Keputusan", use_container_width=True)

        if submit_button:
            input_data = pd.DataFrame([{
                'age': age, 'job': job, 'marital': marital, 'education': education,
                'default': default, 'balance': balance, 'housing': housing, 'loan': loan,
                'contact': contact, 'day': 15, 'month': month, 'duration': duration,
                'campaign': campaign, 'pdays': pdays, 'previous': previous, 'poutcome': poutcome
            }])
            
            df_prep = input_data.copy()
            df_prep['job'] = df_prep['job'].replace('unknown', 'blue-collar')
            df_prep['education'] = df_prep['education'].replace('unknown', 'secondary')
            df_prep['pdays_contacted'] = (df_prep['pdays'] != -1).astype(int)
            df_prep['pdays'] = df_prep['pdays'].replace(-1, 0)
            
            for col in ['default', 'housing', 'loan']:
                df_prep[col] = df_prep[col].map({'no': 0, 'yes': 1})
                
            nominal_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
            df_encoded = pd.get_dummies(df_prep, columns=nominal_cols)
            
            X_final = df_encoded.reindex(columns=selected_features, fill_value=0)
            X_scaled = scaler.transform(X_final)
            
            log_probs = model.predict_proba_log(X_scaled)
            temperature = 15.0 
            scaled_log_probs = log_probs / temperature
            
            max_log_probs = np.max(scaled_log_probs, axis=1, keepdims=True)
            exp_probs = np.exp(scaled_log_probs - max_log_probs)
            probs = exp_probs / np.sum(exp_probs, axis=1, keepdims=True)
            
            peluang_sukses = np.round(probs[0, 1] * 100, 2)
            
            st.write("---")
            st.subheader("📊 Hasil Prediksi Sistem")
            
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.metric("Probabilitas Berlangganan (YES)", f"{peluang_sukses}%")
                
            with res_col2:
                if peluang_sukses >= 50.0:
                    st.success("### 🟢 REKOMENDASI: TELEPON\nNasabah ini memiliki profil yang sangat mirip dengan klien-klien kita yang sebelumnya mengambil deposito.")
                else:
                    st.error("### 🔴 REKOMENDASI: LEWATI\nNasabah ini memiliki probabilitas rendah. Fokuskan waktu *telemarketer* pada nasabah lain yang lebih potensial.")

# --- TAB 2: TAMPILAN TABEL DATA ---
with tab_table:
    st.write("### 📋 Daftar Seluruh Data Klien")
    kolom_utama = ['Nama Klien', 'Peluang Sukses (%)', 'Rekomendasi', 'job', 'balance', 'email', 'telpon']
    kolom_tampil = [col for col in kolom_utama if col in df_filtered.columns]
    st.dataframe(df_filtered[kolom_tampil], use_container_width=True, height=400)

# --- TAB 3: TAMPILAN KARTU DETAIL (CARD LIST UI) ---
with tab_cards:
    st.write("### 📇 Feed Kartu Profil Klien (Prioritas Tertinggi)")
    st.write("Daftar di bawah ini otomatis diurutkan dari nasabah dengan probabilitas berlangganan paling besar.")
    
    max_cards = 15
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

# --- TAB 4: EVALUASI PERFORMA DATA BARU ---
with tab_evaluasi:
    st.write("### 📈 Uji Performa Model terhadap Dataset Baru")
    st.write("Unggah file data evaluasi yang memiliki kolom target aktual (`y`) untuk melihat perubahan akurasi model.")

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    uploaded_eval_file = st.file_uploader("Pilih file CSV Evaluasi (contoh: data_baru_evaluasi_bersih.csv):", type=["csv"], key="eval_uploader")

    if uploaded_eval_file is not None:
        df_new = pd.read_csv(uploaded_eval_file)

        if 'y' not in df_new.columns:
            st.error("Gagal! File harus memiliki kolom target 'y'.")
        else:
            st.success(f"Berhasil memuat {len(df_new)} baris data evaluasi.")

            # Preprocessing
            X_eval = df_new.copy()
            y_actual = X_eval['y'].map({'no': 0, 'yes': 1}).values
            X_eval = X_eval.drop(columns=['y'])

            X_eval['job'] = X_eval['job'].replace('unknown', 'blue-collar')
            X_eval['education'] = X_eval['education'].replace('unknown', 'secondary')
            X_eval['pdays_contacted'] = (X_eval['pdays'] != -1).astype(int)
            X_eval['pdays'] = X_eval['pdays'].replace(-1, 0)

            for col in ['default', 'housing', 'loan']:
                if col in X_eval.columns:
                    X_eval[col] = X_eval[col].map({'no': 0, 'yes': 1})

            nominal_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
            existing_nominal = [c for c in nominal_cols if c in X_eval.columns]
            X_eval_encoded = pd.get_dummies(X_eval, columns=existing_nominal)

            X_eval_final = X_eval_encoded.reindex(columns=selected_features, fill_value=0)
            X_eval_scaled = scaler.transform(X_eval_final)

            # Prediksi & Hitung Metrik
            y_pred_new = model.predict(X_eval_scaled)

            new_acc = accuracy_score(y_actual, y_pred_new)
            new_prec = precision_score(y_actual, y_pred_new, zero_division=0)
            new_rec = recall_score(y_actual, y_pred_new, zero_division=0)
            new_f1 = f1_score(y_actual, y_pred_new, zero_division=0)

            st.write("#### 📊 Perbandingan Metrik Performa Model")

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)

            # Nilai 0.8362 dll ini adalah nilai historis dari Colab Anda
            m_col1.metric("Accuracy", f"{new_acc:.4f}", f"{new_acc - 0.8362:.4f}")
            m_col2.metric("Precision", f"{new_prec:.4f}", f"{new_prec - 0.8452:.4f}")
            m_col3.metric("Recall", f"{new_rec:.4f}", f"{new_rec - 0.8231:.4f}")
            m_col4.metric("F1-Score", f"{new_f1:.4f}", f"{new_f1 - 0.8340:.4f}")