# 🌤️ JakU — Dashboard Kualitas Udara DKI Jakarta

Platform monitoring kualitas udara DKI Jakarta berbasis **machine learning (XGBoost)**, dibangun dengan **Streamlit**. Model dilatih dari dataset ISPU nyata (`Data_ISPU.csv`) mengikuti pipeline CRISP-DM.

> **Pantau Udara, Jaga Jakarta**

---

## ✨ Fitur

| Halaman | Isi |
|---|---|
| **Dashboard** | Ringkasan ISPU DKI Jakarta, peta wilayah (Folium), prediksi 7 hari, tren ISPU |
| **Detail Wilayah** | Tab per kota: Jakarta Pusat, Utara, Barat, Selatan, Timur, Kep. Seribu |
| **Simulasi Prediksi ISPU** | Slider 6 polutan + pilih model (XGBoost / Random Forest / SVM) → prediksi real-time |
| **Edukasi & Insight** | 5 kategori ISPU, dampak kesehatan, sumber polusi, tips |

Popup **"Informasi Polutan"** muncul di Dashboard, Detail Wilayah, dan Simulasi.

---

## 📂 Struktur Project

```
jaku-dashboard/
├── app.py                      # Aplikasi Streamlit (4 halaman)
├── train_model.py              # Script training dari Data_ISPU.csv
├── Data_ISPU.csv               # Dataset ISPU asli (3350 baris)
├── requirements.txt
├── .gitignore
├── README.md
├── .streamlit/
│   └── config.toml
├── assets/
│   └── logo.svg
├── data/                       # Data tampilan (dari statistik dataset asli)
│   ├── wilayah_dummy.csv
│   ├── ispu_dummy.csv
│   ├── prediksi_dummy.csv
│   └── edukasi_dummy.csv
└── models/                     # Model terlatih dari Data_ISPU.csv
    ├── model_xgboost.pkl       # ⭐ model utama (akurasi 98%)
    ├── model_random_forest.pkl
    ├── model_svm.pkl
    ├── label_encoder.pkl
    ├── standard_scaler.pkl
    └── fitur_polutan.pkl
```

---

## ▶️ Menjalankan Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

Buka http://localhost:8501

---

## 🚀 Deploy ke GitHub + Streamlit Community Cloud

### 1. Push ke GitHub

Buat repo baru di [github.com/new](https://github.com/new) (set **Public**, jangan centang "Add README"). Lalu dari folder ini:

```bash
git init
git add .
git commit -m "Initial commit: JakU dashboard kualitas udara"
git branch -M main
git remote add origin https://github.com/USERNAME-ANDA/jaku-dashboard.git
git push -u origin main
```

> Ganti `USERNAME-ANDA`. Saat diminta login, pakai **Personal Access Token** GitHub sebagai password (Settings → Developer settings → Personal access tokens → centang scope `repo`).

### 2. Deploy

1. Buka [share.streamlit.io](https://share.streamlit.io/) → login dengan GitHub
2. **Create app** → **Deploy a public app from GitHub**
3. Isi:
   - **Repository:** `USERNAME-ANDA/jaku-dashboard`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. **Deploy** — tunggu 2–5 menit

Aplikasi akan online di `https://<nama-app>.streamlit.app/`

### 3. Update setelah perubahan

```bash
git add .
git commit -m "deskripsi perubahan"
git push
```

Streamlit Cloud otomatis re-deploy. Jika model `.pkl` diganti, lakukan **Reboot app** dari dashboard (menu ⋮) agar cache `@st.cache_resource` ter-clear.

---

## 🧠 Tentang Model

Model dilatih dari `Data_ISPU.csv` (3350 baris) dengan pipeline identik notebook penelitian:

1. Filter kategori valid (BAIK / SEDANG / TIDAK SEHAT)
2. Standarisasi nama stasiun + konversi numerik
3. Imputasi median + IQR outlier removal (3330 → 3057 baris)
4. Split `test_size=0.2, random_state=42, stratify`
5. Hyperparameter tuning (RandomizedSearchCV / GridSearchCV, 5-fold)

**Akurasi test:** XGBoost **98.0%**, Random Forest 96.9%, SVM 96.9%.

Urutan fitur (wajib sama saat prediksi):
`pm_sepuluh, pm_duakomalima, sulfur_dioksida, karbon_monoksida, ozon, nitrogen_dioksida`

### Melatih ulang model

```bash
python train_model.py
```

Script membaca `Data_ISPU.csv`, melatih ketiga model, dan menyimpan 6 file `.pkl` ke `models/`.

---

## 🛠️ Stack

Streamlit · Pandas · NumPy · Plotly · Folium · scikit-learn · XGBoost · joblib
