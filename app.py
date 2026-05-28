"""
================================================================
JakU - Dashboard Kualitas Udara DKI Jakarta
================================================================
Aplikasi Streamlit untuk monitoring kualitas udara DKI Jakarta
dengan integrasi model machine learning XGBoost.

Halaman:
    1. Dashboard          - Ringkasan kualitas udara provinsi
    2. Detail Wilayah     - Informasi per kota administratif
    3. Simulasi Prediksi  - Prediksi ISPU dari 6 polutan
    4. Edukasi & Insight  - Pengetahuan ISPU, dampak, dan tips
"""

import os
import base64
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from streamlit_option_menu import option_menu
import joblib

# ================================================================
# KONFIGURASI HALAMAN
# ================================================================
st.set_page_config(
    page_title="JakU - Dashboard Kualitas Udara",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================================
# KONSTANTA
# ================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

# Mapping kategori ISPU -> warna, emoji, deskripsi
KATEGORI_INFO = {
    "Baik": {
        "warna": "#16A34A", "warna_bg": "#DCFCE7", "emoji": "😊",
        "rentang": "0 - 50",
        "deskripsi": "Udara bersih, aman untuk beraktivitas sehari-hari.",
        "rekomendasi": "Cocok untuk berolahraga, jalan kaki, dan kegiatan outdoor lainnya."
    },
    "Sedang": {
        "warna": "#2563EB", "warna_bg": "#DBEAFE", "emoji": "😐",
        "rentang": "51 - 100",
        "deskripsi": "Masih dapat diterima untuk beraktivitas di luar ruangan.",
        "rekomendasi": "Aman untuk beraktivitas di luar ruangan. Cocok untuk berolahraga, jalan kaki, dan kegiatan outdoor lainnya."
    },
    "Tidak Sehat": {
        "warna": "#F59E0B", "warna_bg": "#FEF3C7", "emoji": "😷",
        "rentang": "101 - 200",
        "deskripsi": "Kurangi aktivitas luar ruangan, terutama bagi kelompok sensitif.",
        "rekomendasi": "Kurangi aktivitas di luar ruangan. Gunakan masker jika harus keluar."
    },
    "Sangat Tidak Sehat": {
        "warna": "#EF4444", "warna_bg": "#FEE2E2", "emoji": "🤢",
        "rentang": "201 - 300",
        "deskripsi": "Hindari aktivitas luar ruangan. Gunakan masker jika harus keluar.",
        "rekomendasi": "Hindari semua aktivitas luar ruangan. Pakai masker N95 jika terpaksa keluar."
    },
    "Berbahaya": {
        "warna": "#7C3AED", "warna_bg": "#EDE9FE", "emoji": "☠️",
        "rentang": "≥ 301",
        "deskripsi": "Hindari semua aktivitas luar ruangan. Tetap di dalam ruangan.",
        "rekomendasi": "Tetap di dalam ruangan. Gunakan air purifier jika tersedia."
    },
}

# Informasi 6 polutan untuk popup
INFO_POLUTAN = {
    "PM2.5": {
        "warna": "#2563EB",
        "satuan": "µg/m³",
        "deskripsi_pendek": "Partikel sangat halus berukuran ≤ 2.5 mikron",
        "deskripsi": "Partikel sangat halus yang dapat masuk jauh ke dalam paru-paru dan aliran darah."
    },
    "PM10": {
        "warna": "#60A5FA",
        "satuan": "µg/m³",
        "deskripsi_pendek": "Partikel halus berukuran ≤ 10 mikron",
        "deskripsi": "Partikel halus yang dapat masuk ke saluran pernapasan bagian atas dan menyebabkan iritasi."
    },
    "NO₂": {
        "warna": "#8B5CF6",
        "satuan": "µg/m³",
        "deskripsi_pendek": "Nitrogen dioksida, gas hasil pembakaran",
        "deskripsi": "Gas hasil pembakaran kendaraan bermotor dan industri, dapat mengiritasi paru-paru."
    },
    "SO₂": {
        "warna": "#F59E0B",
        "satuan": "µg/m³",
        "deskripsi_pendek": "Sulfur dioksida, gas dari pembakaran bahan bakar fosil",
        "deskripsi": "Gas dari pembakaran bahan bakar fosil, dapat menyebabkan iritasi mata dan saluran pernapasan."
    },
    "CO": {
        "warna": "#10B981",
        "satuan": "mg/m³",
        "deskripsi_pendek": "Karbon monoksida, gas tidak berwarna dan tidak berbau",
        "deskripsi": "Gas tidak berwarna dan tidak berbau yang dapat mengganggu pasokan oksigen dalam tubuh."
    },
    "O₃": {
        "warna": "#06B6D4",
        "satuan": "µg/m³",
        "deskripsi_pendek": "Ozon, terbentuk dari reaksi kimia di atmosfer",
        "deskripsi": "Ozon terbentuk dari reaksi kimia polutan dengan sinar matahari, dapat menyebabkan sesak napas."
    },
}


# ================================================================
# CUSTOM CSS
# ================================================================
def inject_css():
    st.markdown("""
    <style>
    /* Import font modern */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp, .main, .block-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Background utama */
    .stApp {
        background-color: #FAFBFC;
    }

    /* Hilangkan top padding default */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 100% !important;
    }

    /* Hilangkan header & footer Streamlit */
    header[data-testid="stHeader"] {
        background: transparent;
        height: 0;
    }
    #MainMenu, footer {visibility: hidden;}

    /* FIX TAMBAHAN — hilangkan SEMUA chrome Streamlit yang masih muncul
       (toolbar Share/star/edit/GitHub di kanan atas + "Manage app" di kanan bawah) */
    [data-testid="stToolbar"],
    [data-testid="stActionButton"],
    [data-testid="stStatusWidget"],
    [data-testid="stDecoration"],
    .stDeployButton,
    .stAppDeployButton,
    button[kind="header"],
    button[kind="headerNoPadding"],
    div[class*="viewerBadge"],
    iframe[title="streamlit_app"] {
        display: none !important;
        visibility: hidden !important;
    }
    /* Toolbar wrapper kosong tetap memakan tinggi → set 0 */
    .stApp > header { height: 0 !important; }

    /* ============ SIDEBAR ============ */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
        padding-top: 1rem;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
    }

    .sidebar-logo {
        text-align: center;
        padding: 0.5rem 1rem 0.25rem 1rem;
    }
    .sidebar-subtitle {
        text-align: center;
        font-size: 0.78rem;
        color: #64748B;
        font-weight: 500;
        margin-bottom: 1.5rem;
        letter-spacing: 0.02em;
    }

    .sidebar-footer {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin: 1rem 0.5rem;
    }
    .sidebar-footer-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.35rem;
    }
    .sidebar-footer-desc {
        font-size: 0.72rem;
        color: #64748B;
        line-height: 1.45;
        margin-bottom: 0.6rem;
    }
    .sidebar-footer-ts-label {
        font-size: 0.7rem;
        color: #94A3B8;
        margin-bottom: 0.15rem;
    }
    .sidebar-footer-ts {
        font-size: 0.78rem;
        font-weight: 700;
        color: #0F172A;
    }

    /* ============ HEADER HALAMAN ============ */
    .page-title {
        font-size: 1.65rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.25rem;
        letter-spacing: -0.01em;
    }
    .page-subtitle {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }

    .updated-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 0.85rem 1.25rem;
        display: inline-block;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .updated-card-label {
        font-size: 0.72rem;
        color: #64748B;
        margin-bottom: 0.15rem;
    }
    .updated-card-value {
        font-size: 0.92rem;
        font-weight: 700;
        color: #0F172A;
    }

    /* ============ CARD UMUM ============ */
    .card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        transition: all 0.25s ease;
        height: 100%;
    }
    .card:hover {
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
        transform: translateY(-1px);
    }
    .card-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.5rem;
    }

    /* ============ CARD via st.container(border=True) — FIX #3 ============
       Pattern lama (st.markdown("<div class='card'>") ... </div>) bocor
       karena tiap st.markdown jadi DOM container terpisah. Solusi: pakai
       st.container(border=True) native + style border wrapper-nya. */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px !important;
        border: 1px solid #E5E7EB !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        padding: 1.25rem 1.4rem !important;
        transition: all 0.25s ease;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
    }

    /* Map container — rounded corners untuk iframe folium */
    iframe[title="streamlit_folium.st_folium"] {
        border-radius: 12px;
        border: 1px solid #EEF2F7;
    }

    /* ============ ISPU BESAR ============ */
    .ispu-hero {
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }
    .ispu-number {
        font-size: 4.5rem;
        font-weight: 800;
        line-height: 1;
        color: #2563EB;
        letter-spacing: -0.04em;
    }
    .ispu-label {
        font-size: 0.95rem;
        font-weight: 600;
        color: #64748B;
        margin-top: 0.25rem;
    }
    .ispu-status {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .ispu-desc {
        font-size: 0.88rem;
        color: #475569;
        line-height: 1.5;
        max-width: 24rem;
    }
    .ispu-emoji {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }

    /* ============ POLUTAN DOMINAN ============ */
    .polutan-dominan-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-top: 1.5rem;
        padding-top: 1rem;
        border-top: 1px solid #F1F5F9;
    }
    .polutan-dominan-text {
        font-size: 0.9rem;
        color: #0F172A;
    }
    .polutan-dominan-icon {
        color: #16A34A;
    }

    /* ============ METRIC POLUTAN ROW ============ */
    .pollutant-grid {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 1rem;
        margin-top: 1rem;
    }
    .pollutant-cell {
        text-align: center;
    }
    .pollutant-name {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748B;
        margin-bottom: 0.25rem;
    }
    .pollutant-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.1;
    }
    .pollutant-unit {
        font-size: 0.7rem;
        color: #94A3B8;
        margin-top: 0.1rem;
    }

    /* ============ PREDIKSI LIST ============ */
    .pred-row {
        display: grid;
        /* FIX — kolom fixed-width supaya semua baris align presisi: tanggal | badge | kategori | µg/m³
           sebelumnya pakai fr-ratio → spasi tidak konsisten antar baris */
        grid-template-columns: 105px 70px 1fr 90px;
        align-items: center;
        gap: 0.75rem;
        padding: 0.65rem 0;
        border-bottom: 1px solid #F1F5F9;
    }
    .pred-row:last-child { border-bottom: none; }
    .pred-date {
        font-size: 0.88rem;
        color: #334155;
        font-weight: 500;
        white-space: nowrap;
    }
    .pred-pill {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        font-size: 0.88rem;
        font-weight: 700;
        color: #FFFFFF;
        text-align: center;
        min-width: 3rem;
    }
    .pred-cat {
        font-size: 0.88rem;
        font-weight: 600;
        white-space: nowrap;
    }
    .pred-pm {
        font-size: 0.83rem;
        color: #64748B;
        text-align: right;
        white-space: nowrap;
    }

    /* ============ REKOMENDASI CARD ============ */
    .rekom-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        display: flex;
        gap: 0.85rem;
        align-items: flex-start;
        transition: all 0.25s ease;
        height: 100%;
    }
    .rekom-card:hover {
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
        transform: translateY(-1px);
    }
    .rekom-icon {
        font-size: 2rem;
        flex-shrink: 0;
        line-height: 1;
    }
    .rekom-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .rekom-desc {
        font-size: 0.78rem;
        color: #64748B;
        line-height: 1.45;
    }

    /* ============ INFO BOX (ML) ============ */
    .info-box {
        background-color: #EFF6FF;
        border: 1px solid #DBEAFE;
        border-radius: 12px;
        padding: 0.85rem 1.15rem;
        display: flex;
        gap: 0.65rem;
        align-items: flex-start;
        margin-top: 1rem;
    }
    .info-box-icon { color: #2563EB; font-size: 1.1rem; line-height: 1.4; flex-shrink: 0;}
    .info-box-text {
        font-size: 0.85rem;
        color: #1E40AF;
        line-height: 1.5;
    }

    /* ============ KATEGORI ISPU CARD (Edukasi) ============ */
    .kat-card {
        border-radius: 16px;
        padding: 1.3rem 1.1rem;
        height: 100%;
        border: 1px solid;
    }
    .kat-range {
        font-size: 1.7rem;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -0.02em;
    }
    .kat-emoji { font-size: 1.7rem; }
    .kat-name {
        font-size: 1.05rem;
        font-weight: 700;
        margin-top: 0.85rem;
        margin-bottom: 0.4rem;
    }
    .kat-desc {
        font-size: 0.78rem;
        color: #334155;
        line-height: 1.45;
    }

    /* ============ STEP BAR (Simulasi) ============ */
    .step-bar {
        background: #EFF6FF;
        border: 1px solid #DBEAFE;
        border-radius: 14px;
        padding: 1.1rem 1.4rem;
        display: grid;
        grid-template-columns: auto repeat(3, 1fr);
        gap: 1.5rem;
        align-items: center;
        margin-bottom: 1.5rem;
    }
    .step-title {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 700;
        color: #2563EB;
        font-size: 0.95rem;
    }
    .step-item {
        display: flex;
        gap: 0.65rem;
        align-items: flex-start;
    }
    .step-num {
        background: #FFFFFF;
        border: 1px solid #DBEAFE;
        color: #2563EB;
        width: 1.7rem;
        height: 1.7rem;
        border-radius: 999px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
        flex-shrink: 0;
    }
    .step-text {
        font-size: 0.85rem;
        color: #1E40AF;
        line-height: 1.45;
    }

    /* ============ HASIL PREDIKSI ============ */
    .hasil-hero {
        display: flex;
        align-items: flex-start;
        gap: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .hasil-num {
        font-size: 4rem;
        font-weight: 800;
        line-height: 1;
        color: #2563EB;
        letter-spacing: -0.04em;
    }
    .hasil-label-ispu {
        font-size: 0.95rem;
        color: #64748B;
        font-weight: 600;
        text-align: center;
    }
    .rekom-box {
        background-color: #EFF6FF;
        border: 1px solid #DBEAFE;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
    }
    .rekom-box-title {
        font-size: 1rem;
        font-weight: 700;
        color: #2563EB;
        margin-bottom: 0.4rem;
    }
    .rekom-box-text {
        font-size: 0.86rem;
        color: #1E40AF;
        line-height: 1.5;
    }

    /* ============ PRESET BUTTONS (Simulasi) ============ */
    /* Marker wrapper sebelum tombol preset: <div class="preset-marker-{active|idle}"></div>.
       Sibling-combinator selector di bawah menarget tombol Streamlit tepat
       setelah marker tsb. Cara ini bypass keterbatasan Streamlit yang tidak
       memberi kita class CSS langsung pada tombol. */
    .preset-marker-active + div [data-testid="stBaseButton-secondary"],
    .preset-marker-active + div button[kind="secondary"] {
        background: linear-gradient(135deg, #DCFCE7 0%, #BBF7D0 100%) !important;
        border: 2px solid #16A34A !important;
        color: #14532D !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 12px -2px rgba(22, 163, 74, 0.35) !important;
        transform: translateY(-1px);
    }
    .preset-marker-idle + div [data-testid="stBaseButton-secondary"],
    .preset-marker-idle + div button[kind="secondary"] {
        transition: all 0.2s ease-in-out !important;
    }
    .preset-marker-idle + div [data-testid="stBaseButton-secondary"]:hover,
    .preset-marker-idle + div button[kind="secondary"]:hover {
        background: #F0FDF4 !important;
        border-color: #86EFAC !important;
        color: #166534 !important;
    }

    /* Reset button — outline merah lembut supaya beda dengan preset */
    .reset-marker + div [data-testid="stBaseButton-secondary"],
    .reset-marker + div button[kind="secondary"] {
        background: #FFFFFF !important;
        border: 1.5px solid #FCA5A5 !important;
        color: #B91C1C !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out !important;
    }
    .reset-marker + div [data-testid="stBaseButton-secondary"]:hover,
    .reset-marker + div button[kind="secondary"]:hover {
        background: #FEF2F2 !important;
        border-color: #EF4444 !important;
    }

    /* Badge preset aktif di header hasil */
    .active-preset-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: #DCFCE7;
        color: #166534;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        border: 1px solid #86EFAC;
        margin-bottom: 0.6rem;
    }
    .active-preset-badge .dot {
        width: 0.5rem; height: 0.5rem; border-radius: 50%;
        background: #16A34A;
        box-shadow: 0 0 0 3px rgba(22,163,74,0.18);
    }

    /* Fade-in halus tiap kali hasil di-recompute */
    @keyframes sim-fade-in {
        from { opacity: 0; transform: translateY(4px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .sim-fade {
        animation: sim-fade-in 0.25s ease-out;
    }

    /* Baris sub-indeks polutan di hasil */
    .subindex-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.8rem;
        padding: 0.35rem 0.6rem;
        border-radius: 8px;
        margin-bottom: 0.2rem;
    }
    .subindex-row.dominan {
        background: #FEF3C7;
        font-weight: 700;
        color: #92400E;
    }
    .subindex-row .label { color: #475569; }
    .subindex-row .val   { font-variant-numeric: tabular-nums; color: #0F172A; font-weight: 600; }

    /* ============ TABS WILAYAH ============ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: none;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 999px;
        padding: 0.5rem 1.1rem;
        font-weight: 600;
        color: #64748B;
        font-size: 0.88rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #DBEAFE !important;
        color: #2563EB !important;
        border-color: #BFDBFE !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    /* ============ BUTTONS ============ */
    .stButton > button {
        border-radius: 999px;
        font-weight: 600;
        padding: 0.5rem 1.4rem;
        border: 1px solid #E2E8F0;
        transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"] {
        background-color: #2563EB;
        color: white;
        border: none;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #1D4ED8;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(37, 99, 235, 0.25);
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: #2563EB;
        color: #2563EB;
    }

    /* FIX TAMBAHAN — outline pill button (sesuai mockup):
       latar putih, teks biru, border biru tipis. Dipakai untuk tombol
       "Lihat penjelasan polutan" dan "Lihat Selengkapnya". */
    .stButton > button.outline-pill,
    div[data-testid="stButton"] > button {
        /* default semua button non-primary jadi outline pill modern */
    }
    /* Khusus untuk tombol info polutan & lihat selengkapnya — pakai key match */
    div[data-testid="stButton"]:has(button[aria-label*="penjelasan"]) > button,
    div[data-testid="stButton"]:has(button[aria-label*="Selengkapnya"]) > button {
        background: #FFFFFF;
        color: #2563EB;
        border: 1px solid #2563EB;
        font-weight: 600;
    }
    div[data-testid="stButton"]:has(button[aria-label*="penjelasan"]) > button:hover,
    div[data-testid="stButton"]:has(button[aria-label*="Selengkapnya"]) > button:hover {
        background: #EFF6FF;
        color: #1D4ED8;
        border-color: #1D4ED8;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15);
    }

    /* FIX TAMBAHAN — right-align tombol "Lihat penjelasan polutan"
       di dalam kolomnya. Tanpa ini, tombol rapat kiri di column 1/3
       dengan whitespace di kanan (floating effect yang tidak rapi). */
    div[data-testid="stHorizontalBlock"]:has(button[aria-label*="penjelasan"])
        > div:last-child > div[data-testid="stVerticalBlock"] {
        align-items: flex-end !important;
    }
    div[data-testid="stHorizontalBlock"]:has(button[aria-label*="penjelasan"])
        > div:last-child div[data-testid="stButton"] {
        display: flex !important;
        justify-content: flex-end !important;
        width: 100%;
    }

    /* ============ SLIDER ============ */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #2563EB;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.3);
    }

    /* ============ EXPANDER (popup polutan) ============ */
    .streamlit-expanderHeader {
        background-color: #FFFFFF !important;
        border-radius: 14px !important;
        font-weight: 600 !important;
        border: 1px solid #E2E8F0 !important;
    }

    /* ============ DONUT LEGEND CUSTOM ============ */
    .donut-legend-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.45rem 0;
        font-size: 0.88rem;
    }
    .donut-legend-left {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        color: #0F172A;
    }
    .donut-legend-dot {
        width: 0.6rem;
        height: 0.6rem;
        border-radius: 999px;
    }
    .donut-legend-pct {
        font-weight: 700;
        color: #0F172A;
    }

    /* Responsivitas tablet/mobile */
    @media (max-width: 768px) {
        .ispu-number { font-size: 3rem; }
        .pollutant-value { font-size: 1.35rem; }
        .pollutant-grid { grid-template-columns: repeat(3, 1fr); }
        .step-bar { grid-template-columns: 1fr; }
    }
    </style>
    """, unsafe_allow_html=True)


# ================================================================
# UTILITIES
# ================================================================
@st.cache_data
def load_data():
    """Memuat semua data dummy."""
    return {
        "ispu":     pd.read_csv(DATA_DIR / "ispu_dummy.csv"),
        "wilayah":  pd.read_csv(DATA_DIR / "wilayah_dummy.csv"),
        "prediksi": pd.read_csv(DATA_DIR / "prediksi_dummy.csv"),
        "edukasi":  pd.read_csv(DATA_DIR / "edukasi_dummy.csv"),
    }


@st.cache_resource
def load_model():
    """
    Memuat SEMUA artefak model terlatih (sama seperti yang disimpan notebook
    di cell [66]): XGBoost, Random Forest, SVM, LabelEncoder, StandardScaler,
    dan daftar fitur. SVM butuh scaler agar prediksinya identik dengan notebook.
    """
    try:
        return {
            "xgb":    joblib.load(MODELS_DIR / "model_xgboost.pkl"),
            "rf":     joblib.load(MODELS_DIR / "model_random_forest.pkl"),
            "svm":    joblib.load(MODELS_DIR / "model_svm.pkl"),
            "le":     joblib.load(MODELS_DIR / "label_encoder.pkl"),
            "scaler": joblib.load(MODELS_DIR / "standard_scaler.pkl"),
            "fitur":  joblib.load(MODELS_DIR / "fitur_polutan.pkl"),
            "ok": True,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def get_logo_b64():
    """Logo SVG ke base64 untuk disisipkan sebagai <img>."""
    logo_path = ASSETS_DIR / "logo.svg"
    if logo_path.exists():
        return base64.b64encode(logo_path.read_bytes()).decode()
    return ""


def kategori_dari_ispu(ispu):
    """Konversi nilai ISPU ke kategori berdasarkan PERMEN LHK 14/2020."""
    if ispu <= 50:    return "Baik"
    if ispu <= 100:   return "Sedang"
    if ispu <= 200:   return "Tidak Sehat"
    if ispu <= 300:   return "Sangat Tidak Sehat"
    return "Berbahaya"


# ─────────────────────────────────────────────────────────────────
# ISPU sub-index breakpoints — PerMenLHK No. 14/2020
# Tiap polutan punya 5 pita konsentrasi (Baik→Berbahaya) yang dipetakan
# ke 5 pita indeks (0–50, 51–100, 101–200, 201–300, 301–500).
# Konversi konsentrasi → sub-indeks pakai interpolasi linier per pita:
#   I = ((Ia - Ib) / (Xa - Xb)) * (X - Xb) + Ib
# Sub-indeks tertinggi di antara 6 polutan menjadi nilai ISPU final
# dan menentukan kategori — polutan tersebut disebut "polutan dominan".
# Cara ini konsisten dengan praktik pemerintah & badan lingkungan global,
# dan menggantikan rumus linier naif yang membuat semua preset bias ke
# kategori "Tidak Sehat".
# ─────────────────────────────────────────────────────────────────
ISPU_INDEX_BANDS = [(0, 50), (51, 100), (101, 200), (201, 300), (301, 500)]
ISPU_CONC_BANDS = {
    "pm25": [(0, 15.5),   (15.6, 55.4),   (55.5, 150.4),  (150.5, 250.4), (250.5, 500)],
    "pm10": [(0, 50),     (51, 150),      (151, 350),     (351, 420),     (421, 500)],
    "so2":  [(0, 52),     (53, 180),      (181, 400),     (401, 800),     (801, 1200)],
    "co":   [(0, 4),      (4.001, 8),     (8.001, 15),    (15.001, 30),   (30.001, 45)],
    "o3":   [(0, 120),    (121, 235),     (236, 400),     (401, 800),     (801, 1000)],
    "no2":  [(0, 80),     (81, 200),      (201, 1130),    (1131, 2260),   (2261, 3000)],
}


def _ispu_subindex(polutan: str, x: float) -> float:
    """Konversi konsentrasi satu polutan ke sub-indeks ISPU (0–500)."""
    bands = ISPU_CONC_BANDS[polutan]
    if x <= 0:
        return 0.0
    if x >= bands[-1][1]:
        return 500.0
    for i, (lo, hi) in enumerate(bands):
        if lo <= x <= hi:
            Ib, Ia = ISPU_INDEX_BANDS[i]
            return ((Ia - Ib) / (hi - lo)) * (x - lo) + Ib
    return 500.0


def calculate_ispu_category(pm10, pm25, so2, co, o3, no2):
    """
    Hitung nilai ISPU + kategori sesuai PerMenLHK 14/2020.

    Returns: (nilai_ispu, kategori, polutan_dominan, dict_subindex_per_polutan)
    """
    subs = {
        "pm10": _ispu_subindex("pm10", pm10),
        "pm25": _ispu_subindex("pm25", pm25),
        "so2":  _ispu_subindex("so2",  so2),
        "co":   _ispu_subindex("co",   co),
        "o3":   _ispu_subindex("o3",   o3),
        "no2":  _ispu_subindex("no2",  no2),
    }
    final = max(subs.values())
    dominan = max(subs, key=subs.get)

    if final <= 50:
        kategori = "Baik"
    elif final <= 100:
        kategori = "Sedang"
    elif final <= 200:
        kategori = "Tidak Sehat"
    elif final <= 300:
        kategori = "Sangat Tidak Sehat"
    else:
        kategori = "Berbahaya"

    return round(final, 1), kategori, dominan, subs


def hitung_ispu(pm10, pm25, so2, co, o3, no2):
    """Wrapper backward-compatible — hanya mengembalikan (nilai, kategori)."""
    nilai, kategori, _, _ = calculate_ispu_category(pm10, pm25, so2, co, o3, no2)
    return nilai, kategori


# =================================================================
# SVG INLINE HELPERS (FIX #4, #5, #6)
# -----------------------------------------------------------------
# Mengganti emoji native (yang terlihat seperti emoji default sistem)
# dengan SVG inline kustom — konsisten lintas device & sesuai mockup.
# Logo sprout #0A6847 dan ilustrasi Jakarta juga dipindah ke SVG inline.
# =================================================================
def logo_jaku_svg(size=40):
    """
    Logo JakU - sprout sesuai mockup Figma.
    Tiga daun mekar (gelap-terang-tunas) + 2 tetesan biru kecil di bawah daun.
    """
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 64 64"
         xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;">
      <!-- Daun kiri (gelap) -->
      <path d="M30 36 C16 36 8 24 12 10 C26 12 34 24 30 36 Z"
            fill="#0A6847"/>
      <!-- Daun kanan (sedang) -->
      <path d="M34 32 C48 32 56 20 52 6 C38 8 30 20 34 32 Z"
            fill="#16A34A"/>
      <!-- Tunas tengah (lancip ke atas, hijau muda) -->
      <path d="M32 30 C30 22 32 14 32 8 C32 14 34 22 32 30 Z"
            fill="#22C55E"/>
      <!-- Batang -->
      <path d="M32 48 L32 32" stroke="#0A6847" stroke-width="2.5"
            stroke-linecap="round" fill="none"/>
      <!-- Tetesan biru kiri & kanan (aksen air) -->
      <circle cx="26" cy="52" r="2.5" fill="#3B82F6"/>
      <circle cx="38" cy="52" r="2.5" fill="#3B82F6"/>
      <ellipse cx="32" cy="56" rx="3" ry="2" fill="#2563EB" opacity="0.85"/>
    </svg>
    """.strip()


def ispu_emoji_svg(kategori, size=72):
    """
    Emoji status udara dalam SVG inline (flat, clean, konsisten).
    Mengganti emoji native (😐 dll) yang terlihat random per OS.
    """
    cfg = {
        "Baik": {
            "fill": "#16A34A",
            "mouth": '<path d="M30 60 Q50 75 70 60" stroke="white" stroke-width="5" stroke-linecap="round" fill="none"/>',
            "eyes": '<circle cx="36" cy="42" r="4" fill="white"/><circle cx="64" cy="42" r="4" fill="white"/>',
        },
        "Sedang": {
            "fill": "#3B82F6",
            "mouth": '<line x1="35" y1="62" x2="65" y2="62" stroke="white" stroke-width="5" stroke-linecap="round"/>',
            "eyes": '<circle cx="36" cy="42" r="4" fill="white"/><circle cx="64" cy="42" r="4" fill="white"/>',
        },
        "Tidak Sehat": {
            "fill": "#F59E0B",
            "mouth": '<path d="M30 68 Q50 56 70 68" stroke="white" stroke-width="5" stroke-linecap="round" fill="none"/>',
            "eyes": '<line x1="30" y1="40" x2="42" y2="44" stroke="white" stroke-width="4" stroke-linecap="round"/><line x1="70" y1="40" x2="58" y2="44" stroke="white" stroke-width="4" stroke-linecap="round"/>',
        },
        "Sangat Tidak Sehat": {
            "fill": "#EF4444",
            "mouth": '<path d="M30 70 Q50 55 70 70" stroke="white" stroke-width="5" stroke-linecap="round" fill="none"/>',
            "eyes": '<path d="M30 38 L42 48 M42 38 L30 48" stroke="white" stroke-width="4" stroke-linecap="round"/><path d="M58 38 L70 48 M70 38 L58 48" stroke="white" stroke-width="4" stroke-linecap="round"/>',
        },
        "Berbahaya": {
            "fill": "#7C3AED",
            "mouth": '<path d="M30 70 Q50 55 70 70" stroke="white" stroke-width="5" stroke-linecap="round" fill="none"/>',
            "eyes": '<circle cx="36" cy="44" r="6" fill="white"/><circle cx="64" cy="44" r="6" fill="white"/><circle cx="36" cy="44" r="2" fill="#7C3AED"/><circle cx="64" cy="44" r="2" fill="#7C3AED"/>',
        },
    }
    c = cfg.get(kategori, cfg["Sedang"])
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 100 100" '
        f'xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;">'
        f'<circle cx="50" cy="50" r="46" fill="{c["fill"]}"/>'
        f'{c["eyes"]}{c["mouth"]}'
        f'</svg>'
    )


def jakarta_skyline_svg(width=180):
    """
    Ilustrasi flat Jakarta skyline (Monas + gedung).
    Mengikuti mockup: gradient lembut, gedung outline tipis biru-abu,
    Monas tegak dengan ujung emas, pohon-pohon hijau di foreground.
    """
    return f"""
    <svg width="{width}" viewBox="0 0 200 130"
         xmlns="http://www.w3.org/2000/svg"
         style="display:block; opacity:0.95;">
      <defs>
        <linearGradient id="skyGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#EFF6FF"/>
          <stop offset="55%" stop-color="#F0FDF4"/>
          <stop offset="100%" stop-color="#FFFFFF"/>
        </linearGradient>
      </defs>
      <!-- Background gradient -->
      <rect width="200" height="120" fill="url(#skyGrad)" rx="6"/>
      <!-- Gedung-gedung latar (outline tipis, fill sangat lembut) -->
      <rect x="10" y="78" width="18" height="38" fill="#DBEAFE"
            stroke="#94A3B8" stroke-width="0.6" opacity="0.55" rx="1"/>
      <rect x="30" y="62" width="14" height="54" fill="#E0E7FF"
            stroke="#94A3B8" stroke-width="0.6" opacity="0.55" rx="1"/>
      <rect x="46" y="72" width="20" height="44" fill="#DBEAFE"
            stroke="#94A3B8" stroke-width="0.6" opacity="0.55" rx="1"/>
      <rect x="68" y="58" width="16" height="58" fill="#E0E7FF"
            stroke="#94A3B8" stroke-width="0.6" opacity="0.55" rx="1"/>
      <!-- Jendela2 simbolik untuk gedung kiri -->
      <line x1="34" y1="72" x2="42" y2="72" stroke="#94A3B8"
            stroke-width="0.4" opacity="0.6"/>
      <line x1="34" y1="82" x2="42" y2="82" stroke="#94A3B8"
            stroke-width="0.4" opacity="0.6"/>
      <line x1="34" y1="92" x2="42" y2="92" stroke="#94A3B8"
            stroke-width="0.4" opacity="0.6"/>
      <!-- Monas (tugu tengah, paling tinggi) -->
      <rect x="98" y="40" width="4" height="76" fill="#E5E7EB"
            stroke="#64748B" stroke-width="0.5"/>
      <!-- Ujung emas Monas (puncak api) -->
      <polygon points="96,40 104,40 100,28" fill="#FBBF24"
               stroke="#D97706" stroke-width="0.4"/>
      <!-- Base Monas (alas) -->
      <rect x="92" y="106" width="16" height="10" fill="#E5E7EB"
            stroke="#64748B" stroke-width="0.5"/>
      <!-- Gedung-gedung kanan -->
      <rect x="116" y="68" width="16" height="48" fill="#E0E7FF"
            stroke="#94A3B8" stroke-width="0.6" opacity="0.55" rx="1"/>
      <rect x="134" y="75" width="20" height="41" fill="#DBEAFE"
            stroke="#94A3B8" stroke-width="0.6" opacity="0.55" rx="1"/>
      <rect x="156" y="60" width="14" height="56" fill="#E0E7FF"
            stroke="#94A3B8" stroke-width="0.6" opacity="0.55" rx="1"/>
      <rect x="172" y="72" width="18" height="44" fill="#DBEAFE"
            stroke="#94A3B8" stroke-width="0.6" opacity="0.55" rx="1"/>
      <!-- Jendela2 simbolik gedung kanan -->
      <line x1="138" y1="85" x2="150" y2="85" stroke="#94A3B8"
            stroke-width="0.4" opacity="0.6"/>
      <line x1="138" y1="95" x2="150" y2="95" stroke="#94A3B8"
            stroke-width="0.4" opacity="0.6"/>
      <!-- Pohon-pohon foreground (hijau bulat) -->
      <circle cx="14" cy="116" r="8" fill="#16A34A" opacity="0.9"/>
      <circle cx="74" cy="118" r="6" fill="#16A34A" opacity="0.9"/>
      <circle cx="124" cy="118" r="7" fill="#16A34A" opacity="0.9"/>
      <circle cx="186" cy="116" r="8" fill="#16A34A" opacity="0.9"/>
      <!-- Detail pohon (tone berbeda untuk depth) -->
      <circle cx="20" cy="114" r="5" fill="#22C55E" opacity="0.85"/>
      <circle cx="180" cy="114" r="5" fill="#22C55E" opacity="0.85"/>
    </svg>
    """.strip()


def render_legend_safe(kategori_info):
    """
    FIX #1 & #2 — Legend peta yang reliable.

    Sebelumnya: triple-quote + "".join + indentasi membuat Streamlit/markdown
    salah mendeteksi code block, sehingga hanya baris pertama yang terender.

    Sekarang: bangun SATU string HTML utuh tanpa newline & tanpa indentasi
    awal-baris. SATU panggilan st.markdown.
    """
    rows = ""
    for nama, info in kategori_info.items():
        # Inline-only HTML, NO leading whitespace di awal tag baru
        rows += (
            '<div style="display:flex;align-items:center;gap:8px;'
            'margin:7px 0;font-size:13px;color:#334155;">'
            f'<span style="width:11px;height:11px;border-radius:50%;'
            f'background:{info["warna"]};display:inline-block;flex-shrink:0;'
            'box-shadow:0 0 0 2px #fff,0 0 0 3px rgba(15,23,42,0.06);"></span>'
            f'<span><strong style="color:#0F172A;font-weight:600;">{nama}</strong> '
            f'({info["rentang"]})</span>'
            '</div>'
        )
    html = (
        '<div style="padding-top:4px;">'
        '<div style="font-weight:700;font-size:14px;color:#0F172A;'
        'margin-bottom:10px;">Keterangan:</div>'
        + rows +
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def prediksi_ispu_xgboost(pm10, pm25, so2, co, o3, no2, model_choice="xgboost"):
    """
    Prediksi kategori ISPU - replikasi PERSIS fungsi prediksi_ispu() notebook
    (cell [70]). Mendukung 3 model:
        - 'xgboost'        -> model_xgboost.pkl, input mentah
        - 'random_forest'  -> model_random_forest.pkl, input mentah
        - 'svm'            -> model_svm.pkl, input WAJIB di-scale dulu

    Urutan fitur & cara prediksi identik notebook, jadi hasil == notebook
    (selama file .pkl-nya juga dari notebook / dataset yang sama).

    Mengembalikan dict: kategori, nilai_ispu (estimasi untuk display),
    confidence, model_used, fallback.
    """
    art = load_model()
    if not art["ok"]:
        # Fallback bobot polutan jika model gagal dimuat
        nilai = pm25 * 0.30 + pm10 * 0.20 + no2 * 0.15 + so2 * 0.15 + co * 0.10 + o3 * 0.10
        return {
            "kategori": kategori_dari_ispu(nilai),
            "nilai_ispu": int(round(nilai)),
            "confidence": None,
            "model_used": "Formula (fallback)",
            "fallback": True,
        }

    # Susun input PERSIS urutan notebook (cell [70])
    input_df = pd.DataFrame([{
        "pm_sepuluh":        pm10,
        "pm_duakomalima":    pm25,
        "sulfur_dioksida":   so2,
        "karbon_monoksida":  co,
        "ozon":              o3,
        "nitrogen_dioksida": no2,
    }])[art["fitur"]]

    # Pilih model + cara prediksi (sama persis logika notebook)
    confidence = None
    if model_choice == "random_forest":
        model = art["rf"]
        pred_idx = model.predict(input_df)[0]
        model_used = "Random Forest"
        try:
            confidence = float(np.max(model.predict_proba(input_df)[0]))
        except Exception:
            pass
    elif model_choice == "svm":
        model = art["svm"]
        # SVM WAJIB di-scale dulu (cell [70] notebook)
        input_scaled = art["scaler"].transform(input_df)
        pred_idx = model.predict(input_scaled)[0]
        model_used = "SVM"
        # SVC default tanpa probability=True -> tidak ada predict_proba
        try:
            confidence = float(np.max(model.predict_proba(input_scaled)[0]))
        except Exception:
            confidence = None
    else:  # default xgboost
        model = art["xgb"]
        pred_idx = model.predict(input_df)[0]
        model_used = "XGBoost"
        try:
            confidence = float(np.max(model.predict_proba(input_df)[0]))
        except Exception:
            pass

    # Decode label (BAIK / SEDANG / TIDAK SEHAT) -> sama dengan notebook
    kategori_raw = art["le"].inverse_transform([pred_idx])[0]
    kat_map = {"BAIK": "Baik", "SEDANG": "Sedang", "TIDAK SEHAT": "Tidak Sehat"}
    kategori = kat_map.get(kategori_raw, "Sedang")

    # Estimasi nilai ISPU numerik (HANYA untuk display angka besar di UI;
    # kategori tetap mengikuti output model, bukan angka ini)
    nilai = pm25 * 0.30 + pm10 * 0.20 + no2 * 0.15 + so2 * 0.15 + co * 0.10 + o3 * 0.10
    if kategori == "Baik":          nilai = min(nilai, 50)
    elif kategori == "Sedang":      nilai = max(51, min(nilai, 100))
    elif kategori == "Tidak Sehat": nilai = max(101, min(nilai, 200))

    return {
        "kategori": kategori,
        "nilai_ispu": int(round(nilai)),
        "confidence": confidence,
        "model_used": model_used,
        "fallback": False,
    }


def render_popup_polutan():
    """
    Popup "Informasi Polutan" - dipakai di Dashboard, Detail Wilayah,
    dan Simulasi Prediksi. Konten mengikuti gambar referensi POPUP.png.
    """
    @st.dialog("Informasi Polutan", width="large")
    def _popup():
        st.markdown("""
        <p style='color:#64748B; font-size:0.88rem; margin-bottom:1rem; margin-top:-0.5rem;'>
            Penjelasan singkat tiap polutan udara yang dipantau JakU.
        </p>
        """, unsafe_allow_html=True)

        items = list(INFO_POLUTAN.items())
        for i in range(0, len(items), 2):
            cols = st.columns(2, gap="medium")
            for j, col in enumerate(cols):
                if i + j >= len(items):
                    continue
                nama, info = items[i + j]
                with col:
                    st.markdown(f"""
                    <div style="
                        background:#FFFFFF;
                        border:1px solid #E2E8F0;
                        border-radius:14px;
                        padding:1rem 1.1rem;
                        height:100%;
                        min-height:130px;
                    ">
                      <div style="font-weight:700; font-size:1rem; color:#0F172A; margin-bottom:0.45rem;">
                        {nama}
                      </div>
                      <div style="font-size:0.82rem; color:#475569; line-height:1.5;">
                        {info["deskripsi"]}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

    _popup()


# ================================================================
# SIDEBAR
# ================================================================
def render_sidebar():
    """
    FIX #4 — Logo lama (file logo.svg) diganti dengan SVG sprout inline.
    Tidak bergantung file eksternal, ukuran konsisten, warna brand #0A6847.
    """
    with st.sidebar:
        # Logo sprout + teks "JakU" — sejajar horizontal
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; justify-content:center;
                        gap:10px; padding:0.5rem 0 0.2rem 0;">
                {logo_jaku_svg(size=42)}
                <span style="font-size:1.85rem; font-weight:800; letter-spacing:-0.02em;
                             line-height:1;">
                    <span style="color:#0A6847;">Jak</span><span style="color:#2563EB;">U</span>
                </span>
            </div>
            <div class='sidebar-subtitle'>Pantau Udara, Jaga Jakarta</div>
            """,
            unsafe_allow_html=True,
        )

        # Menu utama
        selected = option_menu(
            menu_title=None,
            options=[
                "Dashboard",
                "Detail Wilayah",
                "Simulasi Prediksi ISPU",
                "Edukasi & Insight",
            ],
            icons=["grid", "geo-alt", "bar-chart", "book"],
            default_index=0,
            styles={
                "container": {
                    "padding": "0.25rem 0.5rem",
                    "background-color": "#FFFFFF",
                },
                "icon": {"font-size": "1.05rem"},
                "nav-link": {
                    "font-size": "0.92rem",
                    "font-weight": "500",
                    "color": "#475569",
                    "padding": "0.7rem 1rem",
                    "margin": "0.18rem 0",
                    "border-radius": "10px",
                    "--hover-color": "#F1F5F9",
                },
                "nav-link-selected": {
                    "background-color": "#DBEAFE",
                    "color": "#2563EB",
                    "font-weight": "600",
                },
            },
        )

        # Spacer untuk dorong footer ke bawah
        st.markdown("<div style='flex:1; min-height:6rem;'></div>", unsafe_allow_html=True)

        # Footer sidebar
        st.markdown(
            """
            <div class='sidebar-footer'>
                <div class='sidebar-footer-title'>Data tidak realtime</div>
                <div class='sidebar-footer-desc'>
                    Data yang ditampilkan berdasarkan sampel dan diperbarui secara berkala.
                </div>
                <div class='sidebar-footer-ts-label'>Data terakhir diperbarui</div>
                <div class='sidebar-footer-ts'>26 Mei 2025, 10:00 WIB</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        return selected


# ================================================================
# ================================================================
# HALAMAN 1: DASHBOARD (REWRITE TOTAL — FIX #1, #2, #3, #5, #6, #7)
# ================================================================
def page_dashboard(data):
    """
    Perubahan dari versi lama:
    • Setiap "kartu" sekarang dibungkus st.container(border=True), BUKAN
      pasangan st.markdown("<div class='card'>") + </div>. Sebelumnya
      pattern itu menghasilkan div kosong (FIX #3) karena tiap st.markdown
      dibungkus DOM container terpisah oleh Streamlit.
    • Emoji status → ispu_emoji_svg(kategori) (FIX #5).
    • Ilustrasi kota → jakarta_skyline_svg() (FIX #6).
    • Legend peta → render_legend_safe() (FIX #1 + #2).
    • Tombol "Lihat Selengkapnya" dipindah ke bawah peta+legend dalam
      kartu yang SAMA (FIX #7).
    • Zoom peta 10 → 11 supaya fokus ke DKI Jakarta (FIX #7).
    """
    # ──────────────────────────── HEADER ────────────────────────────
    head1, head2 = st.columns([3, 1.1])
    with head1:
        st.markdown(
            "<div class='page-title'>Halo, Selamat Datang di JakU!</div>"
            "<div class='page-subtitle'>Berikut ringkasan kualitas udara di "
            "Provinsi DKI Jakarta</div>",
            unsafe_allow_html=True,
        )
    with head2:
        st.markdown(
            "<div style='display:flex; justify-content:flex-end; padding-top:0.4rem;'>"
            "<div class='updated-card'>"
            "<div class='updated-card-label'>📅 Data terakhir diperbarui</div>"
            "<div class='updated-card-value'>15 Juni 2024, 10:00 WIB</div>"
            "</div></div>",
            unsafe_allow_html=True,
        )

    # ──────────────── ROW 1: HERO ISPU + PETA WILAYAH ────────────────
    col_left, col_right = st.columns([1.18, 1], gap="medium")

    # ─── KIRI: Hero ISPU ───
    with col_left:
        with st.container(border=True):           # ← FIX #3
            ispu_avg = 78
            kat = kategori_dari_ispu(ispu_avg)
            info = KATEGORI_INFO[kat]

            st.markdown(
                "<div class='card-title'>Kualitas Udara di Jakarta Hari ini "
                "(Rata-rata)</div>",
                unsafe_allow_html=True,
            )

            # Hero layout: pakai Streamlit columns [2, 1] untuk presisi.
            # Sebelumnya pakai single markdown dengan flex 3-child → ilustrasi
            # tidak konsisten posisinya.
            hero_main, hero_illust = st.columns([2.4, 1], gap="small")

            with hero_main:
                # SATU markdown: angka 78 (kiri) + emoji SVG/status/desc (kanan)
                # dengan flex inline, predictable height.
                st.markdown(
                    "<div style='display:flex; align-items:flex-start; "
                    "gap:1.5rem; margin-top:0.25rem;'>"
                    # Kolom kiri: angka ISPU + label
                    "<div style='flex-shrink:0;'>"
                    f"<div style='font-size:5rem; font-weight:800; "
                    f"line-height:0.95; letter-spacing:-0.05em; "
                    f"color:{info['warna']};'>{ispu_avg}</div>"
                    "<div style='font-size:0.92rem; font-weight:600; "
                    "color:#64748B; margin-top:0.3rem;'>ISPU</div>"
                    "</div>"
                    # Kolom kanan: emoji SVG + status + deskripsi
                    "<div style='flex:1; padding-top:0.3rem;'>"
                    f"<div style='margin-bottom:0.55rem;'>"
                    f"{ispu_emoji_svg(kat, size=52)}</div>"
                    f"<div style='font-size:1.35rem; font-weight:700; "
                    f"color:{info['warna']}; margin-bottom:0.4rem;'>"
                    f"Udara {kat}</div>"
                    "<div style='font-size:0.86rem; color:#475569; "
                    "line-height:1.55;'>"
                    f"{info['deskripsi']}</div>"
                    "</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )

            with hero_illust:
                # Ilustrasi Jakarta + caption — center di kolomnya sendiri,
                # tidak lagi tergantung margin-left:auto yang plin-plan.
                st.markdown(
                    "<div style='text-align:center; padding-top:0.4rem;'>"
                    f"{jakarta_skyline_svg(width=180)}"
                    "<div style='font-size:0.8rem; color:#64748B; "
                    "font-weight:500; margin-top:0.2rem;'>DKI Jakarta</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )

            # ─── Polutan dominan strip + tombol info polutan ───
            # FIX — sebelumnya tombol "floating" di tengah card karena
            # padding-top fix tidak match dengan baseline polutan strip.
            # Sekarang: garis separator full-width via markdown, lalu
            # strip pakai 2-column dengan vertical_alignment="center"
            # supaya tombol & teks polutan benar-benar sejajar baseline.
            st.markdown(
                "<div style='border-top:1px solid #F1F5F9; "
                "margin-top:1.1rem;'></div>",
                unsafe_allow_html=True,
            )

            try:
                pdc1, pdc2 = st.columns([1.6, 1], vertical_alignment="center")
            except TypeError:
                # Fallback untuk Streamlit < 1.36 yang tidak punya vertical_alignment
                pdc1, pdc2 = st.columns([1.6, 1])

            with pdc1:
                st.markdown(
                    "<div style='display:flex; align-items:center; "
                    "gap:0.5rem; padding-top:0.85rem; font-size:0.92rem; "
                    "color:#0F172A;'>"
                    "<span style='color:#16A34A; font-size:1.05rem;'>🌿</span>"
                    "<span><strong>Polutan dominan:</strong>&nbsp; "
                    "PM2.5 (24 µg/m³)</span>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            with pdc2:
                # Tombol natural-width; CSS di awal file akan right-align
                # via :has selector untuk kolom yang memuat tombol ini.
                if st.button("ⓘ  Lihat penjelasan polutan",
                             key="btn_info_dashboard"):
                    render_popup_polutan()

            # 6 polutan compact — SATU markdown call
            st.markdown(
                "<div class='pollutant-grid'>"
                "<div class='pollutant-cell'><div class='pollutant-name'>PM2.5</div>"
                "<div class='pollutant-value'>24</div>"
                "<div class='pollutant-unit'>µg/m³</div></div>"
                "<div class='pollutant-cell'><div class='pollutant-name'>PM10</div>"
                "<div class='pollutant-value'>41</div>"
                "<div class='pollutant-unit'>µg/m³</div></div>"
                "<div class='pollutant-cell'><div class='pollutant-name'>NO₂</div>"
                "<div class='pollutant-value'>18</div>"
                "<div class='pollutant-unit'>µg/m³</div></div>"
                "<div class='pollutant-cell'><div class='pollutant-name'>SO₂</div>"
                "<div class='pollutant-value'>7</div>"
                "<div class='pollutant-unit'>µg/m³</div></div>"
                "<div class='pollutant-cell'><div class='pollutant-name'>CO</div>"
                "<div class='pollutant-value'>0.6</div>"
                "<div class='pollutant-unit'>mg/m³</div></div>"
                "<div class='pollutant-cell'><div class='pollutant-name'>O₃</div>"
                "<div class='pollutant-value'>50</div>"
                "<div class='pollutant-unit'>µg/m³</div></div>"
                "</div>",
                unsafe_allow_html=True,
            )

    # ─── KANAN: Peta wilayah + legend + tombol ───
    with col_right:
        with st.container(border=True):           # ← FIX #3
            st.markdown(
                "<div class='card-title'>Kualitas Udara per Wilayah di Jakarta</div>",
                unsafe_allow_html=True,
            )

            # Peta + legend side-by-side
            mc1, mc2 = st.columns([1.9, 1], gap="small")
            with mc1:
                # FIX — zoom 11 → 12 dan max_bounds untuk benar-benar kunci ke DKI.
                # Sebelumnya fit_bounds tidak cukup ketat → Tangerang & Bekasi
                # masih besar di viewport.
                m = folium.Map(
                    location=[-6.2088, 106.8456],
                    zoom_start=12,
                    tiles="CartoDB positron",
                    zoom_control=False,
                    scrollWheelZoom=False,
                    dragging=True,
                    min_zoom=11,
                    max_zoom=14,
                )
                # Hard-lock viewport ke DKI Jakarta
                m.options['maxBounds'] = [[-6.40, 106.65], [-6.05, 107.05]]
                m.options['maxBoundsViscosity'] = 1.0
                m.fit_bounds([[-6.30, 106.78], [-6.10, 106.95]])
                for _, row in data["wilayah"].iterrows():
                    kat_w = row["kategori"]
                    warna = KATEGORI_INFO.get(
                        kat_w, KATEGORI_INFO["Sedang"]
                    )["warna"]
                    folium.CircleMarker(
                        location=[row["lat"], row["lon"]],
                        radius=24,
                        color="white",
                        weight=3,
                        fill=True,
                        fillColor=warna,
                        fillOpacity=0.95,
                        tooltip=f"{row['wilayah']}: {row['ispu']}",
                    ).add_to(m)
                    folium.map.Marker(
                        [row["lat"], row["lon"]],
                        icon=folium.DivIcon(
                            icon_size=(40, 40),
                            icon_anchor=(20, 20),
                            html=(
                                "<div style='font-size:12px; font-weight:800; "
                                "color:white; text-align:center; "
                                f"line-height:40px;'>{row['ispu']}</div>"
                            ),
                        ),
                    ).add_to(m)
                st_folium(m, height=290, use_container_width=True,
                          returned_objects=[])

            with mc2:
                # FIX #1 + #2 — legend reliable via render_legend_safe
                render_legend_safe(KATEGORI_INFO)

                # FIX — tombol "Lihat Selengkapnya" sekarang di KOLOM LEGEND
                # (kanan-bawah, sejajar di samping peta) sesuai mockup,
                # bukan di baris terpisah di bawah peta + legend.
                # Style: outline pill (bukan solid primary) — match mockup.
                st.markdown(
                    "<div style='margin-top:1.2rem;'></div>",
                    unsafe_allow_html=True,
                )
                if st.button("Lihat Selengkapnya  →",
                             key="btn_selengkapnya",
                             use_container_width=True):
                    st.session_state["jump_to_detail"] = True
                    st.rerun()

    # ──────────────── ROW 2: PREDIKSI + TREN ────────────────
    st.markdown("<div style='margin-top:1.25rem;'></div>", unsafe_allow_html=True)
    pcol1, pcol2 = st.columns([1, 1.4], gap="medium")

    # ─── Prediksi 7 hari mendatang ───
    with pcol1:
        with st.container(border=True):           # ← FIX #3
            st.markdown(
                "<div class='card-title'>Prediksi ISPU di Jakarta "
                "(7 Hari Mendatang)</div>",
                unsafe_allow_html=True,
            )
            pred_dki = data["prediksi"][data["prediksi"]["wilayah"] == "DKI Jakarta"]
            rows_html = ""
            for _, r in pred_dki.iterrows():
                kat2 = r["kategori"]
                warna = KATEGORI_INFO.get(
                    kat2, KATEGORI_INFO["Sedang"]
                )["warna"]
                tanggal = pd.to_datetime(r["tanggal"]).strftime("%d %b %Y")
                rows_html += (
                    "<div class='pred-row'>"
                    f"<div class='pred-date'>{tanggal}</div>"
                    "<div>"
                    f"<span class='pred-pill' style='background:{warna};'>"
                    f"{r['ispu']}</span>"
                    "</div>"
                    f"<div class='pred-cat' style='color:{warna};'>{kat2}</div>"
                    f"<div class='pred-pm'>{r['pm25']} µg/m³</div>"
                    "</div>"
                )
            st.markdown(rows_html, unsafe_allow_html=True)

    # ─── Tren 7 hari terakhir (chart) ───
    with pcol2:
        with st.container(border=True):           # ← FIX #3
            st.markdown(
                "<div class='card-title'>Tren ISPU di Jakarta (7 Hari Terakhir)</div>",
                unsafe_allow_html=True,
            )

            df_tren = data["ispu"].copy()
            df_tren["tanggal"] = pd.to_datetime(df_tren["tanggal"])
            df_tren["label_x"] = df_tren["tanggal"].dt.strftime("%d %b")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_tren["label_x"], y=df_tren["ispu"],
                mode="lines+markers+text",
                text=df_tren["ispu"],
                textposition="top center",
                textfont=dict(size=11, color="#0F172A", weight=600),
                line=dict(color="#2563EB", width=3,
                          shape="spline", smoothing=1.0),
                marker=dict(size=9, color="#2563EB",
                            line=dict(color="white", width=2)),
                fill="tozeroy",
                fillcolor="rgba(37, 99, 235, 0.08)",
                hovertemplate="<b>%{x}</b><br>ISPU: %{y}<extra></extra>",
                showlegend=False,
            ))
            for nilai, label, warna in [
                (50, "Baik", "#16A34A"),
                (100, "Sedang", "#2563EB"),
                (200, "Tidak Sehat", "#F59E0B"),
                (300, "Sangat Tidak Sehat", "#EF4444"),
            ]:
                fig.add_hline(y=nilai, line_dash="dot",
                              line_color="#E2E8F0", line_width=1)
                fig.add_annotation(
                    x=1.0, xref="paper", y=nilai,
                    text=label, showarrow=False,
                    xanchor="left", yanchor="middle",
                    font=dict(size=10, color=warna, weight=600),
                    xshift=8,
                )
            fig.update_layout(
                # FIX — margin lebih besar di kiri/kanan/atas supaya label "62"
                # (di awal) dan "78" (di akhir) tidak terpotong; t=45 supaya
                # angka di atas marker tidak nyentuh batas card.
                height=340,
                margin=dict(l=40, r=140, t=50, b=30),
                paper_bgcolor="white", plot_bgcolor="white",
                xaxis=dict(
                    showgrid=False, showline=False,
                    tickfont=dict(size=11, color="#64748B"),
                    # Padding kiri-kanan: extend domain agar marker awal/akhir
                    # punya breathing room untuk label
                    range=[-0.4, 6.4],
                ),
                yaxis=dict(
                    range=[0, 320], gridcolor="#F1F5F9", showline=False,
                    tickfont=dict(size=11, color="#94A3B8"),
                    tickvals=[0, 50, 100, 150, 200, 300],
                ),
            )
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})

    # ──────────────── INFO BOX ML ────────────────
    st.markdown(
        "<div class='info-box'>"
        "<div class='info-box-icon'>ⓘ</div>"
        "<div class='info-box-text'>"
        "Prediksi ini dibuat menggunakan model machine learning "
        "<strong>XGBoost</strong> berdasarkan data historis ISPU pada tahun 2024."
        "</div></div>",
        unsafe_allow_html=True,
    )

    # ──────────────── REKOMENDASI AKTIVITAS ────────────────
    st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
    with st.container(border=True):               # ← FIX #3
        st.markdown(
            "<div class='card-title'>Rekomendasi Aktivitas</div>",
            unsafe_allow_html=True,
        )
        rekomendasi = [
            ("🏃‍♀️", "Olahraga Luar Ruangan",
             "Aktivitas luar ruangan aman dilakukan."),
            ("😷",   "Gunakan Masker",
             "Gunakan masker jika Anda sensitif terhadap polusi."),
            ("👵",   "Kelompok Sensitif",
             "Jaga kesehatan dan hindari area dengan polusi tinggi."),
            ("🌳",   "Buka Jendela",
             "Sirkulasi udara di dalam ruangan masih aman."),
        ]
        rc = st.columns(4, gap="medium")
        for col, (icon, judul, desc) in zip(rc, rekomendasi):
            with col:
                st.markdown(
                    "<div class='rekom-card'>"
                    f"<div class='rekom-icon'>{icon}</div>"
                    "<div>"
                    f"<div class='rekom-title'>{judul}</div>"
                    f"<div class='rekom-desc'>{desc}</div>"
                    "</div></div>",
                    unsafe_allow_html=True,
                )




# ================================================================
# HALAMAN 2: DETAIL WILAYAH
# ================================================================
def page_detail_wilayah(data):
    st.markdown(
        "<div class='page-title'>Detail Wilayah</div>"
        "<div class='page-subtitle'>Pilih wilayah untuk melihat informasi kualitas udara lebih detail.</div>",
        unsafe_allow_html=True,
    )

    # Tabs wilayah
    wilayah_list = data["wilayah"]["wilayah"].tolist()
    tabs = st.tabs(wilayah_list)

    for tab, wilayah in zip(tabs, wilayah_list):
        with tab:
            row = data["wilayah"][data["wilayah"]["wilayah"] == wilayah].iloc[0]
            kat = row["kategori"]
            info = KATEGORI_INFO[kat]

            # Kualitas udara + Rekomendasi
            c1, c2 = st.columns([1.1, 1], gap="medium")

            # ---- Card kualitas udara
            with c1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='card-title'>Kualitas Udara {wilayah}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <div class='ispu-hero'>
                        <div>
                            <div class='ispu-number' style='color:{info["warna"]};'>{row["ispu"]}</div>
                            <div class='ispu-label'>ISPU</div>
                        </div>
                        <div>
                            <div class='ispu-emoji'>{info["emoji"]}</div>
                            <div class='ispu-status' style='color:{info["warna"]};'>Udara {kat}</div>
                            <div class='ispu-desc'>{info["deskripsi"]}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                pdc1, pdc2 = st.columns([2, 1])
                with pdc1:
                    st.markdown(
                        f"""
                        <div class='polutan-dominan-row' style='border-top:1px solid #F1F5F9; padding-top:1rem; margin-top:1.2rem;'>
                            <div class='polutan-dominan-text'>
                                🌿 <strong>Polutan dominan:</strong> PM2.5 ({row["pm25"]} µg/m³)
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with pdc2:
                    st.markdown("<div style='padding-top:1.2rem;'></div>", unsafe_allow_html=True)
                    if st.button("ⓘ Lihat penjelasan polutan", key=f"btn_info_{wilayah}", use_container_width=True):
                        render_popup_polutan()

                st.markdown(
                    f"""
                    <div class='pollutant-grid'>
                      <div class='pollutant-cell'><div class='pollutant-name'>PM2.5</div><div class='pollutant-value'>{row["pm25"]}</div><div class='pollutant-unit'>µg/m³</div></div>
                      <div class='pollutant-cell'><div class='pollutant-name'>PM10</div><div class='pollutant-value'>{row["pm10"]}</div><div class='pollutant-unit'>µg/m³</div></div>
                      <div class='pollutant-cell'><div class='pollutant-name'>NO₂</div><div class='pollutant-value'>{row["no2"]}</div><div class='pollutant-unit'>µg/m³</div></div>
                      <div class='pollutant-cell'><div class='pollutant-name'>SO₂</div><div class='pollutant-value'>{row["so2"]}</div><div class='pollutant-unit'>µg/m³</div></div>
                      <div class='pollutant-cell'><div class='pollutant-name'>CO</div><div class='pollutant-value'>{row["co"]}</div><div class='pollutant-unit'>mg/m³</div></div>
                      <div class='pollutant-cell'><div class='pollutant-name'>O₃</div><div class='pollutant-value'>{row["o3"]}</div><div class='pollutant-unit'>µg/m³</div></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            # ---- Rekomendasi aktivitas (4 item dalam 2x2 grid)
            with c2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<div class='card-title'>Rekomendasi Aktivitas</div>", unsafe_allow_html=True)

                rekomendasi = [
                    ("🏃‍♀️", "Olahraga Luar Ruangan", "Aktivitas luar ruangan aman dilakukan."),
                    ("😷",   "Gunakan Masker",         "Gunakan masker jika Anda sensitif terhadap polusi."),
                    ("👵",   "Kelompok Sensitif",      "Jaga kesehatan dan hindari area dengan polusi tinggi."),
                    ("🌳",   "Buka Jendela",           "Sirkulasi udara di dalam ruangan masih aman."),
                ]
                gc1, gc2 = st.columns(2, gap="small")
                for idx, (icon, judul, desc) in enumerate(rekomendasi):
                    with (gc1 if idx % 2 == 0 else gc2):
                        st.markdown(
                            f"""
                            <div class='rekom-card' style='margin-bottom:0.6rem;'>
                                <div class='rekom-icon'>{icon}</div>
                                <div>
                                    <div class='rekom-title'>{judul}</div>
                                    <div class='rekom-desc'>{desc}</div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)

            # Prediksi + Tren
            pc1, pc2 = st.columns([1, 1.4], gap="medium")

            # Prediksi 7 hari
            with pc1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='card-title'>Prediksi ISPU di {wilayah} (7 Hari Mendatang)</div>",
                    unsafe_allow_html=True,
                )
                pred_w = data["prediksi"][data["prediksi"]["wilayah"] == wilayah]
                rows_html = ""
                for _, r in pred_w.iterrows():
                    kat2 = r["kategori"]
                    warna = KATEGORI_INFO.get(kat2, KATEGORI_INFO["Sedang"])["warna"]
                    tanggal = pd.to_datetime(r["tanggal"]).strftime("%d %b %Y")
                    rows_html += f"""
                    <div class='pred-row'>
                        <div class='pred-date'>{tanggal}</div>
                        <div><span class='pred-pill' style='background:{warna};'>{r["ispu"]}</span></div>
                        <div class='pred-cat' style='color:{warna};'>{kat2}</div>
                        <div class='pred-pm'>{r["pm25"]} µg/m³</div>
                    </div>
                    """
                st.markdown(rows_html, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Tren 7 hari (data dummy diolah per wilayah)
            with pc2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='card-title'>Tren ISPU di {wilayah} (7 Hari Terakhir)</div>",
                    unsafe_allow_html=True,
                )

                df_tren = data["ispu"].copy()
                df_tren["tanggal"] = pd.to_datetime(df_tren["tanggal"])
                # Tambahkan variasi kecil per wilayah agar tidak monoton
                np.random.seed(hash(wilayah) % 1000)
                offset = np.random.uniform(-15, 15, len(df_tren))
                df_tren["ispu_w"] = (df_tren["ispu"] + offset).clip(20, 250).round().astype(int)
                df_tren["label_x"] = df_tren["tanggal"].dt.strftime("%d %b")

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_tren["label_x"], y=df_tren["ispu_w"],
                    mode="lines+markers+text",
                    text=df_tren["ispu_w"],
                    textposition="top center",
                    textfont=dict(size=11, color="#0F172A", weight=600),
                    line=dict(color="#2563EB", width=3, shape="spline", smoothing=1.0),
                    marker=dict(size=8, color="#2563EB", line=dict(color="white", width=2)),
                    fill="tozeroy",
                    fillcolor="rgba(37, 99, 235, 0.08)",
                    hovertemplate="<b>%{x}</b><br>ISPU: %{y}<extra></extra>",
                    showlegend=False,
                ))
                for nilai, label, warna in [
                    (50, "Baik", "#16A34A"),
                    (100, "Sedang", "#2563EB"),
                    (200, "Tidak Sehat", "#F59E0B"),
                    (300, "Sangat Tidak Sehat", "#EF4444"),
                ]:
                    fig.add_annotation(
                        x=1.0, xref="paper", y=nilai,
                        text=label, showarrow=False,
                        xanchor="left", yanchor="middle",
                        font=dict(size=10, color=warna, weight=600),
                        xshift=8,
                    )
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=120, t=20, b=20),
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    xaxis=dict(showgrid=False, showline=False, tickfont=dict(size=11, color="#64748B")),
                    yaxis=dict(
                        range=[0, 310],
                        gridcolor="#F1F5F9",
                        showline=False,
                        tickfont=dict(size=11, color="#94A3B8"),
                        tickvals=[0, 50, 100, 150, 200, 300],
                    ),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                st.markdown("</div>", unsafe_allow_html=True)

            # Info box ML
            st.markdown(
                """
                <div class='info-box'>
                    <div class='info-box-icon'>ⓘ</div>
                    <div class='info-box-text'>
                        Prediksi ini dibuat menggunakan model machine learning <strong>XGBoost</strong>
                        berdasarkan data historis ISPU pada tahun 2024.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ================================================================
# HALAMAN 3: SIMULASI PREDIKSI ISPU
# ================================================================
# ── Konfigurasi terpusat ───────────────────────────────────────
# DEFAULT_VALUES untuk state awal & target tombol Reset.
# Semua nol → ISPU final = 0, kategori = "Baik" (kondisi netral).
# Preset di bawah dirancang agar masing-masing jatuh tepat di
# kategori yang dimaksud sesuai breakpoint PerMenLHK No. 14/2020.
SIM_DEFAULT_VALUES = {
    "pm25": 0.0, "pm10": 0.0, "no2": 0.0,
    "so2":  0.0, "co":  0.0, "o3":  0.0,
}

# Preset: 5 skenario, satu per kategori ISPU.
# Setiap preset dirancang agar polutan dominan jatuh di pita target,
# sehingga ISPU final berada dalam rentang yang user inginkan.
SIM_PRESETS = {
    "Baik":               {"pm25": 10.0,  "pm10": 20.0,  "no2": 10.0,  "so2": 10.0,  "co": 1.0,  "o3": 20.0},
    "Sedang":             {"pm25": 35.0,  "pm10": 60.0,  "no2": 20.0,  "so2": 25.0,  "co": 2.0,  "o3": 45.0},
    "Tidak Sehat":        {"pm25": 90.0,  "pm10": 140.0, "no2": 60.0,  "so2": 70.0,  "co": 8.0,  "o3": 120.0},
    "Sangat Tidak Sehat": {"pm25": 180.0, "pm10": 260.0, "no2": 150.0, "so2": 180.0, "co": 18.0, "o3": 220.0},
    "Berbahaya":          {"pm25": 300.0, "pm10": 450.0, "no2": 300.0, "so2": 320.0, "co": 35.0, "o3": 400.0},
}

# Konfigurasi slider per polutan (min/max/step + metadata UI).
# Max range diset agar memuat preset "Berbahaya" tanpa harus extend lagi.
SIM_SLIDER_CONFIG = {
    "pm25": {"label": "PM2.5", "info_key": "PM2.5", "min": 0.0, "max": 500.0, "step": 0.5, "unit": "µg/m³", "decimals": 2, "slider_key": "sl_pm25"},
    "pm10": {"label": "PM10",  "info_key": "PM10",  "min": 0.0, "max": 500.0, "step": 0.5, "unit": "µg/m³", "decimals": 2, "slider_key": "sl_pm10"},
    "no2":  {"label": "NO₂",   "info_key": "NO₂",   "min": 0.0, "max": 500.0, "step": 0.5, "unit": "µg/m³", "decimals": 2, "slider_key": "sl_no2"},
    "so2":  {"label": "SO₂",   "info_key": "SO₂",   "min": 0.0, "max": 500.0, "step": 0.5, "unit": "µg/m³", "decimals": 2, "slider_key": "sl_so2"},
    "co":   {"label": "CO",    "info_key": "CO",    "min": 0.0, "max": 50.0,  "step": 0.1, "unit": "mg/m³", "decimals": 2, "slider_key": "sl_co"},
    "o3":   {"label": "O₃",    "info_key": "O₃",    "min": 0.0, "max": 500.0, "step": 0.5, "unit": "µg/m³", "decimals": 2, "slider_key": "sl_o3"},
}

POLUTAN_DISPLAY_NAME = {
    "pm25": "PM2.5", "pm10": "PM10", "no2": "NO₂",
    "so2":  "SO₂",   "co":   "CO",   "o3":  "O₃",
}


def _sim_init_state():
    """Init session state untuk simulasi — idempoten, aman dipanggil tiap rerun."""
    for pol, cfg in SIM_SLIDER_CONFIG.items():
        if cfg["slider_key"] not in st.session_state:
            st.session_state[cfg["slider_key"]] = float(SIM_DEFAULT_VALUES[pol])
    if "sim_active_preset" not in st.session_state:
        st.session_state["sim_active_preset"] = None
    if "sim_model_choice" not in st.session_state:
        st.session_state["sim_model_choice"] = "xgboost"


def apply_preset(name: str):
    """
    Callback `on_click` untuk tombol preset.
    Modifikasi session_state SEBELUM widget di-instantiate pada rerun berikutnya,
    sehingga slider otomatis nge-snap ke nilai preset tanpa st.rerun() manual.
    """
    if name not in SIM_PRESETS:
        return
    preset = SIM_PRESETS[name]
    for pol, val in preset.items():
        st.session_state[SIM_SLIDER_CONFIG[pol]["slider_key"]] = float(val)
    st.session_state["sim_active_preset"] = name


def reset_simulation():
    """
    Callback `on_click` untuk tombol Reset.
    Kembalikan semua slider ke default + hapus penanda preset aktif.
    """
    for pol, val in SIM_DEFAULT_VALUES.items():
        st.session_state[SIM_SLIDER_CONFIG[pol]["slider_key"]] = float(val)
    st.session_state["sim_active_preset"] = None


def _detect_active_preset(current_vals: dict):
    """
    Sinkronisasi 2-arah: deteksi preset aktif dari nilai slider.
    Jika user menggeser slider manual sehingga keluar dari preset,
    badge highlight otomatis hilang.
    """
    for name, preset in SIM_PRESETS.items():
        if all(abs(current_vals[k] - preset[k]) < 0.01 for k in preset):
            return name
    return None


def _polutan_slider_block(pol_key: str):
    """Render satu slider polutan + label header. Mengembalikan nilai terbaru."""
    cfg = SIM_SLIDER_CONFIG[pol_key]
    info = INFO_POLUTAN[cfg["info_key"]]
    st.markdown(
        f"<div style='display:flex; align-items:center; gap:0.4rem; font-weight:600; "
        f"color:#0F172A; margin-bottom:0.1rem;'>"
        f"<span style='width:0.7rem; height:0.7rem; border-radius:999px; "
        f"background:{info['warna']};'></span>{cfg['label']}</div>"
        f"<div style='font-size:0.78rem; color:#64748B; margin-bottom:0.3rem;'>"
        f"{info['deskripsi_pendek']}</div>",
        unsafe_allow_html=True,
    )
    val = st.slider(
        cfg["label"], cfg["min"], cfg["max"],
        value=float(st.session_state[cfg["slider_key"]]),
        step=cfg["step"], key=cfg["slider_key"],
        label_visibility="collapsed",
    )
    st.markdown(
        f"<div style='text-align:right; font-size:0.8rem; color:#64748B;'>"
        f"{val:.{cfg['decimals']}f} ({cfg['unit']})</div>",
        unsafe_allow_html=True,
    )
    return val


def page_simulasi(data):
    st.markdown(
        "<div class='page-title'>Simulasi Prediksi ISPU</div>"
        "<div class='page-subtitle'>Simulasikan kualitas udara berdasarkan konsentrasi polutan.</div>",
        unsafe_allow_html=True,
    )

    # Banner panduan
    st.markdown(
        """
        <div class='step-bar'>
            <div class='step-title'>ⓘ Cara Menggunakan Simulasi</div>
            <div class='step-item'>
                <div class='step-num'>1</div>
                <div class='step-text'>Pilih preset skenario atau geser slider untuk mengatur konsentrasi polutan.</div>
            </div>
            <div class='step-item'>
                <div class='step-num'>2</div>
                <div class='step-text'>Hasil ISPU dan kategori akan ter-update secara real-time di samping kanan.</div>
            </div>
            <div class='step-item'>
                <div class='step-num'>3</div>
                <div class='step-text'>Tekan "Reset" untuk mengembalikan semua slider ke kondisi awal.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Init state (idempoten) ──
    _sim_init_state()

    # Layout: kiri = form polutan, kanan = hasil
    col_left, col_right = st.columns([1.05, 1], gap="medium")

    # ────────────────────── KIRI: Form polutan ──────────────────────
    with col_left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        head = st.columns([5, 1])
        with head[0]:
            st.markdown(
                "<div class='card-title' style='margin-bottom:0.3rem;'>Komposisi Polutan</div>"
                "<div style='font-size:0.85rem; color:#64748B; line-height:1.5; margin-bottom:1rem;'>"
                "Sesuaikan slider di bawah untuk mensimulasikan kondisi polutan dan memprediksi "
                "Indeks Standar Pencemar Udara (ISPU)."
                "</div>",
                unsafe_allow_html=True,
            )
        with head[1]:
            st.markdown("<div style='padding-top:0.3rem;'></div>", unsafe_allow_html=True)
            if st.button("ⓘ Info", key="btn_info_simulasi", use_container_width=True):
                render_popup_polutan()

        # ── Preset buttons (highlight via marker div sebelum tombol) ──
        st.markdown(
            "<div style='margin-bottom:0.5rem; font-size:0.85rem; color:#475569; font-weight:600;'>"
            "Preset Skenario</div>",
            unsafe_allow_html=True,
        )
        preset_labels = {
            "Baik":               ("Baik",        "Kualitas udara Baik (ISPU 0–50)"),
            "Sedang":             ("Sedang",      "Kualitas udara Sedang (ISPU 51–100)"),
            "Tidak Sehat":        ("Tidak Sehat", "Kualitas udara Tidak Sehat (ISPU 101–200)"),
            "Sangat Tidak Sehat": ("Sangat",      "Kualitas udara Sangat Tidak Sehat (ISPU 201–300)"),
            "Berbahaya":          ("Berbahaya",   "Kualitas udara Berbahaya (ISPU ≥ 301)"),
        }
        # Aktif diambil dari session_state (di-set callback). Setelah render
        # selesai, kita re-deteksi berdasar nilai slider terbaru (sinkron 2-arah).
        current_active = st.session_state.get("sim_active_preset")
        pc = st.columns(5, gap="small")
        for col, (name, (label, tip)) in zip(pc, preset_labels.items()):
            with col:
                marker_cls = "preset-marker-active" if current_active == name else "preset-marker-idle"
                st.markdown(f'<div class="{marker_cls}"></div>', unsafe_allow_html=True)
                st.button(
                    label, key=f"preset_{name.lower().replace(' ', '_')}",
                    use_container_width=True, help=tip,
                    on_click=apply_preset, args=(name,),
                )

        # ── Pilih model klasifikasi ML ──
        st.markdown(
            "<div style='margin-top:1rem; margin-bottom:0.3rem; font-size:0.85rem; "
            "color:#475569; font-weight:600;'>Model Klasifikasi</div>",
            unsafe_allow_html=True,
        )
        model_label = st.selectbox(
            "Model Klasifikasi",
            ["XGBoost (Rekomendasi)", "Random Forest", "SVM"],
            label_visibility="collapsed",
            help="XGBoost direkomendasikan karena akurasi tertinggi pada data uji.",
            key="sim_model_label",
        )
        model_choice_map = {
            "XGBoost (Rekomendasi)": "xgboost",
            "Random Forest": "random_forest",
            "SVM": "svm",
        }
        st.session_state["sim_model_choice"] = model_choice_map[model_label]

        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

        # ── Sliders 6 polutan, dalam 2 kolom (urutan UI sama dengan sebelumnya) ──
        sc1, sc2 = st.columns(2, gap="medium")
        vals = {}
        with sc1:
            vals["pm25"] = _polutan_slider_block("pm25")
            st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
            vals["no2"]  = _polutan_slider_block("no2")
            st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
            vals["co"]   = _polutan_slider_block("co")
        with sc2:
            vals["pm10"] = _polutan_slider_block("pm10")
            st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
            vals["so2"]  = _polutan_slider_block("so2")
            st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
            vals["o3"]   = _polutan_slider_block("o3")

        # ── Sinkronisasi 2-arah: kalau user menggeser slider keluar preset,
        #    badge highlight hilang otomatis pada rerun berikutnya ──
        detected = _detect_active_preset(vals)
        if detected != st.session_state.get("sim_active_preset"):
            st.session_state["sim_active_preset"] = detected

        # ── Tombol Reset (Submit dihilangkan, hasil sudah realtime) ──
        st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
        bc1, bc2 = st.columns([1, 3])
        with bc1:
            st.markdown('<div class="reset-marker"></div>', unsafe_allow_html=True)
            st.button(
                "↺ Reset", key="btn_reset",
                type="secondary", use_container_width=True,
                on_click=reset_simulation,
                help="Kembalikan semua slider ke nilai default & hapus preset aktif.",
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # ────────────────────── KANAN: Hasil prediksi (REALTIME) ──────────────────────
    with col_right:
        # Hitung ISPU + ML tiap rerun berdasarkan nilai slider terkini.
        nilai_ispu, kategori, polutan_dominan, subindeks = calculate_ispu_category(
            pm10=vals["pm10"], pm25=vals["pm25"], so2=vals["so2"],
            co=vals["co"],   o3=vals["o3"],     no2=vals["no2"],
        )

        try:
            ml = prediksi_ispu_xgboost(
                pm10=vals["pm10"], pm25=vals["pm25"], so2=vals["so2"],
                co=vals["co"],   o3=vals["o3"],     no2=vals["no2"],
                model_choice=st.session_state.get("sim_model_choice", "xgboost"),
            )
            ml_kategori   = ml.get("kategori")
            ml_model_used = ml.get("model_used", "XGBoost")
            ml_confidence = ml.get("confidence")
        except Exception:
            ml_kategori = None
            ml_model_used = None
            ml_confidence = None

        info = KATEGORI_INFO[kategori]

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Hasil Prediksi ISPU</div>", unsafe_allow_html=True)

        # Badge preset aktif (kalau ada)
        active_name = st.session_state.get("sim_active_preset")
        if active_name:
            st.markdown(
                f"<div class='active-preset-badge'><span class='dot'></span>"
                f"Preset aktif: <strong>{active_name}</strong></div>",
                unsafe_allow_html=True,
            )

        # State netral = semua polutan 0 (mis. setelah klik Reset).
        # Saat netral, tetap tampilkan struktur penuh tapi dengan label
        # "Belum Ada Simulasi" supaya jelas ini bukan hasil yang valid.
        is_neutral = (nilai_ispu == 0)
        status_text = "Belum Ada Simulasi" if is_neutral else f"Udara {kategori}"
        deskripsi_text = (
            "Geser slider atau pilih preset untuk memulai simulasi."
            if is_neutral else info["deskripsi"]
        )
        rekom_text = (
            "Belum ada rekomendasi — silakan atur nilai polutan terlebih dahulu."
            if is_neutral else info["rekomendasi"]
        )

        # Hero result — angka ISPU + emoji + status
        st.markdown(
            f"""
            <div class='hasil-hero sim-fade'>
                <div>
                    <div class='hasil-num' style='color:{info["warna"]};'>{nilai_ispu:.0f}</div>
                    <div class='hasil-label-ispu'>ISPU</div>
                </div>
                <div>
                    <div style='font-size:2.5rem; line-height:1;'>{info["emoji"]}</div>
                    <div class='ispu-status' style='color:{info["warna"]}; margin-top:0.4rem;'>{status_text}</div>
                    <div class='ispu-desc'>{deskripsi_text}</div>
                </div>
            </div>

            <div class='rekom-box sim-fade' style='border:1px solid {info["warna"]}40; background:{info["warna_bg"]};'>
                <div class='rekom-box-title' style='color:{info["warna"]};'>Rekomendasi Aktivitas</div>
                <div class='rekom-box-text' style='color:#334155;'>{rekom_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Sub-indeks tiap polutan. Label "← dominan" tidak relevan saat
        # semua nilai 0, jadi disembunyikan dalam kondisi netral.
        rows_html = ""
        for pol, sub_val in sorted(subindeks.items(), key=lambda kv: -kv[1]):
            is_dom = (not is_neutral) and (pol == polutan_dominan)
            dom_cls = " dominan" if is_dom else ""
            tag = "  ← dominan" if is_dom else ""
            rows_html += (
                f"<div class='subindex-row{dom_cls}'>"
                f"<span class='label'>{POLUTAN_DISPLAY_NAME[pol]}{tag}</span>"
                f"<span class='val'>{sub_val:.1f}</span>"
                f"</div>"
            )
        st.markdown(
            "<div style='margin-top:1rem; font-size:0.8rem; color:#475569; font-weight:600; margin-bottom:0.4rem;'>"
            "Sub-Indeks per Polutan"
            "</div>" + rows_html,
            unsafe_allow_html=True,
        )

        # Pembanding klasifikasi Model ML
        if ml_kategori is not None:
            conf_txt = f" (keyakinan {ml_confidence*100:.1f}%)" if ml_confidence is not None else ""
            st.markdown(
                f"""
                <div class='info-box' style='margin-top:1rem;'>
                    <div class='info-box-icon'>ⓘ</div>
                    <div class='info-box-text'>
                        Nilai ISPU dihitung dengan formula sub-indeks PerMenLHK 14/2020.<br>
                        Klasifikasi model <strong>{ml_model_used}</strong>: <strong>{ml_kategori}</strong>{conf_txt}.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)


# ================================================================
# HALAMAN 4: EDUKASI & INSIGHT
# ================================================================
def page_edukasi(data):
    st.markdown(
        "<div class='page-title'>Edukasi & Insight</div>"
        "<div class='page-subtitle'>Pelajari kategori ISPU, dampak kesehatan, dan tips menjaga kualitas hidup saat polusi udara meningkat.</div>",
        unsafe_allow_html=True,
    )

    # Section 1: Mengenal ISPU + 5 kategori
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='card-title'>Mengenal ISPU (Indeks Standar Pencemar Udara)</div>"
        "<div style='font-size:0.88rem; color:#475569; margin-bottom:1.2rem; line-height:1.5;'>"
        "ISPU digunakan untuk menggambarkan kualitas udara ambien di sekitar kita."
        "</div>",
        unsafe_allow_html=True,
    )

    kc = st.columns(5, gap="small")
    for col, (nama, info) in zip(kc, KATEGORI_INFO.items()):
        with col:
            st.markdown(
                f"""
                <div class='kat-card' style='background:{info["warna_bg"]}; border-color:{info["warna"]}40;'>
                    <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
                        <div class='kat-range' style='color:{info["warna"]};'>{info["rentang"]}</div>
                        <div class='kat-emoji'>{info["emoji"]}</div>
                    </div>
                    <div class='kat-name' style='color:{info["warna"]};'>{nama}</div>
                    <div class='kat-desc'>{info["deskripsi"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)

    # Section 2: Dampak Kesehatan + Sumber Polusi
    dc1, dc2 = st.columns([1.4, 1], gap="medium")

    # Dampak Kesehatan
    with dc1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='card-title'>Dampak Kualitas Udara terhadap Kesehatan</div>",
            unsafe_allow_html=True,
        )

        dampak = [
            ("🫁", "Sistem Pernapasan", "Polusi udara dapat menyebabkan iritasi, batuk, sesak napas, dan memperparah asma."),
            ("❤️", "Sistem Kardiovaskular", "Paparan jangka panjang meningkatkan risiko penyakit jantung dan tekanan darah tinggi."),
            ("👶", "Anak-anak", "Anak lebih rentan terhadap infeksi pernapasan dan gangguan perkembangan paru-paru."),
            ("👴", "Lansia", "Risiko penyakit kronis meningkat, terutama jika memiliki riwayat penyakit."),
        ]
        dr1, dr2 = st.columns(2, gap="medium")
        for idx, (icon, judul, desc) in enumerate(dampak):
            with (dr1 if idx % 2 == 0 else dr2):
                st.markdown(
                    f"""
                    <div style='display:flex; gap:0.85rem; align-items:flex-start; margin-bottom:1.2rem;'>
                        <div style='font-size:1.8rem; flex-shrink:0; line-height:1;'>{icon}</div>
                        <div>
                            <div style='font-size:0.95rem; font-weight:700; color:#0F172A; margin-bottom:0.25rem;'>{judul}</div>
                            <div style='font-size:0.82rem; color:#475569; line-height:1.5;'>{desc}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

    # Sumber Polusi - donut chart
    with dc2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Sumber Polusi Udara di Jakarta</div>", unsafe_allow_html=True)

        sumber = {
            "Transportasi": (45, "#2563EB"),
            "Industri": (20, "#16A34A"),
            "Aktivitas Rumah Tangga": (15, "#F59E0B"),
            "Konstruksi": (10, "#EF4444"),
            "Lainnya": (10, "#7C3AED"),
        }

        chart_col, leg_col = st.columns([1, 1.1], gap="small")
        with chart_col:
            fig = go.Figure(go.Pie(
                labels=list(sumber.keys()),
                values=[v[0] for v in sumber.values()],
                hole=0.6,
                marker=dict(colors=[v[1] for v in sumber.values()],
                            line=dict(color="white", width=3)),
                textinfo="none",
                hovertemplate="<b>%{label}</b><br>%{value}%<extra></extra>",
            ))
            fig.update_layout(
                height=240,
                margin=dict(l=0, r=0, t=10, b=10),
                showlegend=False,
                paper_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with leg_col:
            st.markdown("<div style='padding-top:1rem;'>", unsafe_allow_html=True)
            for nama, (pct, warna) in sumber.items():
                st.markdown(
                    f"""
                    <div class='donut-legend-row'>
                        <div class='donut-legend-left'>
                            <div class='donut-legend-dot' style='background:{warna};'></div>
                            <span>{nama}</span>
                        </div>
                        <div class='donut-legend-pct'>{pct}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)

    # Section 3: Tips Kesehatan
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='card-title'>💡 Tips Menjaga Kesehatan Saat Kualitas Udara Tidak Sehat</div>",
        unsafe_allow_html=True,
    )

    tips = [
        ("😷",  "Gunakan Masker",       "Gunakan masker berstandar untuk mengurangi paparan polusi udara."),
        ("❌",  "Batasi Aktivitas Luar","Kurangi aktivitas fisik berat di luar ruangan, terutama saat sore hingga malam hari."),
        ("💨",  "Ventilasi yang Baik",  "Tutup jendela saat polusi tinggi dan pastikan ventilasi rumah tetap berfungsi baik."),
        ("💧",  "Perbanyak Minum Air",  "Cairan tubuh yang cukup membantu mengurangi efek polutan pada tubuh."),
        ("🌬️", "Gunakan Air Purifier", "Jika memungkinkan, gunakan alat penyaring udara di dalam ruangan untuk udara lebih bersih."),
    ]
    tc = st.columns(5, gap="medium")
    for col, (icon, judul, desc) in zip(tc, tips):
        with col:
            st.markdown(
                f"""
                <div style='background:#FFFFFF; border:1px solid #E2E8F0; border-radius:14px;
                            padding:1.1rem; height:100%; transition:all 0.25s ease;'>
                    <div style='font-size:2rem; color:#2563EB; margin-bottom:0.6rem; line-height:1;'>{icon}</div>
                    <div style='font-size:0.95rem; font-weight:700; color:#0F172A; margin-bottom:0.4rem;'>{judul}</div>
                    <div style='font-size:0.78rem; color:#64748B; line-height:1.5;'>{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


# ================================================================
# MAIN ROUTER
# ================================================================
def main():
    inject_css()
    data = load_data()
    page = render_sidebar()

    # Handle redirect dari tombol "Lihat Selengkapnya" di dashboard
    if st.session_state.get("jump_to_detail"):
        st.session_state["jump_to_detail"] = False
        page = "Detail Wilayah"

    if page == "Dashboard":
        page_dashboard(data)
    elif page == "Detail Wilayah":
        page_detail_wilayah(data)
    elif page == "Simulasi Prediksi ISPU":
        page_simulasi(data)
    elif page == "Edukasi & Insight":
        page_edukasi(data)


if __name__ == "__main__":
    main()
