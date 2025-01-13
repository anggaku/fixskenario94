import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.ensemble import VotingRegressor

# Judul aplikasi
st.title("Prediksi RAB dengan Voting Regressor 94 Row")
# Judul aplikasi
st.text("R2 Score: 1.000, Train 80%, Testing 20%")
st.text("Ensemble Voting Regressor: Bayesian Ridge dan Linear Regression")

# Input data dari pengguna
st.subheader("Masukkan Data Proyek")	

# Form input untuk data baru
with st.form("input_form"):
    namaproyek = st.selectbox("Nama Proyek", ['Gedung Apartemen'])
    waktu = st.number_input("Waktu (dalam hari)", min_value=1, step=1)
    provinsi = st.selectbox("Provinsi", ["Bali", "Banten", "DKI Jakarta", "Jawa Barat", "Jawa Timur", "Kalimantan Timur", "Sulawesi Selatan"])
    tahun = st.number_input("Tahun", min_value=2000, max_value=2100, step=1)
    luas = st.number_input("Luas (m²)", min_value=0.0, step=1.0)
    subitem = st.number_input("Subitem", min_value=1, step=1)
    tinggi = st.number_input("Tinggi (m)", min_value=0.0, step=0.1)
    lantai = st.number_input("Jumlah Lantai", min_value=1, step=1)
    ikk = st.number_input("IKK", min_value=0.0, step=0.01)
    ihbp = st.number_input("IHBP", min_value=0.0, step=0.01)
    inflasi = st.number_input("Inflasi (%)", min_value=0.0, step=0.01)
    
    # Tombol submit
    submit = st.form_submit_button("Prediksi")

if submit:
    # Membuat DataFrame dari input pengguna
    data_baru = pd.DataFrame({
        'namaproyek': [namaproyek],
        'waktu': [waktu],
        'provinsi': [provinsi],
        'tahun': [tahun],
        'luas': [luas],
        'subitem': [subitem],
        'tinggi': [tinggi],
        'lantai': [lantai],
        'ikk': [ikk],
        'ihbp': [ihbp],
        'inflasi': [inflasi]
    })

    # Transformasi log pada kolom numerikal
    numerical_cols = ['waktu', 'tahun', 'luas', 'subitem', 'tinggi', 'lantai', 'ikk', 'ihbp', 'inflasi']
    data_transformed = data_baru.copy()
    data_transformed[numerical_cols] = np.log1p(data_transformed[numerical_cols])

    # Muat encoder yang telah disimpan
    encoder_namaproyek = load('../model/encoder_namaproyek.pkl')
    encoder_provinsi = load('../model/encoder_provinsi.pkl')

    # Transformasi kolom kategorikal
    data_transformed['label_provinsi'] = encoder_provinsi.transform(data_transformed['provinsi'])
    data_transformed['label_namaproyek'] = encoder_namaproyek.transform(data_transformed['namaproyek'])

    # Muat scaler yang telah disimpan
    scaler = load('../model/scaler.pkl')
    data_transformed[numerical_cols] = scaler.transform(data_transformed[numerical_cols])

    # Hapus kolom yang tidak diperlukan lagi
    data_transformed.drop(columns=['lantai', 'namaproyek', 'provinsi'], inplace=True)

    # Muat model Voting Regressor yang telah disimpan
    voting_regressor2 = load('../model/voting_regressor2_model.joblib')

    # Prediksi hasil menggunakan data baru
    y_pred_baru = voting_regressor2.predict(data_transformed)

    # Invers transformasi log (kembali ke nilai asli)
    y_pred_asli = np.expm1(y_pred_baru)

    # Tambahkan hasil ke DataFrame asli
    data_baru['Hasil Prediksi'] = y_pred_asli

    # Konversi hasil prediksi ke dalam bentuk Rupiah
    data_baru['Hasil Prediksi (Rupiah)'] = data_baru['Hasil Prediksi'].apply(
        lambda x: f"Rp {x:,.2f}".replace(',', '.').replace('.', ',', 1)
    )

    # Tampilkan hasil prediksi
    st.subheader("Hasil Prediksi")
    st.write(data_baru)
