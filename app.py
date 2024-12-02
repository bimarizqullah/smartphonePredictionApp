import pandas as pd
import numpy as np
import streamlit as st
import re
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Konfigurasi Streamlit
st.set_page_config(page_title="Prediksi Harga Smartphone", layout="wide")

st.title("📱 Prediksi Harga Smartphone")
st.sidebar.header("Pengaturan")

# Lokasi file CSV
csv_file_path = "phone_under_20K.csv"  # Ganti dengan nama file Anda

try:
    # Baca file CSV
    data = pd.read_csv(csv_file_path)
    st.write(f"Dataset memiliki {len(data)} baris dan {len(data.columns)} kolom.")
    
    rows_to_display = st.sidebar.slider("Pilih jumlah baris untuk ditampilkan", min_value=5, max_value=len(data), value=10)
    st.write("### Dataset yang Diunggah")
    st.dataframe(data.head(rows_to_display))

    # Bersihkan data
    data['Price'] = data['Price'].replace('[₹,]', '', regex=True).astype(float)

    # Ekstrak RAM dan ROM dari deskripsi
    def extract_ram(description):
        match = re.search(r'(\d+)\s?GB RAM', description)
        return int(match.group(1)) if match else None

    def extract_rom(description):
        match = re.search(r'(\d+)\s?GB ROM', description)
        return int(match.group(1)) if match else None

    data['RAM'] = data['description'].apply(extract_ram)
    data['ROM'] = data['description'].apply(extract_rom)

    # Bersihkan kolom yang tidak diperlukan
    data_cleaned = data.drop(columns=['Unnamed: 0', 'product_name', 'description'])

    # Tampilkan data yang telah diproses
    st.write("### Data yang Diproses")
    st.dataframe(data_cleaned.head(rows_to_display))

    # Pisahkan fitur dan target
    X = data_cleaned[['Rating', 'RAM', 'ROM']]
    y = data_cleaned['Price']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Latih model regresi linear
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prediksi pada test set
    y_pred = model.predict(X_test)

    # Evaluasi model
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write("### Evaluasi Model")
    st.write(f"- Mean Absolute Error: **{mae:.2f}**")
    st.write(f"- R² Score: **{r2:.2f}**")

    # Prediksi dengan input pengguna
    st.sidebar.subheader("Prediksi Harga")
    input_rating = st.sidebar.slider("Rating", min_value=1.0, max_value=5.0, step=0.1)
    input_ram = st.sidebar.selectbox("RAM (GB)", options=[2, 4, 6, 8, 12, 16])
    input_rom = st.sidebar.selectbox("ROM (GB)", options=[32, 64, 128, 256, 512, 1024])

    # Membuat prediksi harga
    if st.sidebar.button("Prediksi Harga"):
        prediction = model.predict(np.array([[input_rating, input_ram, input_rom]]))
        st.sidebar.write(f"### Harga Prediksi: ₹{prediction[0]:.2f}")
except FileNotFoundError:
    st.error(f"File CSV '{csv_file_path}' tidak ditemukan. Pastikan file berada di direktori yang benar.")
