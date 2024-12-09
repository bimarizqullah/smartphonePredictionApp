import pandas as pd
import numpy as np
import streamlit as st
import re
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Konfigurasi Streamlit
st.set_page_config(page_title="Prediksi Harga Smartphone", layout="wide")

# Navigasi Halaman
page = st.sidebar.selectbox("Navigasi", ["Beranda", "Prediksi", "Visualisasi Data", "Tentang & Kredit"])

if page == "Beranda":
    st.title("\ud83d\udcf1 Prediksi Harga Smartphone")
    st.image("image.png", width=500, caption="Temukan Harga Smartphone Berdasarkan Spesifikasi")
    st.write("""
        Selamat datang di aplikasi *Prediksi Harga Smartphone*!  
        Aplikasi ini dirancang untuk membantu Anda memperkirakan harga smartphone berdasarkan spesifikasi seperti **Rating**, **RAM**, dan **ROM**.  
        Dengan menggunakan model **Regresi Linear**, Anda dapat melihat harga prediksi berdasarkan input yang diberikan.
    """)
    st.write("""
        ### Fitur Utama:
        - **Eksplorasi Dataset**: Lihat dan analisis data yang digunakan dalam prediksi.
        - **Prediksi Harga**: Dapatkan estimasi harga smartphone berdasarkan input Anda.
        - **Evaluasi Model**: Tampilkan akurasi model dalam bentuk **Mean Absolute Error** dan **R² Score**.
    """)

elif page == "Prediksi":
    st.title("\ud83d\udcca Prediksi Harga Smartphone")

    # Lokasi file CSV
    csv_file_path = "phone_under_20K.csv"  # Ganti dengan nama file Anda

    try:
        # Baca file CSV
        data = pd.read_csv(csv_file_path)
        st.write(f"Dataset memiliki {len(data)} baris dan {len(data.columns)} kolom.")

        rows_to_display = st.slider("Pilih jumlah baris untuk ditampilkan", min_value=5, max_value=len(data), value=10)
        st.write("### Dataset yang Diunggah")
        st.dataframe(data.head(rows_to_display))

        # Bersihkan data
        data['Price'] = data['Price'].replace('[\u20ac,]', '', regex=True).astype(float)

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
        st.write(f"- Mean Absolute Error: **{mae:.2f} Euro**")
        st.write(f"- R\u00b2 Score: **{r2:.2f}**")

        # Prediksi dengan input pengguna
        st.write("### Prediksi Harga Berdasarkan Input")
        col1, col2, col3 = st.columns(3)

        with col1:
            input_rating = st.slider("Rating", min_value=1.0, max_value=5.0, step=0.1)
        with col2:
            input_ram = st.selectbox("RAM (GB)", options=[2, 4, 6, 8, 12, 16])
        with col3:
            input_rom = st.selectbox("ROM (GB)", options=[32, 64, 128, 256, 512, 1024])

        if st.button("Prediksi Harga"):
            # Nilai tukar Euro ke Rupiah
            euro_to_idr = 16000  # Ganti dengan nilai tukar yang relevan

            # Prediksi harga dalam Euro
            prediction = model.predict(np.array([[input_rating, input_ram, input_rom]]))

            # Konversi hasil prediksi ke Rupiah
            prediction_idr = prediction[0] * euro_to_idr

            # Tampilkan hasil prediksi dalam Rupiah
            st.write(f"### Harga Prediksi: Rp {prediction_idr:,.2f}")
    except FileNotFoundError:
        st.error(f"File CSV '{csv_file_path}' tidak ditemukan. Pastikan file berada di direktori yang benar.")

elif page == "Visualisasi Data":
    st.title("\ud83d\udcca Visualisasi Data Smartphone")

    try:
        # Baca file CSV
        data = pd.read_csv("phone_under_20K.csv")

        # Bersihkan data
        data['Price'] = data['Price'].replace('[\u20ac,]', '', regex=True).astype(float)

        # Ekstrak RAM dan ROM
        data['RAM'] = data['description'].apply(lambda x: int(re.search(r'(\d+)\s?GB RAM', x).group(1)) if re.search(r'(\d+)\s?GB RAM', x) else None)
        data['ROM'] = data['description'].apply(lambda x: int(re.search(r'(\d+)\s?GB ROM', x).group(1)) if re.search(r'(\d+)\s?GB ROM', x) else None)

        # Sort data for proper line chart display
        data_sorted = data.sort_values(by='Price')

        # Line chart untuk hubungan RAM dan Harga
        st.write("### Line Chart RAM vs Harga")
        st.line_chart(data_sorted[['RAM', 'Price']].dropna())

        # Line chart untuk hubungan ROM dan Harga
        st.write("### Line Chart ROM vs Harga")
        st.line_chart(data_sorted[['ROM', 'Price']].dropna())

        # Line chart untuk hubungan Rating dan Harga
        st.write("### Line Chart Rating vs Harga")
        st.line_chart(data_sorted[['Rating', 'Price']].dropna())

    except FileNotFoundError:
        st.error("File CSV tidak ditemukan. Pastikan file berada di direktori yang benar.")

elif page == "Tentang & Kredit":
    st.title("\u2139\ufe0f Tentang Aplikasi")
    st.write("""
        Aplikasi ini dikembangkan untuk memberikan solusi prediksi harga smartphone dengan memanfaatkan dataset publik 
        dan teknik **Regresi Linear**.  
        Kami berharap aplikasi ini dapat membantu pengguna dalam memahami dan menganalisis faktor-faktor yang mempengaruhi harga smartphone.
    """)
    st.write("""
        ### Kredit:
        - **Pengembang**: Bima Cahya, Dafa Surya, & Walid Habibi
        - **Sumber Data**: Kaggle.com 
        - **Framework**: Streamlit
        - **Model ML**: Scikit-learn
    """)
