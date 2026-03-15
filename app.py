import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pmdarima import auto_arima

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Sales Forecast Dashboard",
    layout="wide"
)

# ==============================
# CUSTOM CSS (BLACK PINK THEME)
# ==============================
st.markdown("""
<style>

body {
    background-color: #0f0f0f;
}

.stApp {
    background-color: #0f0f0f;
    color: white;
}

h1,h2,h3,h4 {
    color: #ff4da6;
}

section[data-testid="stSidebar"] {
    background-color: #1c1c1c;
}

section[data-testid="stSidebar"] .css-1v0mbdj p {
    font-size: 20px;
}

.sidebar-title {
    font-size: 28px;
    font-weight: bold;
    color: #ff4da6;
    margin-bottom: 20px;
}

.stButton>button {
    background-color:#ff4da6;
    color:white;
}

</style>
""", unsafe_allow_html=True)

st.title("📊 Dashboard Forecasting Penjualan")

# ==============================
# LOAD DATASET DARI FOLDER LOKAL
# ==============================

@st.cache_data
def load_data():
    df = pd.read_excel("Forecasting.xlsx")
    return df

df = load_data()

st.subheader("🔍 Dataset Preview")
st.dataframe(df.head())

# ==============================
# PREPROCESSING
# ==============================
df['Date'] = pd.to_datetime(df['Date'])
df = df.dropna(subset=['Date'])
df.set_index('Date', inplace=True)

# ==============================
# SIDEBAR MENU
# ==============================

st.sidebar.markdown('<div class="sidebar-title">Dashboard Menu</div>', unsafe_allow_html=True)

menu = st.sidebar.radio(
    "Pilih Menu",
    [
        "📂 Dataset Info",
        "📊 Exploratory Data Analysis",
        "🏪 Forecast per Store",
        "📈 Forecast Total"
    ]
)

    # ==============================
    # DATASET INFO
    # ==============================
if menu == "📂 Dataset Info":

    st.subheader("📋 Statistik Deskriptif")
    st.write(df.describe())

    # ==============================
    # EDA
    # ==============================
if menu == "📊 Exploratory Data Analysis":

    st.subheader("📈 Distribusi Weekly Sales")

    fig, ax = plt.subplots()
    sns.histplot(df['Weekly_Sales'], bins=50, kde=True, color="pink")
    st.pyplot(fig)

    st.subheader("💰 Total Sales Semua Store")

    total_sales = df.groupby('Date')['Weekly_Sales'].sum()

    fig, ax = plt.subplots()
    ax.plot(total_sales, color="pink")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Total Weekly Sales")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("🖇️ Korelasi Feature")

    feature_cols = ['Holiday_Flag','Fuel_Price','CPI','Unemployment']
    corr = df[['Weekly_Sales'] + feature_cols].corr()

    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="RdPu")
    st.pyplot(fig)
    st.subheader("💵 Total Penjualan per Store")

    store_sales = df.groupby('Store')['Weekly_Sales'].sum()

    fig, ax = plt.subplots(figsize=(12,6))

    sns.barplot(
        x=store_sales.index.astype(str),
        y=store_sales.values,
        palette="RdPu",
        ax=ax
    )

    plt.xticks(rotation=45)  # memiringkan label store
    ax.set_xlabel("Store")
    ax.set_ylabel("Total Weekly Sales")

    plt.tight_layout()
    st.pyplot(fig)

# ==============================
# FORECAST PER STORE
# ==============================
if menu == "🏪 Forecast per Store":

    st.subheader("🔮 Forecast per Store")

    store_id = st.selectbox("Pilih Store", df['Store'].unique())

    store_df = df[df['Store'] == store_id]
    monthly_sales = store_df['Weekly_Sales'].resample('M').sum()
    model = auto_arima(monthly_sales, seasonal=True, m=12)
    forecast = model.predict(n_periods=6)

    forecast_index = pd.date_range(
        start=monthly_sales.index[-1],
        periods=6,
        freq='M'
    )
        
    # titik terakhir data historis
    last_date = monthly_sales.index[-1]
    last_value = monthly_sales.iloc[-1]

    # gabungkan titik terakhir historis dengan forecast
    forecast_dates = forecast_index
    forecast_values = forecast

    connect_dates = [last_date] + list(forecast_dates)
    connect_values = [last_value] + list(forecast_values)

    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(monthly_sales, label="Historical", color="pink")
    ax.plot(connect_dates, connect_values, "--", label="Forecast", color="red")

    ax.legend()

    st.pyplot(fig)

    forecast_df = pd.DataFrame({
        "Date":forecast_index,
        "Forecast":forecast
    })

    st.dataframe(forecast_df)

# ==============================
# FORECAST TOTAL
# ==============================
if menu == "📈 Forecast Total":

    st.subheader("🔮 Forecast Total Semua Store")

    total_sales = df.groupby('Date')['Weekly_Sales'].sum().resample('M').sum()

    model = auto_arima(total_sales, seasonal=True, m=12)

    forecast = model.predict(n_periods=6)

    forecast_index = pd.date_range(
        start=total_sales.index[-1],
        periods=6,
        freq='M'
    )

    # titik terakhir data historis
    last_date = total_sales.index[-1]
    last_value = total_sales.iloc[-1]

    # gabungkan titik historis dengan forecast
    connect_dates = [last_date] + list(forecast_index)
    connect_values = [last_value] + list(forecast)
    fig, ax = plt.subplots(figsize=(10,5))

    # garis historis
    ax.plot(total_sales, label="Historical", color="pink")

    # garis forecast + penghubung
    ax.plot(connect_dates, connect_values, "--", label="Forecast", color="red")

    ax.legend()

    st.pyplot(fig)

    forecast_df = pd.DataFrame({
        "Date": forecast_index,
        "Forecast": forecast
    })

    st.dataframe(forecast_df)

# ==============================
# INTERPRETASI
# ==============================

    st.subheader("📝 Interpretasi Forecast")

    with st.expander("Lihat Interpretasi Forecast"):

        st.markdown("""
        ### Tren Historis
        - Penjualan historis dari **2010 hingga akhir 2012** menunjukkan **fluktuasi yang cukup besar setiap bulan**.
        - Terdapat beberapa **puncak penjualan dan penurunan tajam**, yang menunjukkan dinamika permintaan yang berubah-ubah.
        - Pola ini kemungkinan dipengaruhi oleh **harga bahan bakar, hari libur, atau faktor ekonomi lainnya**.

        ### Forecast 6 Bulan ke Depan
        - **Januari 2013** diprediksi berada di sekitar **203 juta**, konsisten dengan tren historis sebelumnya.
        - **Februari 2013** sedikit meningkat menjadi sekitar **209 juta**, menunjukkan awal tahun yang stabil.
        - **Maret–April 2013** diperkirakan menurun menjadi sekitar **166–177 juta**, mengikuti pola penurunan setelah awal tahun.
        - **Mei 2013** diprediksi naik kembali hingga sekitar **201 juta**, kemungkinan karena faktor promosi atau musiman.
        - **Juni 2013** diperkirakan turun menjadi sekitar **149 juta**, yang menjadi titik terendah dalam periode forecast.

        ### Pola Musiman
        Grafik menunjukkan bahwa meskipun total penjualan bersifat fluktuatif, model forecasting mampu **mengikuti pola musiman historis**, dengan puncak di awal tahun dan penurunan di pertengahan tahun.

        ### Kesimpulan
        Forecast memberikan gambaran yang cukup realistis mengenai **potensi penjualan dalam 6 bulan ke depan**. Informasi ini dapat digunakan oleh manajemen untuk:
        - merencanakan **stok barang**
        - menyusun **strategi promosi**
        - mempersiapkan **operasional toko secara lebih optimal**.
        """)
