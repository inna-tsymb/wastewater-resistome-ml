import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Global AMR Dashboard", layout="wide")

st.title("🌍 Глобальний аналіз метагеномів стічних вод")

# 1. Завантаження даних
@st.cache_data
def load_data():
    # Твій датасет
    df = pd.read_excel('data/41467_2019_8853_MOESM7_ESM.xlsx', index_col=0).T
    metadata = pd.read_csv('data/41467_2019_8853_MOESM4_ESM.txt', sep='\t', index_col='sample_ID')
    return df.join(metadata['Continent'])

data = load_data()

# 2. Фільтр
continents = st.multiselect("Виберіть регіони для порівняння:", options=data['Continent'].unique(), default=['Europe', 'Asia'])
filtered_data = data[data['Continent'].isin(continents)]

# 3. Візуалізація
st.subheader("Розподіл генів стійкості")
gene = st.selectbox("Виберіть ген або кластер:", options=filtered_data.columns[:-1])

fig = px.box(filtered_data, x="Continent", y=gene, color="Continent", title=f"Розподіл {gene} за регіонами")
st.plotly_chart(fig, use_container_width=True)

st.write("Це дозволяє побачити, чи є статистично значуща різниця в навантаженні конкретним геном між континентами.")