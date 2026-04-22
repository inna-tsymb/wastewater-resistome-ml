import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Налаштування сторінки
st.set_page_config(page_title="AMR Global Dashboard", layout="wide")
st.title("🌍 Глобальний нагляд за антимікробною резистентністю")
st.markdown("Інтерактивний дашборд для аналізу метагеномів стічних вод. Досліджуйте ТОП-маркери стійкості у різних регіонах світу.")

# 1. ЗАВАНТАЖЕННЯ ДАНИХ (Кешуємо для швидкості)
@st.cache_data
def load_data():
    counts_df = pd.read_excel('data/41467_2019_8853_MOESM7_ESM.xlsx', index_col=0)
    X_raw = counts_df.T 
    metadata_df = pd.read_csv('data/41467_2019_8853_MOESM4_ESM.txt', sep='\t', index_col='sample_ID')
    
    merged = X_raw.join(metadata_df['Continent'], how='inner')
    merged = merged[merged['Continent'].notna()]
    
    # Логарифмуємо для красивих графіків
    continents = merged['Continent']
    X_log = np.log1p(merged.drop(columns=['Continent']).fillna(0))
    X_log['Continent'] = continents
    return X_log

with st.spinner('Завантаження масивів даних...'):
    df = load_data()

# ТОП-10 генів, які ми знайшли на попередньому кроці
top_genes = [
    'tet(G)', 'tet(A)', 'blaOXA_clust7', 'sul1_sul3_clust', 
    'blaTLA-1', 'mph(A)', 'tet(Q)', 'erm(G)_clust', 'cml_clust', 'sul2_clust'
]

# 2. ПАНЕЛЬ КЕРУВАННЯ (Sidebar)
st.sidebar.header("Налаштування аналізу")
selected_gene = st.sidebar.selectbox("Оберіть ген-маркер для аналізу:", top_genes)

all_continents = df['Continent'].unique().tolist()
selected_continents = st.sidebar.multiselect(
    "Оберіть регіони:", 
    options=all_continents, 
    default=all_continents
)

# 3. ВІЗУАЛІЗАЦІЯ
if selected_continents:
    filtered_df = df[df['Continent'].isin(selected_continents)]
    
    # Boxplot для порівняння експресії
    fig = px.box(
        filtered_df, x='Continent', y=selected_gene, color='Continent',
        title=f"Розподіл експресії гена {selected_gene} за регіонами",
        labels={selected_gene: 'Рівень експресії Log(FPKM + 1)', 'Continent': 'Континент'},
        template="plotly_white", points="all"
    )
    fig.update_layout(showlegend=False, title_font_size=20)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info(f"**Біологічний контекст:** Високий рівень маркера **{selected_gene}** вказує на інтенсивний селективний тиск антибіотиків цього класу в популяції.")
else:
    st.warning("Будь ласка, оберіть хоча б один регіон у боковому меню.")