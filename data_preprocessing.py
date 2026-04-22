import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

print("Починаємо обробку даних...")

# 1. ЗАВАНТАЖЕННЯ ДАНИХ
counts_df = pd.read_excel('data/41467_2019_8853_MOESM7_ESM.xlsx', index_col=0)
X_raw = counts_df.T 

# 2. ЗАВАНТАЖЕННЯ МЕТАДАНИХ (з правильним роздільником і назвою колонки!)
metadata_df = pd.read_csv('data/41467_2019_8853_MOESM4_ESM.txt', sep='\t', index_col='sample_ID')

# 3. ОБ'ЄДНАННЯ ТА ОЧИСТКА (використовуємо 'Continent' замість 'Region')
merged_data = X_raw.join(metadata_df['Continent'], how='inner')

# Залишаємо лише Європу та Азію
merged_data = merged_data[merged_data['Continent'].isin(['Europe', 'Asia'])]

# Відокремлюємо фічі (X) та цільову змінну (y)
y = merged_data['Continent']
X = merged_data.drop(columns=['Continent'])

# Заповнюємо NaN нулями (ген не знайдено)
X = X.fillna(0)

# ФІЛЬТРАЦІЯ
threshold = int(X.shape[0] * 0.05)
X = X.loc[:, (X > 0).sum(axis=0) >= threshold]

# 4. ТРАНСФОРМАЦІЯ ДЛЯ МЕТАГЕНОМІКИ
X_log = np.log1p(X)

# 5. ДЕМОНСТРАЦІЯ ТА ЗБЕРЕЖЕННЯ
print(f"Розмір фінальної матриці фіч (X): {X_log.shape}")
print(f"Розподіл по континентах:\n{y.value_counts()}")

# Зберігаємо очищені X та y у файли, щоб потім легко завантажити їх у модель
X_log.to_csv('X_features_cleaned.csv')
y.to_csv('y_labels.csv')

print("\nУСПІХ! Дані очищено, логарифмовано та збережено у файли 'X_features_cleaned.csv' та 'y_labels.csv'.")