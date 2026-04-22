import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

print("Запуск крос-континентальної валідації...")

# 1. ЗАВАНТАЖЕННЯ ДАНИХ
counts_df = pd.read_excel('data/41467_2019_8853_MOESM7_ESM.xlsx', index_col=0)
X_raw = counts_df.T
metadata_df = pd.read_csv('data/41467_2019_8853_MOESM4_ESM.txt', sep='\t', index_col='sample_ID')

merged_data = X_raw.join(metadata_df['Continent'], how='inner')

# 2. РОЗДІЛЯЄМО НА "ВІДОМІ" ТА "НЕВІДОМІ" РЕГІОНИ
known_regions = ['Europe', 'Asia']
df_known = merged_data[merged_data['Continent'].isin(known_regions)]
# Відкидаємо також NaN, якщо в когось не вказано континент
df_unknown = merged_data[~merged_data['Continent'].isin(known_regions) & merged_data['Continent'].notna()]

# 3. ПІДГОТОВКА ТРЕНУВАЛЬНИХ ДАНИХ (ЄВРОПА ТА АЗІЯ)
y_train = df_known['Continent']
X_train_raw = df_known.drop(columns=['Continent']).fillna(0)

# Залишаємо лише гени, які є хоча б у 5% зразків Європи/Азії (наші 323 фічі)
threshold = int(X_train_raw.shape[0] * 0.05)
valid_genes = X_train_raw.columns[(X_train_raw > 0).sum(axis=0) >= threshold]

X_train = np.log1p(X_train_raw[valid_genes])

# 4. НАВЧАННЯ МОДЕЛІ (Використовуємо Random Forest як найстабільніший)
print(f"Тренуємо модель на {len(X_train)} зразках з Європи та Азії...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

# 5. ПІДГОТОВКА ДАНИХ ДЛЯ ІНШИХ КОНТИНЕНТІВ
# ВАЖЛИВО: залишаємо ТІЛЬКИ ті самі гени (valid_genes), які бачила модель!
X_unknown_raw = df_unknown.drop(columns=['Continent']).fillna(0)
X_unknown = np.log1p(X_unknown_raw[valid_genes])
actual_continents = df_unknown['Continent']

# 6. ПРОГНОЗУВАННЯ
print(f"Робимо прогнози для {len(X_unknown)} зразків з інших континентів...\n")
predictions = rf.predict(X_unknown)

# 7. ЗБІР ТА АНАЛІЗ РЕЗУЛЬТАТІВ
results = pd.DataFrame({
    'Справжній Континент': actual_continents,
    'Прогноз Моделі': predictions
})

print("================ РЕЗУЛЬТАТИ ВАЛІДАЦІЇ ================")
for continent in results['Справжній Континент'].unique():
    subset = results[results['Справжній Континент'] == continent]
    total = len(subset)
    europe_votes = sum(subset['Прогноз Моделі'] == 'Europe')
    asia_votes = sum(subset['Прогноз Моделі'] == 'Asia')
    
    print(f"📍 {continent} (Всього зразків: {total})")
    print(f"   Схоже на Європу (низький AMR): {europe_votes} ({europe_votes/total*100:.1f}%)")
    print(f"   Схоже на Азію (високий AMR):   {asia_votes} ({asia_votes/total*100:.1f}%)\n")