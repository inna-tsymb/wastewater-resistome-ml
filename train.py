import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA

print("Завантаження підготовлених даних...")

# 1. ЗАВАНТАЖЕННЯ ДАНИХ
X = pd.read_csv('X_features_cleaned.csv', index_col=0)
y = pd.read_csv('y_labels.csv', index_col=0).squeeze() 

# 2. РОЗБИТТЯ НА ТРАЙН І ТЕСТ
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. НАВЧАННЯ МОДЕЛІ
print("Навчання моделі Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Оцінка
y_pred = rf_model.predict(X_test)
print(f"\nЗагальна точність (Accuracy): {accuracy_score(y_test, y_pred) * 100:.1f}%")

# 4. ПОШУК БІОМАРКЕРІВ
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importances = feature_importances.sort_values(ascending=False)
top_15_genes = feature_importances.head(15)

# ==========================================
# 5. КОМПЛЕКСНА ВІЗУАЛІЗАЦІЯ (3 Графіки)
# ==========================================
print("\nГенеруємо комплексні графіки...")

# Створюємо полотно для 3 графіків в один ряд
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- Графік 1: Важливість маркерів (Barplot) ---
sns.barplot(x=top_15_genes.values, y=top_15_genes.index, hue=top_15_genes.index, legend=False, palette='viridis', ax=axes[0])
axes[0].set_title('ТОП-15 генів-маркерів', fontsize=12)
axes[0].set_xlabel('Важливість для моделі', fontsize=10)
axes[0].set_ylabel('Ген', fontsize=10)
axes[0].grid(axis='x', linestyle='--', alpha=0.7)

# --- Графік 2: PCA на ТОП-15 генах (Скаттерплот) ---
pca = PCA(n_components=2)
X_test_top15 = X_test[top_15_genes.index]
pca_result = pca.fit_transform(X_test_top15)

# Задаємо кольори для континентів (синій для Європи, червоний для Азії)
palette_colors = {'Europe': '#3498db', 'Asia': '#e74c3c'}

sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=y_test, palette=palette_colors, s=100, alpha=0.8, ax=axes[1])
axes[1].set_title('PCA: Тестова вибірка\n(Лише 15 генів-маркерів)', fontsize=12)
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
axes[1].grid(True, linestyle='--', alpha=0.7)
axes[1].legend(title='Continent')

# --- Графік 3: Експресія маркерів (Heatmap) ---
# Збираємо дані тесту разом з лейблами та сортуємо, щоб розділити континенти візуально
test_data_combined = X_test_top15.copy()
test_data_combined['Continent'] = y_test
test_data_combined = test_data_combined.sort_values('Continent', ascending=False) # Спочатку Europe, потім Asia

# Готуємо матрицю для хітмапу (гени в рядки, зразки в стовпці)
heatmap_data = test_data_combined.drop(columns=['Continent']).T 

sns.heatmap(heatmap_data, cmap='coolwarm', ax=axes[2], cbar_kws={'label': 'Log(FPKM + 1)'})
axes[2].set_title('Експресія маркерів (Test)', fontsize=12)
axes[2].set_xlabel('Зразки (Європа -> Азія)', fontsize=10)
axes[2].set_xticks([]) # Ховаємо назви конкретних зразків, щоб не захаращувати вісь

# Фінальне компонування та збереження
plt.tight_layout()
plt.savefig('ml_resistome_full_dashboard.png', dpi=300)
print("УСПІХ! Новий дашборд збережено у файл 'ml_resistome_full_dashboard.png'.")