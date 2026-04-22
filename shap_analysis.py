import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

print("Ініціалізація SHAP-аналізу (Відкриваємо 'чорну скриньку' ШІ)...")

# 1. ЗАВАНТАЖЕННЯ ДАНИХ
X = pd.read_csv('X_features_cleaned.csv', index_col=0)
y = pd.read_csv('y_labels.csv', index_col=0).squeeze()

# Перетворюємо лейбли (Asia/Europe) на числа
le = LabelEncoder()
y_encoded = le.fit_transform(y)
# Дізнаємося, який індекс отримав клас 'Asia'
asia_class_index = list(le.classes_).index('Asia')

# 2. НАВЧАННЯ МОДЕЛІ
print("Навчання моделі Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X, y_encoded)

# 3. SHAP АНАЛІЗ
print("Обчислення математичних ваг SHAP для кожного гена (це може зайняти кілька секунд)...")
# Використовуємо TreeExplainer, який ідеально оптимізований для випадкових лісів
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Для Random Forest shap_values зазвичай є списком масивів (по одному для кожного класу).
# Ми беремо масив для класу "Asia", щоб зрозуміти, що саме тягне прогноз у бік високого AMR
if isinstance(shap_values, list):
    shap_values_to_plot = shap_values[asia_class_index]
elif len(shap_values.shape) == 3:
    shap_values_to_plot = shap_values[:, :, asia_class_index]
else:
    shap_values_to_plot = shap_values

# 4. ВІЗУАЛІЗАЦІЯ (SHAP Summary Plot)
plt.figure(figsize=(10, 8))

# Малюємо графік. Параметр show=False дозволяє нам зберегти його у файл замість виведення на екран
shap.summary_plot(shap_values_to_plot, X, show=False, max_display=15, 
                  plot_size=(10, 8), cmap="coolwarm")

plt.title('SHAP Аналіз: Що формує Азійський профіль (високий AMR)?\n(Червоний колір = висока кількість гена)', 
          fontsize=14, pad=15)
plt.tight_layout()

# Зберігаємо графік
plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
print("\nУСПІХ! Графік SHAP збережено у файл 'shap_summary.png'.")