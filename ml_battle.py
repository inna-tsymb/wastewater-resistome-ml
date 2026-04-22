import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc

print("Підготовка арени для битви алгоритмів...")

# 1. ЗАВАНТАЖЕННЯ ДАНИХ
X = pd.read_csv('X_features_cleaned.csv', index_col=0)
y = pd.read_csv('y_labels.csv', index_col=0).squeeze()

# Перетворюємо текстові лейбли (Europe, Asia) на числа (1 та 0)
# XGBoost вимагає, щоб таргет був числовим
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 2. РОЗБИТТЯ ДАНИХ
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# 3. ІНІЦІАЛІЗАЦІЯ МОДЕЛЕЙ
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'SVM (RBF Kernel)': SVC(probability=True, random_state=42, class_weight='balanced'),
    # Для XGBoost налаштовуємо баланс класів (відношення більшого класу до меншого)
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, 
                             scale_pos_weight=(sum(y_train==0)/sum(y_train==1)))
}

# 4. НАВЧАННЯ ТА ВІЗУАЛІЗАЦІЯ (ROC Curves)
plt.figure(figsize=(10, 8))

colors = {'Random Forest': '#2ecc71', 'SVM (RBF Kernel)': '#3498db', 'XGBoost': '#e74c3c'}

for name, model in models.items():
    print(f"Тренування {name}...")
    model.fit(X_train, y_train)
    
    # Отримуємо ймовірності приналежності до класу
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Рахуємо False Positive Rate, True Positive Rate та площу під кривою (AUC)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Малюємо криву для кожної моделі
    plt.plot(fpr, tpr, lw=2.5, color=colors[name], label=f'{name} (AUC = {roc_auc:.2f})')

# Малюємо діагональ випадкового вгадування
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Скільки разів помилились)', fontsize=12)
plt.ylabel('True Positive Rate (Скільки разів вгадали правильно)', fontsize=12)
plt.title('ROC Curves: Битва Алгоритмів (Прогнозування регіону за резистомом)', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Зберігаємо графік
plt.savefig('ml_battle_roc.png', dpi=300)
print("\nУСПІХ! Графік турніру збережено у файл 'ml_battle_roc.png'.")