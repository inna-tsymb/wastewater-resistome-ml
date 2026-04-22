import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings

# Вимикаємо зайві попередження pandas для чистоти терміналу
warnings.filterwarnings('ignore')

print("Починаємо епідеміологічний аналіз (AMR vs Світовий банк)...")

# 1. РАХУЄМО ЗАГАЛЬНЕ НАВАНТАЖЕННЯ РЕЗИСТЕНТНОСТІ (Total AMR Burden)
# Беремо наші очищені гени і сумуємо їх для кожного зразка
X = pd.read_csv('X_features_cleaned.csv', index_col=0)
total_amr = X.sum(axis=1).rename('Total_AMR_Burden')

# 2. ЗАВАНТАЖУЄМО ОСНОВНІ МЕТАДАНІ
metadata = pd.read_csv('data/41467_2019_8853_MOESM4_ESM.txt', sep='\t', index_col='sample_ID')
df = metadata.join(total_amr, how='inner')

# 3. ДОСЛІДЖУЄМО НОВИЙ ФАЙЛ СВІТОВОГО БАНКУ
wb_file_path = '/Users/inna.kucherova/Documents/metagenomic_project/data/41467_2019_8853_MOESM10_ESM.xlsx'
wb_extra = pd.read_excel(wb_file_path)
print(f"\nНовий документ успішно завантажено! Колонки всередині:")
print(wb_extra.columns.tolist())

# 4. ПОШУК КОРЕЛЯЦІЙ
# Знаходимо всі колонки Світового банку (починаються на 'wba_') у базі
wba_cols = [col for col in df.columns if str(col).startswith('wba_')]
print(f"\nШукаємо найсильніші зв'язки серед {len(wba_cols)} показників Світового банку...")

correlations = {}
for col in wba_cols:
    # Відкидаємо порожні значення
    temp = df[[col, 'Total_AMR_Burden']].dropna()
    
    # Конвертуємо в числа на випадок, якщо там є текст
    temp[col] = pd.to_numeric(temp[col], errors='coerce')
    temp = temp.dropna()
    
    # Рахуємо статистику лише там, де є хоча б 15 країн
    if len(temp) >= 15: 
        r, p = pearsonr(temp[col], temp['Total_AMR_Burden'])
        correlations[col] = {'r': r, 'p': p}

# Збираємо таблицю та фільтруємо значущі результати (p < 0.05)
corr_df = pd.DataFrame(correlations).T
corr_df = corr_df[corr_df['p'] < 0.05].sort_values(by='r', key=abs, ascending=False)

print("\nТОП-5 факторів, що визначають рівень резистентності в країні:")
print(corr_df.head(5))

# ==========================================
# 5. ВІЗУАЛІЗАЦІЯ (з виправленням DType)
# ==========================================
if len(corr_df) >= 3:
    top_3_factors = corr_df.head(3).index.tolist()
    
    # Словник для красивих підписів
    titles = {
        'wba_IC.TAX.PAYM': 'Рівень бюрократії (Податкові платежі)',
        'wba_IC.PRP.PROC': 'Складність реєстрації майна',
        'wba_DT.INT.PCBK.CD': 'Виплати за зовнішнім боргом'
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, factor in enumerate(top_3_factors):
        # ПРИМУСОВО перетворюємо текст на числа прямо в головній таблиці
        df[factor] = pd.to_numeric(df[factor], errors='coerce')
        
        # Відкидаємо порожні рядки перед малюванням, щоб seaborn не падав
        plot_data = df.dropna(subset=[factor, 'Total_AMR_Burden'])
        
        sns.regplot(data=plot_data, x=factor, y='Total_AMR_Burden', ax=axes[i], 
                    scatter_kws={'alpha':0.7, 's': 60, 'color':'#3498db', 'edgecolor': 'white'}, 
                    line_kws={'color':'#e74c3c', 'linewidth': 2.5})
        
        r_val = corr_df.loc[factor, "r"]
        nice_title = titles.get(factor, factor) # Беремо красиву назву або залишаємо код
        
        axes[i].set_title(f'AMR vs {nice_title}\n(Кореляція r = {r_val:.2f})', fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Загальне навантаження AMR (Log FPKM)', fontsize=10)
        axes[i].set_xlabel(f'Значення показника Світового банку', fontsize=10)
        axes[i].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('epi_amr_correlation.png', dpi=300)
    print("\nУСПІХ! Графіки кореляції збережено у файл 'epi_amr_correlation.png'.")
else:
    print("\nНе знайдено достатньо значущих кореляцій.")