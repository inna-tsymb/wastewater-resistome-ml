import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from itertools import combinations

print("Завантаження даних для мережевого аналізу...")

# 1. ЗАВАНТАЖЕННЯ ДАНИХ
# Використовуємо нашу матрицю фіч (гени)
X = pd.read_csv('X_features_cleaned.csv', index_col=0)

# 2. НАЛАШТУВАННЯ ПАРАМЕТРІВ МЕРЕЖІ
# Встановлюємо жорсткі пороги, щоб відсіяти шум
CORR_THRESHOLD = 0.70  # Мінімальна сила зв'язку (r)
P_VALUE_THRESHOLD = 0.05 # Статистична значущість (p-value)

print(f"Розрахунок кореляцій Спірмена для {X.shape[1]} генів (це може зайняти кілька секунд)...")

# 3. ПОШУК ЗНАЧУЩИХ ЗВ'ЯЗКІВ (Edges)
edges = []
genes = X.columns.tolist()

# Перебираємо всі можливі пари генів
for gene1, gene2 in combinations(genes, 2):
    # Рахуємо кореляцію Спірмена та p-value для пари
    corr, p_val = spearmanr(X[gene1], X[gene2])
    
    # Якщо зв'язок сильний і значущий — додаємо до мережі
    if abs(corr) >= CORR_THRESHOLD and p_val <= P_VALUE_THRESHOLD:
        edges.append({
            'source': gene1,
            'target': gene2,
            'weight': round(corr, 3),
            'p_value': p_val
        })

edges_df = pd.DataFrame(edges)
print(f"Знайдено {len(edges_df)} сильних зв'язків!")

# 4. ПОБУДОВА ГРАФА
G = nx.Graph()

# Додаємо ребра (вузли додадуться автоматично)
for _, row in edges_df.iterrows():
    G.add_edge(row['source'], row['target'], weight=row['weight'])

# Видаляємо ізольовані вузли (гени без сильних зв'язків), щоб очистити картинку
isolated_nodes = list(nx.isolates(G))
G.remove_nodes_from(isolated_nodes)

print(f"У фінальній мережі залишилось {G.number_of_nodes()} генів.")

# ==========================================
# 5. ВІЗУАЛІЗАЦІЯ В PYTHON
# ==========================================
plt.figure(figsize=(14, 14))

# Визначаємо макет (layout) для красивого розташування вузлів
# spring_layout імітує фізику: зв'язані вузли притягуються один до одного
pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

# Розмір вузла залежить від того, скільки зв'язків він має (degree)
node_sizes = [v * 100 for k, v in dict(G.degree()).items()]

# Малюємо вузли
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='#3498db', alpha=0.8, edgecolors='white')

# Малюємо ребра (товщина залежить від сили кореляції)
edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color='#95a5a6', alpha=0.5)

# Малюємо підписи (назви генів)
nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif', font_weight='bold')

plt.title(f'Ко-окурентна мережа генів резистентності (AMR)\n(Spearman r > {CORR_THRESHOLD}, p < {P_VALUE_THRESHOLD})', fontsize=16)
plt.axis('off') # Ховаємо осі координат
plt.tight_layout()

# Зберігаємо графік
plt.savefig('amr_network_plot.png', dpi=300)
print("\nУСПІХ! Мережевий графік збережено у файл 'amr_network_plot.png'.")

# ==========================================
# 6. ЕКСПОРТ ДЛЯ CYTOSCAPE (Професійний рівень)
# ==========================================
nx.write_graphml(G, "amr_network_cytoscape.graphml")
print("УСПІХ! Файл 'amr_network_cytoscape.graphml' збережено. Ти можеш відкрити його в програмі Cytoscape для просунутої візуалізації та аналізу топології.")