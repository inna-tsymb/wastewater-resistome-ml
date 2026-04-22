# =========================================================================
# Командний центр проєкту: Глобальний аналіз метагеномів стічних вод
# Автор: Інна Кучерова
# =========================================================================

.PHONY: all install prepare network epi ml_battle train shap validate sankey dashboard global clean

# Запустити весь аналітичний пайплайн однією командою (без інтерактивних дашбордів)
all: prepare network epi ml_battle train shap validate sankey

# 0. Встановлення залежностей
install:
	@echo "==> Встановлення залежностей..."
	pip install -r requirements.txt

# 1. Попередня обробка даних (Фундамент)
prepare:
	@echo "==> 1. Очищення та нормалізація даних (Log FPKM)..."
	python data_preprocessing.py

# 2. Мережевий аналіз
network:
	@echo "==> 2. Побудова мережі ко-окуренції генів (Інтегрони)..."
	python build_network.py

# 3. Епідеміологічний аналіз
epi:
	@echo "==> 3. Аналіз драйверів AMR (Дані Світового банку)..."
	python epi_analysis.py

# 4. Битва алгоритмів
ml_battle:
	@echo "==> 4. Тестування моделей: RF vs SVM vs XGBoost (ROC Curves)..."
	python ml_battle.py

# 5. Основне тренування моделі
train:
	@echo "==> 5. Тренування Random Forest та витяг ТОП-15 генів..."
	python train.py

# 6. Explainable AI (SHAP)
shap:
	@echo "==> 6. Глибинний SHAP-аналіз (Відкриваємо чорну скриньку)..."
	python shap_analysis.py

# 7. Крос-континентальна валідація
validate:
	@echo "==> 7. Zero-shot тестування на невідомих континентах..."
	python validate_regions.py

# 8. Візуалізація прогнозів
sankey:
	@echo "==> 8. Генерація діаграми потоків Sankey..."
	python plot_sankey.py

# --- ІНТЕРАКТИВНІ ДАШБОРДИ ---

# Запуск основного дашборду
dashboard:
	@echo "==> Запуск основного веб-дашборду..."
	streamlit run dashbord.py

# Запуск дашборду глобальних маркерів
global:
	@echo "==> Запуск дашборду глобальних маркерів..."
	streamlit run global_markers.py

# --- ОЧИЩЕННЯ ---
clean:
	@echo "==> Очищення згенерованих файлів та кешу..."
	rm -f *.png *.jpg *.jpeg *.html *.graphml *.csv
	rm -rf __pycache__ .pytest_cache