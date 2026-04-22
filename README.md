# 🌍 Global Wastewater AMR Fingerprint: AI & Macroeconomics

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest%20%7C%20SHAP-orange)
![Bioinformatics](https://img.shields.io/badge/Bioinformatics-Metagenomics-green)
![Status](https://img.shields.io/badge/status-production_ready-success)

## 📌 Overview
Antimicrobial resistance (AMR) is a critical global health threat. Traditional clinical surveillance is expensive and fragmented. This project leverages **wastewater metagenomics, ensemble machine learning, and macroeconomic data** to identify a minimal, highly accurate panel of 15 biomarker genes capable of predicting the epidemiological and institutional status of any region in the world.

By moving from descriptive biology to **Explainable AI (SHAP)**, this project demonstrates that a country's AMR burden is fundamentally driven not just by medical practices, but by institutional capacity and bureaucracy.

## 🚀 Key Features & Pipeline
This project is built as a fully reproducible pipeline:
1. **Data Preprocessing:** Log-normalization (FPKM) of raw metagenomic counts and noise filtration.
2. **Network Analysis:** Construction of a co-occurrence network to prove physical gene linkage (identifying the Class 1 Integron mobile hub: `sul1`, `tet(A)`).
3. **Macroeconomic Drivers:** Cross-domain correlation analysis revealing strong links ($r = 0.80$) between AMR burden and World Bank institutional indices (e.g., bureaucracy and tax payment complexity).
4. **Machine Learning:** Training a Random Forest classifier to distinguish high-risk vs. stable epidemiological profiles with 96% accuracy (AUC).
5. **Explainable AI:** SHAP (Shapley Additive exPlanations) analysis to open the "black box" and isolate true global markers (e.g., `blaVEB`, `ere(A)`).
6. **Zero-Shot Validation:** Cross-continental testing using Sankey diagrams to prove the model generalizes flawlessly to unseen data from Africa, the Americas, and Oceania.

## 📊 Data Sources & Citation
This project builds upon the foundational metagenomic data collected by the Global Sewage Surveillance Project, integrated with global economic indicators.

If you use or reference this work, please acknowledge the original data providers:
* **Metagenomic Data:** Hendriksen, R. S., Munk, P., Njage, P., et al. (2019). *Global monitoring of antimicrobial resistance based on metagenomics analyses of urban sewage.* Nature Communications, 10(1), 1124. [DOI: 10.1038/s41467-019-08853-3](https://doi.org/10.1038/s41467-019-08853-3)
* **Macroeconomic Data:** [The World Bank Open Data](https://data.worldbank.org/). Indicators regarding institutional governance, tax complexity, and GDP.

## 🛠 Installation & Usage
The project uses a `Makefile` for full reproducibility. 

**1. Clone the repository and install dependencies:**
```bash
git clone [https://github.com/yourusername/global-wastewater-amr.git](https://github.com/yourusername/global-wastewater-amr.git)
cd global-wastewater-amr
make install
```
2. Run the entire analytical pipeline:
(This will execute preprocessing, network building, ML training, SHAP analysis, and generate all plots)

```bash
make all
```
3. Launch Interactive Dashboards:

```bash
make dashboard  # Main regional comparison tool
make global     # Global fingerprint analysis
```

## 📂 Project Structure

`data_preprocessing.py` - Matrix normalization and noise reduction.

`build_network.py` - Generates the Spearman correlation graph for mobile elements.

`epi_analysis.py` - Calculates Pearson correlations with World Bank indices.

`ml_battle.py` - ROC curve comparison (RF vs. SVM vs. XGBoost).

`train.py` - Core Random Forest training and PCA visualization.

`shap_analysis.py` - Game-theory-based feature importance extraction.

`validate_regions.py` & `plot_sankey.py` - Zero-shot transfer testing.

`dashbord.py` & `global_markers.py` - Streamlit interactive web apps.

## 👩‍🔬 Author

Inna Kucherova 

Molecular biologist

Bioinformatics Researcher & Data Scientist

LinkedIn: https://www.linkedin.com/in/i-tsymbaliuk/
