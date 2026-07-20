# Machine Learning Practice 🚀

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)

Welcome to **Machine-Learning-Practice**! This repository is a comprehensive, portfolio-quality educational resource designed to guide students and developers from beginner to intermediate mastery of classical Machine Learning. 

Every algorithm is implemented from scratch using real-world synthetic datasets and clean, PEP-8 compliant Python code. There is no deep learning or external data dependency; every notebook is 100% self-contained and ready to execute.

---

## 📚 Learning Objectives

By working through this repository, you will:
- Understand the mathematical intuition and real-world analogies behind 16 core ML algorithms.
- Master exploratory data analysis (EDA) and data visualization using Matplotlib and Seaborn.
- Implement robust preprocessing, scaling, and feature engineering pipelines.
- Evaluate models using standard metrics (MAE, RMSE, F1-Score, ROC-AUC, Silhouette Score, etc.).
- Prepare for technical job interviews and viva examinations with comprehensive Q&A sections.

---

## 🛠️ Algorithms Covered

The repository is structured into 16 modules spanning 5 core areas of classical Machine Learning:

| Phase | Category | Module | Algorithm | Notebook Link |
|:---|:---|:---|:---|:---|
| **Phase 1** | **Regression** | `01_Linear_Regression` | Simple & Multiple Linear Regression | [Open Notebook](./01_Linear_Regression/Linear_Regression.ipynb) |
| | | `02_Polynomial_Regression` | Non-Linear Curve Fitting | [Open Notebook](./02_Polynomial_Regression/Polynomial_Regression.ipynb) |
| **Phase 2** | **Classification** | `03_Logistic_Regression` | Binary & Multi-class Logistic Regression | [Open Notebook](./03_Logistic_Regression/Logistic_Regression.ipynb) |
| | | `04_KNN` | k-Nearest Neighbors Distance-based Classifier | [Open Notebook](./04_KNN/KNN.ipynb) |
| | | `05_Naive_Bayes` | Probabilistic Classification (Bayes Theorem) | [Open Notebook](./05_Naive_Bayes/Naive_Bayes.ipynb) |
| **Phase 3** | **Trees & Support Vector**| `06_Decision_Tree` | Gini/Entropy Splitting Trees | [Open Notebook](./06_Decision_Tree/Decision_Tree.ipynb) |
| | | `07_Random_Forest` | Ensemble Bagging & Feature Importance | [Open Notebook](./07_Random_Forest/Random_Forest.ipynb) |
| | | `08_SVM` | Support Vector Machines (Hyperplanes & Kernels) | [Open Notebook](./08_SVM/SVM.ipynb) |
| **Phase 4** | **Clustering** | `09_KMeans` | K-Means Clustering & Elbow Method | [Open Notebook](./09_KMeans/KMeans.ipynb) |
| | | `10_Hierarchical_Clustering`| Agglomerative Clustering & Dendrograms | [Open Notebook](./10_Hierarchical_Clustering/Hierarchical_Clustering.ipynb) |
| | | `11_DBSCAN` | Density-Based Clustering for Complex Shapes | [Open Notebook](./11_DBSCAN/DBSCAN.ipynb) |
| **Phase 5** | **Dimensionality Reduction**| `12_PCA` | Principal Component Analysis | [Open Notebook](./12_PCA/PCA.ipynb) |
| | | `13_LDA` | Linear Discriminant Analysis | [Open Notebook](./13_LDA/LDA.ipynb) |
| **Phase 6** | **Ensembles & Boosting** | `14_AdaBoost` | Adaptive Boosting with Decision Stumps | [Open Notebook](./14_AdaBoost/AdaBoost.ipynb) |
| | | `15_Gradient_Boosting` | Gradient Descent Boosting | [Open Notebook](./15_Gradient_Boosting/Gradient_Boosting.ipynb) |
| | | `16_XGBoost` | Regularized Extreme Gradient Boosting | [Open Notebook](./16_XGBoost/XGBoost.ipynb) |

---

## 📁 Repository Structure

```text
Machine-Learning-Practice/
│
├── README.md                     # This documentation file
├── requirements.txt              # Standard library requirements
├── LICENSE                       # MIT License
├── .gitignore                    # Git ignore configurations
│
├── 01_Linear_Regression/
│   ├── Linear_Regression.ipynb   # Complete step-by-step notebook
│   ├── dataset.csv               # Synthetic dataset used in the notebook
│   └── images/                   # Static visualizations or media assets
...
└── 16_XGBoost/
    ├── XGBoost.ipynb
    ├── dataset.csv
    └── images/
```

---

## 🧑‍💻 Notebook Blueprint

Every notebook in this repository follows a **strict 21-step schema** for absolute consistency and ease of learning:

1. **Introduction**: Mathematical intuition, analogies, pros & cons, real-world use cases.
2. **Import Libraries**: Explicit list of packages used with descriptions.
3. **Create Synthetic Dataset**: Programmatic generation using NumPy or Scikit-learn, saved locally.
4. **Load Dataset**: Pandas descriptive analysis (`shape`, `columns`, `info()`, `describe()`, `head()`, `tail()`).
5. **Data Cleaning**: Missing/duplicate value checks and format auditing.
6. **Exploratory Data Analysis (EDA)**: Professional visualizations (distributions, correlation heatmaps, pairwise relations).
7. **Feature Engineering**: Feature selection, scaling decisions, target extraction.
8. **Train-Test Split**: Detailed explanation of holdout validation, random states, and split proportions.
9. **Model Building**: Instantiating estimators and explaining hyperparameter meanings.
10. **Prediction**: Making predictions on unseen data and customized input arrays.
11. **Model Evaluation**: Metrics breakdown (MAE, MSE, $R^2$, Accuracy, Recall, Precision, Confusion Matrix, ROC-AUC, etc.).
12. **Visualization**: Graphing boundaries, regression fits, and metrics.
13. **Save Model**: Serialization and reload pipeline via `joblib`.
14. **Common Mistakes**: Pitfalls and how to avoid them (e.g. data leakage, wrong scaling).
15. **Advantages**: Situations where the model excels.
16. **Limitations**: Situations where the model fails.
17. **Real-World Applications**: Industry use cases.
18. **Interview Questions**: ~15 technical Q&As.
19. **Viva Questions**: ~15 short oral exam Q&As.
20. **Practice Exercises**: 5-10 challenges for hands-on practice.
21. **Conclusion**: Summary of concepts learned.

---

## 🚀 Installation & Getting Started

### Prerequisites
Make sure you have **Python 3.8+** installed on your local machine.

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Machine-Learning-Practice.git
cd Machine-Learning-Practice
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Jupyter Notebooks

You can run notebooks individually or execute all of them automatically.

#### Option A: Run All Notebooks Automatically (Recommended)
This executes every notebook sequentially, saves the output cells (including plots and metrics), and prepares the workspace:
```bash
python run_all_notebooks.py
```

#### Option B: Run Manually via Jupyter Interface
```bash
jupyter notebook
# OR
jupyter lab
```
Open any folder and run the cells from top to bottom.

---

## 🧠 Future Improvements
- Add cross-validation and hyperparameter tuning sections to all notebooks.
- Introduce unsupervised anomaly detection algorithms (e.g. Isolation Forests).
- Add pipeline deployments using Streamlit or Flask dashboards.

## 🤝 Contribution Guidelines
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/your-username/Machine-Learning-Practice/issues).

## 📄 License
Distributed under the MIT License. See [LICENSE](./LICENSE) for more details.
