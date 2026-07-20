import os
import json
import glob

# Define the directory structures, custom notebook names, and datasets to generate
algorithms = {
    "01_Linear_Regression": {
        "title": "Linear Regression",
        "filename": "Linear_Regression.ipynb",
        "category": "regression",
        "description": "Simple and Multiple Linear Regression modeling linear relationships.",
        "dataset_headers": "YearsExperience,Salary\n",
        "dataset_rows": [
            "1.1,39343.00", "1.3,46205.00", "1.5,37731.00", "2.0,43525.00", "2.2,39891.00",
            "2.9,56642.00", "3.0,60150.00", "3.2,64445.00", "3.7,57189.00", "3.9,63218.00",
            "4.0,55794.00", "4.5,61111.00", "4.9,67938.00", "5.1,66029.00", "5.3,83088.00",
            "5.9,81363.00", "6.0,93940.00", "6.8,91738.00", "7.1,98273.00", "7.9,101302.00",
            "8.2,113812.00", "8.7,109431.00", "9.0,105582.00", "9.5,116969.00", "10.5,121872.00"
        ]
    },
    "02_Polynomial_Regression": {
        "title": "Polynomial Regression",
        "filename": "Polynomial_Regression.ipynb",
        "category": "regression",
        "description": "Capturing non-linear relationships using polynomial terms.",
        "dataset_headers": "Temperature,Yield\n",
        "dataset_rows": [
            "15.0,62.15", "16.2,63.45", "17.5,65.80", "18.0,66.12", "19.5,67.89",
            "20.1,68.45", "22.0,70.12", "25.0,71.50", "27.5,71.65", "28.0,71.45",
            "29.5,70.50", "30.1,70.12", "32.0,68.20", "35.0,64.20", "45.0,44.95"
        ]
    },
    "03_Logistic_Regression": {
        "title": "Logistic Regression",
        "filename": "Logistic_Regression.ipynb",
        "category": "classification",
        "description": "Binary classification modeling probabilities using Sigmoid.",
        "dataset_headers": "Age,EstimatedSalary,Purchased\n",
        "dataset_rows": [
            "19,19000,0", "35,20000,0", "26,43000,0", "27,57000,0", "32,150000,1",
            "25,33000,0", "35,65000,0", "29,80000,0", "47,25000,1", "45,26000,1",
            "46,28000,1", "48,29000,1", "47,49000,1", "53,34000,1", "60,42000,1"
        ]
    },
    "04_KNN": {
        "title": "KNN",
        "filename": "KNN.ipynb",
        "category": "classification",
        "description": "Distance-based classification algorithm classifying points by neighbor majority vote.",
        "dataset_headers": "Size,Texture,Malignant\n",
        "dataset_rows": [
            "1.5,15,0", "1.2,14,0", "2.0,19,0", "1.6,15,0", "1.3,13,0",
            "2.8,24,1", "3.2,25,1", "3.0,22,1", "3.5,28,1", "2.7,21,1",
            "2.2,21,0", "2.6,20,1", "2.1,22,0", "3.8,30,1", "4.2,35,1"
        ]
    },
    "05_Naive_Bayes": {
        "title": "Naive Bayes",
        "filename": "Naive_Bayes.ipynb",
        "category": "classification",
        "description": "Probabilistic classification using Bayes Theorem and feature independence.",
        "dataset_headers": "WordCount,ContainsUrgentWords,Spam\n",
        "dataset_rows": [
            "15,1,1", "150,0,0", "200,0,0", "12,1,1", "300,0,0",
            "8,1,1", "25,1,1", "180,0,0", "15,1,1", "250,0,0",
            "110,0,0", "14,1,1", "350,0,0", "190,0,0", "30,1,1"
        ]
    },
    "06_Decision_Tree": {
        "title": "Decision Tree",
        "filename": "Decision_Tree.ipynb",
        "category": "classification",
        "description": "Classification using sequential, axis-aligned feature splits.",
        "dataset_headers": "CGPA,Internships,Placed\n",
        "dataset_rows": [
            "8.5,2,1", "7.2,1,1", "6.5,0,0", "9.2,3,1", "5.8,1,0",
            "8.0,0,1", "7.1,0,0", "6.9,2,1", "5.5,0,0", "7.8,2,1",
            "6.1,1,0", "9.5,2,1", "5.2,0,0", "8.3,1,1", "9.0,2,1"
        ]
    },
    "07_Random_Forest": {
        "title": "Random Forest",
        "filename": "Random_Forest.ipynb",
        "category": "classification",
        "description": "Ensemble bagging classifier combining many decision trees.",
        "dataset_headers": "CreditScore,DebtToIncomeRatio,Default\n",
        "dataset_rows": [
            "750,0.25,0", "600,0.55,1", "620,0.45,1", "710,0.20,0", "580,0.65,1",
            "680,0.30,0", "720,0.15,0", "610,0.50,1", "790,0.10,0", "550,0.70,1",
            "660,0.28,0", "800,0.12,0", "570,0.60,1", "730,0.22,0", "780,0.08,0"
        ]
    },
    "08_SVM": {
        "title": "SVM",
        "filename": "SVM.ipynb",
        "category": "classification",
        "description": "Margin-maximizing classifier separating classes using hyperplanes.",
        "dataset_headers": "Test1,Test2,Passed\n",
        "dataset_rows": [
            "2.5,3.5,1", "1.5,2.5,0", "2.0,3.0,1", "1.2,1.8,0", "3.0,4.0,1",
            "1.0,1.5,0", "2.8,3.2,1", "1.4,2.2,0", "2.2,2.8,1", "3.5,4.5,1",
            "0.8,1.2,0", "1.6,2.4,0", "3.2,3.8,1", "2.7,3.3,1", "3.4,3.9,1"
        ]
    },
    "09_KMeans": {
        "title": "KMeans",
        "filename": "KMeans.ipynb",
        "category": "clustering",
        "description": "Unsupervised centroid-based clustering algorithm partitioning data.",
        "dataset_headers": "AnnualIncome,SpendingScore\n",
        "dataset_rows": [
            "15,81", "16,6", "17,77", "18,6", "19,81", "20,13", "21,75", "22,14",
            "50,55", "51,48", "52,59", "53,42", "54,58", "90,79", "91,18", "92,83"
        ]
    },
    "10_Hierarchical_Clustering": {
        "title": "Hierarchical Clustering",
        "filename": "Hierarchical_Clustering.ipynb",
        "category": "clustering",
        "description": "Agglomerative hierarchical clustering building a tree of merged clusters.",
        "dataset_headers": "LatitudeOffset,LongitudeOffset\n",
        "dataset_rows": [
            "1.2,1.5", "1.5,1.8", "1.1,1.4", "4.5,4.8", "4.8,5.1", "4.4,4.7",
            "8.2,2.5", "8.5,2.8", "8.1,2.4", "1.0,5.5", "1.3,5.8", "0.9,5.4",
            "4.2,8.5", "4.5,8.8", "8.0,8.5", "8.3,8.8"
        ]
    },
    "11_DBSCAN": {
        "title": "DBSCAN",
        "filename": "DBSCAN.ipynb",
        "category": "clustering",
        "description": "Density-based clustering grouping points by density and identifying noise.",
        "dataset_headers": "X,Y\n",
        "dataset_rows": [
            "0.0,0.5", "0.1,0.6", "0.2,0.65", "0.3,0.68", "0.4,0.70", "0.5,0.70",
            "1.0,0.5", "1.1,0.6", "1.2,0.65", "1.3,0.68", "1.4,0.70", "1.5,0.70",
            "-0.5,0.8", "2.5,-0.5", "1.0,-0.8", "0.0,-0.6", "2.0,1.2"
        ]
    },
    "12_PCA": {
        "title": "PCA",
        "filename": "PCA.ipynb",
        "category": "pca",
        "description": "Linear dimensionality reduction projecting data onto maximum variance components.",
        "dataset_headers": "Alcohol,MalicAcid,Ash,Alcalinity,Magnesium,TotalPhenols\n",
        "dataset_rows": [
            "14.23,1.71,2.43,15.6,127,2.80", "13.20,1.78,2.14,11.2,100,2.65",
            "13.16,2.36,2.67,18.6,101,2.80", "14.37,1.95,2.50,16.8,113,3.85",
            "13.24,2.59,2.87,21.0,118,2.80", "14.20,1.76,2.45,15.2,112,3.27",
            "14.39,1.87,2.45,14.6,96,2.50", "14.06,2.15,2.61,17.6,121,2.60"
        ]
    },
    "13_LDA": {
        "title": "LDA",
        "filename": "LDA.ipynb",
        "category": "classification",
        "description": "Supervised dimensionality reduction maximizing class separation.",
        "dataset_headers": "Nitrogen,Phosphorus,Potassium,CropType\n",
        "dataset_rows": [
            "12,45,20,0", "15,48,22,0", "11,46,19,0", "50,20,40,1", "52,22,42,1",
            "49,19,39,1", "80,80,10,2", "82,82,12,2", "79,79,9,2", "81,81,11,2"
        ]
    },
    "14_AdaBoost": {
        "title": "AdaBoost",
        "filename": "AdaBoost.ipynb",
        "category": "classification",
        "description": "Adaptive boosting training sequential decision stumps focused on errors.",
        "dataset_headers": "Tenure,MonthlyCharges,Churned\n",
        "dataset_rows": [
            "3,85,1", "54,20,0", "12,90,1", "72,25,0", "1,55,1", "45,60,0",
            "30,75,0", "60,35,0", "8,95,1", "4,40,1", "68,110,0", "2,30,1"
        ]
    },
    "15_Gradient_Boosting": {
        "title": "Gradient Boosting",
        "filename": "Gradient_Boosting.ipynb",
        "category": "classification",
        "description": "Ensemble booster fitting decision trees sequentially to loss gradients.",
        "dataset_headers": "SqFt,Rooms,Expensive\n",
        "dataset_rows": [
            "1500,3,0", "2500,4,1", "1800,3,0", "3000,5,1", "1200,2,0", "2200,4,1",
            "1600,3,0", "2800,4,1", "1400,2,0", "2100,3,0", "3500,5,1", "1100,2,0"
        ]
    },
    "16_XGBoost": {
        "title": "XGBoost",
        "filename": "XGBoost.ipynb",
        "category": "classification",
        "description": "Optimized regularized extreme gradient boosting framework.",
        "dataset_headers": "FICO,LTV,Defaulted\n",
        "dataset_rows": [
            "750,0.60,0", "600,0.85,1", "620,0.80,1", "710,0.70,0", "580,0.90,1",
            "680,0.75,0", "690,0.65,0", "720,0.50,0", "610,0.85,1", "630,0.75,1"
        ]
    }
}

# General template constructor for notebooks
def make_notebook_json(folder, title, category, desc):
    cells = []
    
    # 1. Introduction
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# {title}: A Comprehensive Guide\n",
            "\n",
            "## 1. Introduction\n",
            "\n",
            f"This notebook covers the step-by-step implementation of **{title}**, a popular machine learning algorithm. {desc}\n",
            "\n",
            "### Real-World Analogy\n",
            "Explains the intuitive concepts using standard real-world examples.\n",
            "\n",
            "### Advantages & Limitations\n",
            "- Easy to interpret baseline settings.\n",
            "- Scales well to moderate feature spaces."
        ]
    })
    
    # 2. Import Libraries
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Import Libraries\n",
            "\n",
            "We import the standard classical libraries (numpy, pandas, matplotlib, seaborn, and scikit-learn classes)."
        ]
    })
    
    # Code cell imports
    imports_map = {
        "01_Linear_Regression": "from sklearn.linear_model import LinearRegression\nfrom sklearn import metrics\nimport joblib",
        "02_Polynomial_Regression": "from sklearn.preprocessing import PolynomialFeatures\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn import metrics\nimport joblib",
        "03_Logistic_Regression": "from sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn import metrics\nimport joblib",
        "04_KNN": "from sklearn.preprocessing import StandardScaler\nfrom sklearn.neighbors import KNeighborsClassifier\nfrom sklearn import metrics\nimport joblib",
        "05_Naive_Bayes": "from sklearn.naive_bayes import GaussianNB\nfrom sklearn import metrics\nimport joblib",
        "06_Decision_Tree": "from sklearn.tree import DecisionTreeClassifier, plot_tree\nfrom sklearn import metrics\nimport joblib",
        "07_Random_Forest": "from sklearn.ensemble import RandomForestClassifier\nfrom sklearn import metrics\nimport joblib",
        "08_SVM": "from sklearn.preprocessing import StandardScaler\nfrom sklearn.svm import SVC\nfrom sklearn import metrics\nimport joblib",
        "09_KMeans": "from sklearn.preprocessing import StandardScaler\nfrom sklearn.cluster import KMeans\nfrom sklearn.metrics import silhouette_score\nimport joblib",
        "10_Hierarchical_Clustering": "from sklearn.preprocessing import StandardScaler\nimport scipy.cluster.hierarchy as sch\nfrom sklearn.cluster import AgglomerativeClustering\nfrom sklearn.metrics import silhouette_score\nimport joblib",
        "11_DBSCAN": "from sklearn.preprocessing import StandardScaler\nfrom sklearn.cluster import DBSCAN\nfrom sklearn.metrics import silhouette_score\nimport joblib",
        "12_PCA": "from sklearn.preprocessing import StandardScaler\nfrom sklearn.decomposition import PCA\nimport joblib",
        "13_LDA": "from sklearn.preprocessing import StandardScaler\nfrom sklearn.discriminant_analysis import LinearDiscriminantAnalysis\nfrom sklearn import metrics\nimport joblib",
        "14_AdaBoost": "from sklearn.ensemble import AdaBoostClassifier\nfrom sklearn import metrics\nimport joblib",
        "15_Gradient_Boosting": "from sklearn.ensemble import GradientBoostingClassifier\nfrom sklearn import metrics\nimport joblib",
        "16_XGBoost": "from xgboost import XGBClassifier\nfrom sklearn import metrics\nimport joblib"
    }
    
    code_imports = f"import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.model_selection import train_test_split\n{imports_map.get(folder, '')}\n\nsns.set_theme(style='whitegrid')\nplt.rcParams['figure.figsize'] = (10, 6)"
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [code_imports]
    })
    
    # 3. Load / Create Dataset
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Create / Load Dataset\n",
            "\n",
            "We read the synthetic dataset.csv file prepared in the directory."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df = pd.read_csv('dataset.csv')\n",
            "print('Shape:', df.shape)\n",
            "df.head()"
        ]
    })
    
    # 4. Describe Dataset
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Explore Dataset Schema & Statistics\n",
            "\n",
            "Inspect dataframe columns, info, and statistics."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df.info()\n",
            "print(df.describe())"
        ]
    })
    
    # 5. Data Cleaning
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Data Cleaning\n",
            "\n",
            "Verify missing values and duplicate rows."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('Nulls:\\n', df.isnull().sum())\n",
            "print('Duplicates:', df.duplicated().sum())"
        ]
    })
    
    # 6. EDA
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Exploratory Data Analysis (EDA)\n",
            "\n",
            "Plot features correlation matrices and distribution histograms."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')\n",
            "plt.title('Correlation Matrix')\n",
            "plt.show()"
        ]
    })
    
    # 7. Feature Engineering & Split
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Feature Engineering & Split\n",
            "\n",
            "Divide independent and dependent variables, scaling if required."
        ]
    })
    
    engineering_code = ""
    if category in ["clustering", "pca"]:
        # Unsupervised: keep ALL features, no y target label
        engineering_code = "X = df.copy()\n\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\nprint('Unsupervised Scaling applied successfully on entire dataframe!')"
    else:
        # Supervised: y is target label
        if folder in ["03_Logistic_Regression", "04_KNN", "08_SVM", "13_LDA"]:
            engineering_code = "X = df.iloc[:, :-1]\ny = df.iloc[:, -1]\n\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\nprint('Supervised Scaling applied successfully!')"
        else:
            engineering_code = "X = df.iloc[:, :-1]\ny = df.iloc[:, -1]\nprint('Scaling not strictly needed for tree-based/simple regression models.')"
        
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [engineering_code]
    })
    
    # 8. Train-Test Split
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Train-Test Split\n",
            "\n",
            "Divide training and testing sets."
        ]
    })
    
    split_code = ""
    if category in ["clustering", "pca"]:
        split_code = "# Unsupervised learning - fitting directly on scaled cohort\nprint('Skipping split for unsupervised clustering/dimensionality reduction.')"
    else:
        # Check if X_scaled was created
        if folder in ["03_Logistic_Regression", "04_KNN", "08_SVM", "13_LDA"]:
            split_code = "X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)\nprint(f'Train size: {X_train.shape}, Test size: {X_test.shape}')"
        else:
            split_code = "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\nprint(f'Train size: {X_train.shape}, Test size: {X_test.shape}')"
        
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [split_code]
    })
    
    # 9. Model Building
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 9. Model Building\n",
            "\n",
            "Instantiate and fit the target estimator."
        ]
    })
    
    model_build = {
        "01_Linear_Regression": "model = LinearRegression()\nmodel.fit(X_train, y_train)",
        "02_Polynomial_Regression": "poly = PolynomialFeatures(degree=2)\nX_train_poly = poly.fit_transform(X_train)\nmodel = LinearRegression()\nmodel.fit(X_train_poly, y_train)",
        "03_Logistic_Regression": "model = LogisticRegression()\nmodel.fit(X_train, y_train)",
        "04_KNN": "model = KNeighborsClassifier(n_neighbors=3)\nmodel.fit(X_train, y_train)",
        "05_Naive_Bayes": "model = GaussianNB()\nmodel.fit(X_train, y_train)",
        "06_Decision_Tree": "model = DecisionTreeClassifier(max_depth=3, random_state=42)\nmodel.fit(X_train, y_train)",
        "07_Random_Forest": "model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)\nmodel.fit(X_train, y_train)",
        "08_SVM": "model = SVC(kernel='linear', probability=True, random_state=42)\nmodel.fit(X_train, y_train)",
        "09_KMeans": "model = KMeans(n_clusters=3, random_state=42, n_init=10)\nmodel.fit(X_scaled)",
        "10_Hierarchical_Clustering": "model = AgglomerativeClustering(n_clusters=3, linkage='ward')\nmodel.fit(X_scaled)",
        "11_DBSCAN": "model = DBSCAN(eps=0.5, min_samples=3)\nmodel.fit(X_scaled)",
        "12_PCA": "model = PCA(n_components=2)\nX_pca = model.fit_transform(X_scaled)",
        "13_LDA": "model = LinearDiscriminantAnalysis(n_components=2)\nmodel.fit(X_train, y_train)",
        "14_AdaBoost": "model = AdaBoostClassifier(n_estimators=50, random_state=42)\nmodel.fit(X_train, y_train)",
        "15_Gradient_Boosting": "model = GradientBoostingClassifier(n_estimators=50, random_state=42)\nmodel.fit(X_train, y_train)",
        "16_XGBoost": "model = XGBClassifier(n_estimators=50, random_state=42)\nmodel.fit(X_train, y_train)"
    }
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [model_build.get(folder, '')]
    })
    
    # 10. Prediction & 11. Model Evaluation
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 10. Prediction & 11. Model Evaluation\n",
            "\n",
            "Check metrics (MAE, RMSE, Accuracy, Silhouette, or Variance ratios)."
        ]
    })
    
    eval_code = ""
    if category == "regression":
        eval_code = "y_pred = model.predict(X_test) if 'poly' not in locals() else model.predict(poly.transform(X_test))\nprint('MAE:', metrics.mean_absolute_error(y_test, y_pred))\nprint('R2 Score:', metrics.r2_score(y_test, y_pred))"
    elif category == "clustering":
        eval_code = "labels = model.labels_\n# Guard: silhouette_score requires at least 2 distinct non-noise labels\nunique_labels = set(labels)\nif len(unique_labels - {-1}) >= 2:\n    mask = labels != -1\n    print('Silhouette Score:', silhouette_score(X_scaled[mask], labels[mask]))\nelse:\n    print('Silhouette Score: N/A (fewer than 2 valid clusters found)')"
    elif category == "pca":
        eval_code = "print('Explained Variance Ratio:', model.explained_variance_ratio_)"
    else:
        eval_code = "y_pred = model.predict(X_test)\nprint('Accuracy:', metrics.accuracy_score(y_test, y_pred))\nprint(metrics.classification_report(y_test, y_pred))"
        
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [eval_code]
    })
    
    # 12. Visualization
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 12. Visualizing Fit & Boundaries\n",
            "\n",
            "Generate plots showing classification decision zones, regression lines, or clustering segments."
        ]
    })
    
    vis_code = ""
    if category == "regression":
        vis_code = "plt.scatter(y_test, y_pred, color='blue')\nplt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)\nplt.title('Actual vs Predicted')\nplt.show()"
    elif category == "clustering":
        # Multi-feature support: scatter first two components of scaled features
        vis_code = "n_cols = X_scaled.shape[1]\nx_plot = X_scaled[:, 0]\ny_plot = X_scaled[:, 1] if n_cols > 1 else X_scaled[:, 0]\nplt.scatter(x_plot, y_plot, c=model.labels_, cmap='viridis')\nplt.title('Clustering Segments')\nplt.show()"
    elif category == "pca":
        # PCA projection coordinates
        vis_code = "plt.scatter(X_pca[:, 0], X_pca[:, 1], color='indigo')\nplt.title('PCA Projected Space')\nplt.show()"
    else:
        # Classification: always convert to numpy to support both DataFrame and ndarray
        vis_code = "X_arr = np.array(X_test)\nn_feats = X_arr.shape[1] if X_arr.ndim > 1 else 1\nif n_feats >= 2:\n    plt.scatter(X_arr[:, 0], X_arr[:, 1], c=y_pred, cmap='coolwarm')\nelse:\n    plt.scatter(range(len(y_pred)), X_arr[:, 0], c=y_pred, cmap='coolwarm')\nplt.title('Predicted Classes')\nplt.show()"
        
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [vis_code]
    })
    
    # 13. Save and Load
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 13. Save and Load Model\n",
            "\n",
            "Serialize the model to disk via joblib."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "joblib.dump(model, 'model.joblib')\n",
            "loaded = joblib.load('model.joblib')\n",
            "print('Loaded successfully!')"
        ]
    })
    
    # 14-21 Sections
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 14. Common Mistakes & Best Practices\n",
            "- Skipping data normalization/scaling before distance calculations.\n",
            "- Forgetting to tune regularization parameters.\n",
            "\n",
            "## 15. Advantages & 16. Limitations\n",
            "- Simple and quick baseline model.\n",
            "- Limited in capturing highly complex non-linear patterns.\n",
            "\n",
            "## 17. Real-World Applications\n",
            "- E-commerce transaction screening and analytics.\n",
            "- Risk evaluation in financial credit pipelines.\n",
            "\n",
            "## 18. Interview Questions & Answers\n",
            "1. **Explain the fundamental concept of this algorithm.**\n",
            "   * Answers focus on basic mathematical operations and parameters.\n",
            "2. **What does feature scaling do?**\n",
            "   * Normalizes dimensions to avoid metric bias.\n",
            "3. **What is cross-validation?**\n",
            "   * Folds split validation structure to evaluate generalization capability.\n",
            "4. **How do you avoid overfitting?**\n",
            "   * Apply regularization, pruning, or dropout adjustments.\n",
            "5. **What is bias-variance trade-off?**\n",
            "   * Balance between model simplicity and details tracking error.\n",
            "6. **Can this algorithm handle missing values?**\n",
            "   * Typically requires imputation first.\n",
            "7. **What does learning rate represent?**\n",
            "   * Step size updates weight gradient vectors.\n",
            "8. **What is R2 score?**\n",
            "   * Coefficient of determination, variance explained by model.\n",
            "9. **What is F1 score?**\n",
            "   * Harmonic mean of precision and recall values.\n",
            "10. **Explain precision vs recall.**\n",
            "    * Precision covers positive predictions validity; Recall covers positive capture rate.\n",
            "11. **Explain multicollinearity.**\n",
            "    * High correlation between features inflating weights variance.\n",
            "12. **What is regularization?**\n",
            "    * L1/L2 penalty parameters added to loss function.\n",
            "13. **Is this model parametric?**\n",
            "    * Depends on structure configurations.\n",
            "14. **What does the confusion matrix show?**\n",
            "    * Split matrix of TP, TN, FP, FN counts.\n",
            "15. **How does OLS work?**\n",
            "    * Minimizes residual sum of squares variance.\n",
            "\n",
            "## 19. Viva Questions & Answers\n",
            "1. **Define this algorithm.**\n",
            "   * Standard definition fitting features variables.\n",
            "2. **What is target variable?**\n",
            "   * Dependent variable to predict.\n",
            "3. **What is MSE?**\n",
            "   * Mean Squared Error.\n",
            "4. **What is ROC curve?**\n",
            "   * True Positive vs False Positive rates curves.\n",
            "5. **Define standard scaler.**\n",
            "   * Translate features to mean 0, variance 1.\n",
            "6. **What is silhouette score range?**\n",
            "   * Range [-1, 1] for clustering cohesion.\n",
            "7. **Name the scikit-learn module.**\n",
            "   * Sklearn estimators module.\n",
            "8. **How to load scaler?**\n",
            "   * Joblib deserialization.\n",
            "9. **What does classification report return?**\n",
            "   * Accuracy, F1, precision, and recall scores.\n",
            "10. **Explain the training split size.**\n",
            "    * Standard 70-80% partition.\n",
            "11. **What does fit_transform do?**\n",
            "    * Fit model weights and transform inputs in single pass.\n",
            "12. **Is training fast?**\n",
            "    * Yes, computationally direct estimations.\n",
            "13. **Can it handle outliers?**\n",
            "    * Outliers warp boundaries, scaling is needed.\n",
            "14. **What is joblib?**\n",
            "    * Python object serialization tools.\n",
            "15. **Why drop duplicate rows?**\n",
            "    * Prevents leakage and artificial weight bias.\n",
            "\n",
            "## 20. Practice Exercises\n",
            "1. Try changing model parameters (learning_rate, K, or clusters count) and compare metrics.\n",
            "2. Manually calculate error metrics from predictions arrays.\n",
            "3. Implement custom data cleaning pipeline checks.\n",
            "4. Plot decision boundary margins lines.\n",
            "5. Add noisy columns to dataset.csv and compare fit performance.\n",
            "\n",
            "## 21. Conclusion\n",
            "\n",
            "Successfully demonstrated modules modeling and training baseline frameworks!"
        ]
    })
    
    # Save notebook
    nb_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    return nb_dict

def main():
    print("Generating files...")
    
    for folder, data in algorithms.items():
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        # Scan all .ipynb files in the folder and delete ones that don't match the expected name
        existing_notebooks = glob.glob(os.path.join(folder, "*.ipynb"))
        for nb_path in existing_notebooks:
            basename = os.path.basename(nb_path)
            if basename != data["filename"]:
                try:
                    os.remove(nb_path)
                    print(f"Removed deprecated file: {nb_path}")
                except Exception as e:
                    print(f"Failed to remove {nb_path}: {e}")
                    
        # Also clean up the wrong Linear_Regression.ipynb in 06_Decision_Tree if it exists
        if folder == "06_Decision_Tree":
            stray_path = os.path.join(folder, "Linear_Regression.ipynb")
            if os.path.exists(stray_path):
                try:
                    os.remove(stray_path)
                    print(f"Removed stray file: {stray_path}")
                except:
                    pass

        target_nb_path = os.path.join(folder, data["filename"])
        
        # Write dataset.csv
        csv_path = os.path.join(folder, "dataset.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(data["dataset_headers"])
            for row in data["dataset_rows"]:
                f.write(row + "\n")
        print(f"Created dataset: {csv_path}")
        
        # Write .ipynb
        nb_dict = make_notebook_json(folder, data["title"], data["category"], data["description"])
        with open(target_nb_path, "w", encoding="utf-8") as f:
            json.dump(nb_dict, f, indent=1)
        print(f"Created notebook: {target_nb_path}")
        
    print("\nGeneration finished successfully! Re-created correct files and cleaned all deprecated notebooks.")

if __name__ == "__main__":
    main()
