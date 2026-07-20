import os
import json
import glob
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_moons, make_blobs

# Detailed algorithms configurations with specific formulas, analogies, worked examples, and QA
algorithms = {
    "01_Linear_Regression": {
        "title": "Linear Regression",
        "filename": "Linear_Regression.ipynb",
        "category": "regression",
        "description": "Models a linear relationship between input features and a continuous target.",
        "analogy": "Predicting a student's test score based on the number of hours they studied.",
        "math_formula": "y = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + \\dots + \\beta_n x_n + \\epsilon",
        "worked_example": "Predicting salary from years of experience. With intercept beta_0 = 30000 and slope beta_1 = 9000, a person with 5 years experience is predicted to earn: 30000 + 9000 * 5 = $75,000.",
        "dataset_code": """# Programmatic generation of salary dataset
np.random.seed(42)
X_data = np.random.uniform(1.0, 10.5, 500)
noise = np.random.normal(0, 5000, 500)
y_data = 30000 + 9000 * X_data + noise
df = pd.DataFrame({
    'YearsExperience': np.round(X_data, 1),
    'Salary': np.round(y_data, 2)
})
df.to_csv('dataset.csv', index=False)""",
        "model_build": """model = LinearRegression()
model.fit(X_train, y_train)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)""",
        "model_eval": """y_pred = model.predict(X_test)
print("MAE:", metrics.mean_absolute_error(y_test, y_pred))
print("MSE:", metrics.mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 Score:", metrics.r2_score(y_test, y_pred))""",
        "model_vis": """fig, axes = plt.subplots(1, 2, figsize=(15, 6))
# Plot 1: Regression Line fit
axes[0].scatter(X_test, y_test, color='blue', alpha=0.6, label='Actual')
axes[0].plot(X_test, y_pred, color='red', linewidth=2, label='Fit')
axes[0].set_title("Regression Line Fit")
axes[0].set_xlabel("Years Experience")
axes[0].set_ylabel("Salary")
axes[0].legend()

# Plot 2: Residuals Plot
residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, color='purple', alpha=0.6)
axes[1].axhline(y=0, color='black', linestyle='--')
axes[1].set_title("Residuals vs. Predicted Values")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Residuals")
plt.tight_layout()
plt.show()""",
        "qa_interview": [
            ("What is homoscedasticity?", "Homoscedasticity means the residual variance is constant across all levels of independent variables. Heteroscedasticity violates OLS assumptions."),
            ("What are the 4 assumptions of OLS linear regression?", "Linearity, Independence of errors, Homoscedasticity, and Normality of residual error distributions."),
            ("Why is R-squared not always the best metric?", "R-squared always increases or stays same when features are added. Adjusted R-squared accounts for number of features and is preferred.")
        ],
        "qa_viva": [
            ("Define residual.", "The difference between the actual value and the predicted value: e = y - y_pred."),
            ("What value indicates a perfect fit for R2?", "An R2 score of 1.0 indicates a perfect fit."),
            ("Does feature scaling change the R2 score?", "No, scaling features changes coefficient scale but doesn't change the linear fit or R2 score.")
        ]
    },
    "02_Polynomial_Regression": {
        "title": "Polynomial Regression",
        "filename": "Polynomial_Regression.ipynb",
        "category": "regression",
        "description": "Models relationships where target variations describe curved non-linear patterns using polynomial features transformations.",
        "analogy": "The trajectory of a thrown ball (gravity shapes a curved quadratic parabola path).",
        "math_formula": "y = \\beta_0 + \\beta_1 x + \\beta_2 x^2 + \\dots + \\beta_d x^d + \\epsilon",
        "worked_example": "Chemical yield vs. Temperature. Yield curves up and then down due to boiling. A 2nd-degree polynomial yield = 50 + 2*Temp - 0.05*Temp^2 defines maximum yield at 20 degrees.",
        "dataset_code": """# Programmatic generation of non-linear chemical yield dataset
np.random.seed(42)
X_data = np.random.uniform(-10, 40, 400)
noise = np.random.normal(0, 4, 400)
y_data = 50 + 2.5 * X_data - 0.08 * (X_data ** 2) + noise
df = pd.DataFrame({
    'Temperature': np.round(X_data, 1),
    'Yield': np.round(y_data, 2)
})
df.to_csv('dataset.csv', index=False)""",
        "model_build": """poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
model = LinearRegression()
model.fit(X_train_poly, y_train)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)""",
        "model_eval": """y_pred = model.predict(X_test_poly)
print("MAE:", metrics.mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 Score:", metrics.r2_score(y_test, y_pred))""",
        "model_vis": """plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', alpha=0.6, label='Actual Data')
# Sort values for smooth curve plotting
sort_idx = np.argsort(X_test.iloc[:, 0])
plt.plot(X_test.iloc[sort_idx, 0], y_pred[sort_idx], color='red', linewidth=2.5, label='Polynomial Fit')
plt.title('Polynomial Curve Fit (Degree = 2)')
plt.xlabel('Temperature')
plt.ylabel('Yield')
plt.legend()
plt.show()""",
        "qa_interview": [
            ("How does Polynomial Regression avoid linear limits?", "It transforms the feature space by creating cross-product and power terms (e.g. x^2, x^3) allowing OLS to find linear hyperplane separators in higher dimensions."),
            ("What is the danger of setting the degree parameter too high?", "Setting the degree too high leads to severe overfitting, where the model fits the noise of the training data rather than the underlying pattern."),
            ("Is Polynomial Regression considered a linear model?", "Yes, it is linear in terms of its parameters (coefficients beta) even though the features are raised to non-linear power terms.")
        ],
        "qa_viva": [
            ("What class creates polynomial features in scikit-learn?", "`PolynomialFeatures` from `sklearn.preprocessing`."),
            ("What does the 'degree' parameter mean?", "It specifies the highest power term generated during feature engineering transformations."),
            ("Why do we fit_transform on train but only transform on test?", "To prevent data leakage by estimating feature means/ranges solely from the training partition.")
        ]
    },
    "03_Logistic_Regression": {
        "title": "Logistic Regression",
        "filename": "Logistic_Regression.ipynb",
        "category": "classification",
        "description": "Models the probability of a binary output classification class using the Sigmoid activation function.",
        "analogy": "Deciding whether an email is spam (1) or not spam (0) based on certain keywords.",
        "math_formula": "p = \\sigma(z) = \\frac{1}{1 + e^{-z}} \\quad \\text{where} \\quad z = \\beta_0 + \\beta_1 x_1 + \\dots + \\beta_n x_n",
        "worked_example": "Predicting customer purchase. Sigmoid converts linear output z = 0.5 into a probability: p = 1 / (1 + e^-0.5) = 0.62. Since 62% is >= 50% default cutoff, predict class 1 (Purchased).",
        "dataset_code": """# Programmatic generation of customer purchase data
from sklearn.datasets import make_classification
np.random.seed(42)
X_raw, y_raw = make_classification(n_samples=500, n_features=2, n_informative=2,
                                  n_redundant=0, n_classes=2, n_clusters_per_class=1,
                                  weights=[0.6, 0.4], class_sep=1.2, random_state=42)
# Scale features artificially to mimic Age and Salary
Age = 20 + X_raw[:, 0] * 8 + 15
Salary = 20000 + X_raw[:, 1] * 35000 + 50000
df = pd.DataFrame({
    'Age': np.round(np.clip(Age, 18, 70), 0),
    'EstimatedSalary': np.round(np.clip(Salary, 15000, 150000), -2),
    'Purchased': y_raw
})
df.to_csv('dataset.csv', index=False)""",
        "model_build": """model = LogisticRegression()
model.fit(X_train, y_train)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)""",
        "model_eval": """y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Classification Report:\\n", metrics.classification_report(y_test, y_pred))
print("ROC-AUC Score:", metrics.roc_auc_score(y_test, y_prob))""",
        "model_vis": """fig, axes = plt.subplots(1, 2, figsize=(15, 6))
# Plot 1: Confusion Matrix
cm = metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title("Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# Plot 2: ROC Curve
fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color='orange', lw=2, label=f'ROC curve (AUC = {metrics.roc_auc_score(y_test, y_prob):.2f})')
axes[1].plot([0, 1], [0, 1], color='navy', linestyle='--')
axes[1].set_title("ROC Curve")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend()
plt.tight_layout()
plt.show()""",
        "qa_interview": [
            ("What loss function is minimized in Logistic Regression?", "Log-loss (also called Binary Cross-Entropy Loss) is minimized using optimization algorithms like Gradient Descent."),
            ("Why is linear regression not suitable for classification?", "Linear regression can predict values outside the [0, 1] range and is highly sensitive to outliers in classification labels."),
            ("What are odds and log-odds?", "Odds represent ratio of success to failure p/(1-p). Log-odds is the natural log of odds, which scales predictions linearly from negative to positive infinity.")
        ],
        "qa_viva": [
            ("What does Sigmoid function output range represent?", "It outputs a probability value bounded strictly between 0 and 1."),
            ("What is the default probability threshold for binary classification?", "0.5 (50% probability)."),
            ("Does logistic regression require scaling?", "For convergence speed and coefficient interpretation regularization (L1/L2), scaling is highly recommended.")
        ]
    },
    "04_KNN": {
        "title": "K-Nearest Neighbors",
        "filename": "KNN.ipynb",
        "category": "classification",
        "description": "Classifies objects by comparing distances to their closest labeled neighbors in the feature space.",
        "analogy": "'Birds of a feather flock together.' You behave similarly to the 5 closest friends you interact with.",
        "math_formula": "d(\\mathbf{p}, \\mathbf{q}) = \\sqrt{\\sum_{i=1}^n (p_i - q_i)^2} \\quad \\text{(Euclidean Distance)}",
        "worked_example": "Classifying a new cell as malignant. We calculate Euclidean distances to all known cells. If 5 nearest neighbors contains 4 malignant cells and 1 benign cell, we classify the cell as malignant (80% majority vote).",
        "dataset_code": """# Programmatic generation of tumor coordinates
from sklearn.datasets import make_classification
np.random.seed(42)
X_raw, y_raw = make_classification(n_samples=400, n_features=2, n_informative=2,
                                  n_redundant=0, n_classes=2, n_clusters_per_class=1,
                                  class_sep=1.5, random_state=42)
df = pd.DataFrame({
    'Size': np.round(X_raw[:, 0] * 3 + 6, 2),
    'Texture': np.round(X_raw[:, 1] * 5 + 20, 2),
    'Malignant': y_raw
})
df.to_csv('dataset.csv', index=False)""",
        "model_build": """model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
print("KNN model fitted.")""",
        "model_eval": """y_pred = model.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("F1 Score:", metrics.f1_score(y_test, y_pred))
print("Confusion Matrix:\\n", metrics.confusion_matrix(y_test, y_pred))""",
        "model_vis": """# Decision Boundary Plotting for KNN
h = 0.05
x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', edgecolors='k', s=50)
plt.title("KNN (K=5) Decision Boundary")
plt.xlabel("Scaled Size")
plt.ylabel("Scaled Texture")
plt.show()""",
        "qa_interview": [
            ("Why is KNN called a lazy learner?", "Because it has no explicit training phase. It simply stores the training dataset and performs all computations during inference."),
            ("What is the impact of K parameter choice?", "Small K (e.g. K=1) leads to high variance and overfitting (sensitive to noise). Large K leads to high bias and underfitting (smooths boundaries)."),
            ("Why is feature scaling critical for KNN?", "KNN calculates distance metrics. Features with larger ranges (e.g. income) dominate features with smaller ranges (e.g. age) if left unscaled.")
        ],
        "qa_viva": [
            ("How do you break ties when using KNN for binary classification?", "Use an odd value of K (e.g., K = 3, 5, 7) to avoid equal voting outcomes."),
            ("What is the default distance metric in scikit-learn's KNN?", "Minkowski distance with p=2, which is equivalent to Euclidean distance."),
            ("Name one disadvantage of KNN.", "It is computationally expensive and slow during inference on large datasets because it must scan all training points.")
        ]
    },
    "05_Naive_Bayes": {
        "title": "Naive Bayes",
        "filename": "Naive_Bayes.ipynb",
        "category": "classification",
        "description": "A probabilistic classification algorithm based on Bayes' Theorem that assumes strong independence between features.",
        "analogy": "Deciding if it will rain based on temperature and wind, assuming they operate completely independently.",
        "math_formula": "P(C_k | \\mathbf{x}) = \\frac{P(\\mathbf{x} | C_k) P(C_k)}{P(\\mathbf{x})} \\quad \\text{where Naive assumption is } P(\\mathbf{x}|C_k) = \\prod_{i=1}^n P(x_i | C_k)",
        "worked_example": "Classifying email as spam. If words 'urgent' and 'free' appear, Naive Bayes computes probability: P(Spam|words) proportional to P('urgent'|Spam) * P('free'|Spam) * P(Spam) and compares it to the Non-Spam probability to choose the highest.",
        "dataset_code": """# Programmatic generation of spam classification dataset
from sklearn.datasets import make_classification
np.random.seed(42)
X_raw, y_raw = make_classification(n_samples=400, n_features=3, n_informative=3,
                                  n_redundant=0, n_classes=2, weights=[0.7, 0.3],
                                  class_sep=1.1, random_state=42)
df = pd.DataFrame({
    'WordCount': np.round(np.clip(X_raw[:, 0] * 50 + 150, 10, 500), 0),
    'ContainsUrgentWords': np.where(X_raw[:, 1] > 0.3, 1, 0),
    'ExclamationCount': np.round(np.clip(X_raw[:, 2] * 3 + 2, 0, 15), 0),
    'Spam': y_raw
})
df.to_csv('dataset.csv', index=False)""",
        "model_build": """model = GaussianNB()
model.fit(X_train, y_train)
print("Naive Bayes model trained.")""",
        "model_eval": """y_pred = model.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Classification Report:\\n", metrics.classification_report(y_test, y_pred))""",
        "model_vis": """fig, axes = plt.subplots(1, 2, figsize=(15, 6))
# Plot 1: Confusion Matrix
cm = metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=axes[0])
axes[0].set_title("Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# Plot 2: Feature distributions per class
sns.kdeplot(data=df, x='WordCount', hue='Spam', fill=True, ax=axes[1])
axes[1].set_title("WordCount Density per class")
plt.tight_layout()
plt.show()""",
        "qa_interview": [
            ("Why is it called 'Naive'?", "Because it naively assumes that all input features are completely independent of each other given the class label, which is rarely true in real life."),
            ("What is Laplacian Smoothing?", "A technique to handle zero-probability outcomes. It adds a small value (alpha=1) to feature counts so that new unseen words do not zero out the entire conditional probability product."),
            ("When does Naive Bayes perform exceptionally well?", "It performs very well in text classification (e.g. spam filtering, sentiment analysis) and high-dimensional sparse spaces.")
        ],
        "qa_viva": [
            ("What are the main types of Naive Bayes classifiers?", "Gaussian (continuous features), Multinomial (text counts), and Bernoulli (binary indicators)."),
            ("Does Naive Bayes require scale adjustments?", "No, it computes class-conditional density statistics independently for each feature, so scaling has no effect."),
            ("What formula is it based on?", "Bayes' Theorem: P(A|B) = P(B|A) * P(A) / P(B).")
        ]
    },
    "06_Decision_Tree": {
        "title": "Decision Tree",
        "filename": "Decision_Tree.ipynb",
        "category": "classification",
        "description": "Constructs sequential decision pathways splitting data on axis-aligned feature boundaries.",
        "analogy": "Playing a game of 20 Questions, narrowing down the target category step-by-step.",
        "math_formula": "\\text{Gini Impurity: } G = 1 - \\sum_{i=1}^c p_i^2 \\quad \\text{or Entropy: } H = -\\sum_{i=1}^c p_i \\log_2 p_i",
        "worked_example": "Classifying student job placement. If CGPA >= 7.5, go to next split (Internships). If Internships >= 1, predict placed (1), else predict unplaced (0).",
        "dataset_code": """# Programmatic generation of student placement records
np.random.seed(42)
CGPA = np.random.uniform(5.5, 9.8, 500)
Internships = np.random.randint(0, 4, 500)
# Custom logic: placement likely with high CGPA + Internships
placed_probability = 1 / (1 + np.exp(-(2.5 * CGPA + 1.8 * Internships - 20)))
Placed = np.where(placed_probability > np.random.uniform(0, 1, 500), 1, 0)
df = pd.DataFrame({
    'CGPA': np.round(CGPA, 2),
    'Internships': Internships,
    'Placed': Placed
})
df.to_csv('dataset.csv', index=False)""",
        "model_build": """model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)
print("Decision Tree fit finished.")""",
        "model_eval": """y_pred = model.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Classification Report:\\n", metrics.classification_report(y_test, y_pred))""",
        "model_vis": """fig, axes = plt.subplots(1, 2, figsize=(18, 8))
# Plot 1: Visual Tree Diagram
plot_tree(model, feature_names=X.columns.tolist(), class_names=['Not Placed', 'Placed'], filled=True, ax=axes[0])
axes[0].set_title("Decision Tree Flowchart Map")

# Plot 2: Feature Importance
importances = model.feature_importances_
axes[1].barh(X.columns, importances, color='teal')
axes[1].set_title("Feature Importances")
axes[1].set_xlabel("Relative Gini Reduction")
plt.tight_layout()
plt.show()""",
        "qa_interview": [
            ("What is Information Gain?", "The reduction in entropy or impurity after a dataset partition split. Decision trees choose split criteria that maximize Information Gain."),
            ("How do you prevent Decision Trees from overfitting?", "Use pruning parameters such as `max_depth` (limits tree growth), `min_samples_split`, or `min_samples_leaf`."),
            ("What are axis-aligned splits?", "Decision trees split one feature at a time, creating boundaries that are perpendicular to the feature axes (rectangles/boxes).")
        ],
        "qa_viva": [
            ("What impurity criterion is scikit-learn's default?", "Gini Impurity (`gini`)."),
            ("Does scaling affect Decision Trees?", "No, scaling does not change feature order or threshold split choices, so trees are completely scale-invariant."),
            ("What does a leaf node represent?", "A terminal node containing class predictions (no further splitting occurs).")
        ]
    },
    "07_Random_Forest": {
        "title": "Random Forest",
        "filename": "Random_Forest.ipynb",
        "category": "classification",
        "description": "An ensemble bagging model that trains multiple independent decision trees on bootstrapped data samples.",
        "analogy": "Asking a committee of 100 experts for advice and taking their majority vote, rather than relying on one person.",
        "math_formula": "f(\\mathbf{x}) = \\text{Majority Vote} \\left( h_1(\\mathbf{x}), h_2(\\mathbf{x}), \\dots, h_B(\\mathbf{x}) \\right)",
        "worked_example": "Predicting loan defaults. We train 100 trees on bootstrapped versions of credit profiles. For a customer, 75 trees predict 'Default' (1) and 25 trees predict 'No Default' (0). Predict 1.",
        "dataset_code": """# Programmatic generation of credit default profiles
from sklearn.datasets import make_classification
np.random.seed(42)
X_raw, y_raw = make_classification(n_samples=600, n_features=4, n_informative=3,
                                  n_redundant=1, n_classes=2, weights=[0.8, 0.2],
                                  class_sep=1.3, random_state=42)
df = pd.DataFrame({
    'CreditScore': np.round(X_raw[:, 0] * 80 + 650, 0),
    'DebtToIncome': np.round(np.clip(X_raw[:, 1] * 0.15 + 0.35, 0.05, 0.95), 2),
    'Savings': np.round(np.clip(X_raw[:, 2] * 25000 + 30000, 0, 150000), -2),
    'Age': np.round(X_raw[:, 3] * 10 + 40, 0).astype(int),
    'Default': y_raw
})
df.to_csv('dataset.csv', index=False)""",
        "model_build": """model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)
print("Random Forest model trained successfully.")""",
        "model_eval": """y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("ROC-AUC:", metrics.roc_auc_score(y_test, y_prob))
print("Confusion Matrix:\\n", metrics.confusion_matrix(y_test, y_pred))""",
        "model_vis": """fig, axes = plt.subplots(1, 2, figsize=(15, 6))
# Plot 1: Feature Importances
importances = model.feature_importances_
axes[0].barh(X.columns, importances, color='darkgreen')
axes[0].set_title("Random Forest Feature Importances")
axes[0].set_xlabel("Mean Decrease in Impurity")

# Plot 2: Confusion Matrix
cm = metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title("Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")
plt.tight_layout()
plt.show()""",
        "qa_interview": [
            ("What are the two sources of randomness in Random Forest?", "1. Bootstrap sampling (bagging): each tree is trained on a random sample of rows with replacement. 2. Feature subsetting: each split considers a random subset of columns."),
            ("What is Out-Of-Bag (OOB) error?", "The average error evaluated on each training sample using only the trees that did not include that sample in their bootstrap training set. It serves as validation without split partition."),
            ("How does Random Forest reduce model variance?", "By averaging the predictions of individual, highly uncorrelated decision trees (which individually have high variance).")
        ],
        "qa_viva": [
            ("What does the 'n_estimators' parameter do?", "It sets the number of individual decision trees trained in the forest ensemble."),
            ("Is Random Forest prone to overfitting when n_estimators is large?", "No, increasing trees count does not lead to overfitting; it simply stabilizes predictions variance."),
            ("Name one advantage of Random Forest over a single Decision Tree.", "It has higher accuracy, generalizes much better, and is highly resistant to overfitting.")
        ]
    },
    "08_SVM": {
        "title": "Support Vector Machine",
        "filename": "SVM.ipynb",
        "category": "classification",
        "description": "Finds the margin-maximizing separating boundary hyperplane that creates the widest possible gap between classes.",
        "analogy": "Building a demilitarized zone exactly in the middle between two opposing military camps.",
        "math_formula": "\\min_{\\mathbf{w}, b} \\frac{1}{2}\\|\\mathbf{w}\\|^2 \\quad \\text{subject to} \\quad y_i(\\mathbf{w}\\cdot\\mathbf{x}_i + b) \\ge 1",
        "worked_example": "Classifying quality control data. SVM fits decision boundary line. Points sitting right on the margins are the 'Support Vectors'. Any other point can be deleted without shifting the line.",
        "dataset_code": """# Programmatic generation of microchip QC data
from sklearn.datasets import make_classification
np.random.seed(42)
X_raw, y_raw = make_classification(n_samples=400, n_features=2, n_informative=2,
                                  n_redundant=0, n_classes=2, n_clusters_per_class=1,
                                  weights=[0.5, 0.5], class_sep=1.3, random_state=42)
df = pd.DataFrame({
    'Test1': np.round(X_raw[:, 0] * 2.5 + 5.0, 2),
    'Test2': np.round(X_raw[:, 1] * 2.5 + 5.0, 2),
    'Passed': y_raw
})
df.to_csv('dataset.csv', index=False)""",
        "model_build": """model = SVC(kernel='linear', probability=True, random_state=42)
model.fit(X_train, y_train)
print("SVM classification model fit completed.")""",
        "model_eval": """y_pred = model.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))""",
        "model_vis": """fig, axes = plt.subplots(1, 2, figsize=(15, 6))
# Plot 1: Decision boundary
h = 0.05
x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

axes[0].contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
axes[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', edgecolors='k')
# Plot support vectors
sv = model.support_vectors_
axes[0].scatter(sv[:, 0], sv[:, 1], s=120, facecolors='none', edgecolors='k', label='Support Vectors')
axes[0].set_title("Decision Boundary and Support Vectors")
axes[0].legend()

# Plot 2: Confusion Matrix
cm = metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=axes[1])
axes[1].set_title("Confusion Matrix")
plt.tight_layout()
plt.show()""",
        "qa_interview": [
            ("What is the kernel trick?", "Mapping non-linearly separable inputs into higher-dimensional coordinate systems using dot-product functions (RBF, Polynomial) without explicitly calculating high-dimensional coordinates."),
            ("What is the difference between Hard and Soft Margin SVM?", "Hard margin requires perfect linearly separable data (no errors allowed). Soft margin uses a slack variable C to allow some training errors to find a wider margin that generalizes better."),
            ("How does parameter C regulate regularization?", "A large C penalizes misclassifications heavily, leading to a narrower margin (risk of overfitting). A small C allows errors, creating a wider margin (risk of underfitting).")
        ],
        "qa_viva": [
            ("What are support vectors?", "The key data points sitting right on the margins that determine the boundary hyperplane."),
            ("What is the default kernel in scikit-learn's SVC?", "`rbf` (Radial Basis Function)."),
            ("Why is scaling important for SVM?", "SVM uses geometric distances to maximize margins. Unscaled variables distort the distance computation.")
        ]
    },
    "09_KMeans": {
        "title": "KMeans Clustering",
        "filename": "KMeans.ipynb",
        "category": "clustering",
        "description": "An unsupervised partitioning algorithm that groups data points into K clusters centered around iteratively calculated centroids.",
        "analogy": "Setting up K shipping hubs in a country to minimize delivery distance to all stores.",
        "math_formula": "J = \\sum_{j=1}^K \\sum_{\\mathbf{x}_i \\in S_j} \\|\\mathbf{x}_i - \\boldsymbol{\\mu}_j\\|^2 \\quad \\text{(Minimize WCSS)}",
        "worked_example": "Grouping shoppers by income and spending habits. We set K=3 hubs. KMeans iteratively assigns shoppers to nearest hubs, then shifts hubs to average coordinates, converging into 3 customer segments.",
        "dataset_code": """# Programmatic generation of 2D customer groups
from sklearn.datasets import make_blobs
np.random.seed(42)
X_raw, _ = make_blobs(n_samples=500, n_features=2, centers=3, cluster_std=1.2, random_state=42)
df = pd.DataFrame({
    'AnnualIncome': np.round(X_raw[:, 0] * 6 + 60, 2),
    'SpendingScore': np.round(X_raw[:, 1] * 8 + 50, 2)
})
df.to_csv('dataset.csv', index=False)""",
        "model_build": """model = KMeans(n_clusters=3, random_state=42, n_init=10)
model.fit(X_scaled)
print("KMeans centroids:\\n", model.cluster_centers_)""",
        "model_eval": """labels = model.labels_
# Guard: silhouette_score requires at least 2 distinct non-noise labels
unique_labels = set(labels)
if len(unique_labels - {-1}) >= 2:
    mask = labels != -1
    print('Silhouette Score:', silhouette_score(X_scaled[mask], labels[mask]))
else:
    print('Silhouette Score: N/A')""",
        "model_vis": """fig, axes = plt.subplots(1, 2, figsize=(15, 6))
# Plot 1: Clusters assignment
axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=model.labels_, cmap='viridis', alpha=0.6)
# Plot Centroids
centroids = model.cluster_centers_
axes[0].scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroids')
axes[0].set_title("K-Means Clusters")
axes[0].legend()

# Plot 2: WCSS Elbow sweep (Inertia check)
wcss = []
K_range = range(1, 8)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    wcss.append(km.inertia_)
axes[1].plot(K_range, wcss, marker='o', color='purple')
axes[1].set_title("Elbow Curve (WCSS)")
axes[1].set_xlabel("Number of Clusters K")
axes[1].set_ylabel("Inertia (WCSS)")
plt.tight_layout()
plt.show()""",
        "qa_interview": [
            ("How does KMeans initialize and iterate?", "1. Randomly place K centroids. 2. Assign each point to the closest centroid. 3. Recompute centroids as mean coordinates of assigned points. Repeat steps 2 and 3 until convergence."),
            ("What is the Elbow Method?", "Plotting WCSS (Inertia) against K values. The 'elbow' point is where WCSS decrease slows down dramatically, suggesting the optimal K."),
            ("What are the limitations of KMeans?", "1. Sensitive to initial centroid placement (can get stuck in local minima). 2. Requires specifying K in advance. 3. Struggles with non-spherical shapes and varying density clusters.")
        ],
        "qa_viva": [
            ("What does 'inertia_' represent in scikit-learn?", "The sum of squared distances of samples to their closest cluster centroid (Within-Cluster Sum of Squares)."),
            ("What is the role of 'n_init'?", "Specifies how many times KMeans runs with different centroid seeds, returning the run with the lowest WCSS."),
            ("Does KMeans require feature scaling?", "Yes, because it uses Euclidean distance. Features with larger variance will dominate centroid assignments.")
        ]
    },
    "10_Hierarchical_Clustering": {
        "title": "Hierarchical Clustering",
        "filename": "Hierarchical_Clustering.ipynb",
        "category": "clustering",
        "description": "Constructs tree-like structures (dendrograms) by iteratively grouping (agglomerative) or splitting (divisive) data clusters.",
        "analogy": "Structuring file systems recursively into sub-folders, folders, and parent directories.",
        "math_formula": "\\text{Ward's Linkage: } \\Delta(A, B) = \\frac{n_A n_B}{n_A + n_B} \\|\\boldsymbol{\\mu}_A - \\boldsymbol{\\mu}_B\\|^2",
        "worked_example": "Grouping delivery store sites. Agglomerative clustering starts with each site as its own cluster. It merges the two closest sites, repeating until all merge into one grand tree.",
        "dataset_code": """# Programmatic generation of geographical retail spots
from sklearn.datasets import make_blobs
np.random.seed(42)
X_raw, _ = make_blobs(n_samples=300, n_features=2, centers=4, cluster_std=1.0, random_state=42)
df = pd.DataFrame({
    'LatitudeOffset': np.round(X_raw[:, 0] * 3, 4),
    'LongitudeOffset': np.round(X_raw[:, 1] * 3, 4)
})
df.to_csv('dataset.csv', index=False)""",
        "model_build": """model = AgglomerativeClustering(n_clusters=4, linkage='ward')
model.fit(X_scaled)
print("Agglomerative model fitted.")""",
        "model_eval": """labels = model.labels_
# Guard: silhouette_score requires at least 2 distinct non-noise labels
unique_labels = set(labels)
if len(unique_labels - {-1}) >= 2:
    mask = labels != -1
    print('Silhouette Score:', silhouette_score(X_scaled[mask], labels[mask]))
else:
    print('Silhouette Score: N/A')""",
        "model_vis": """fig, axes = plt.subplots(1, 2, figsize=(16, 6))
# Plot 1: Dendrogram
import scipy.cluster.hierarchy as sch
sch.dendrogram(sch.linkage(X_scaled, method='ward'), ax=axes[0])
axes[0].set_title("Dendrogram Hierarchy Tree")
axes[0].set_xlabel("Sample Indices")
axes[0].set_ylabel("Ward Distance Threshold")

# Plot 2: Agglomerative Scatter
axes[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=model.labels_, cmap='tab10', alpha=0.7)
axes[1].set_title("Agglomerative Clusters (Ward Linkage)")
plt.tight_layout()
plt.show()""",
        "qa_interview": [
            ("What is the difference between Agglomerative and Divisive clustering?", "Agglomerative is a bottom-up approach (starts with individual points and merges them). Divisive is top-down (starts with one cluster and recursively splits it)."),
            ("What are the common linkage criteria?", "1. Single (min distance between points). 2. Complete (max distance). 3. Average (mean distance). 4. Ward (minimizes variance within clusters)."),
            ("How does a dendrogram determine the number of clusters?", "By drawing a horizontal line across the dendrogram at a threshold height that does not intersect any vertical merge branches.")
        ],
        "qa_viva": [
            ("What is linkage method 'ward'?", "An approach that minimizes the total within-cluster variance when merging two clusters."),
            ("Does Hierarchical Clustering require random seed initialization?", "No, unlike KMeans, it is deterministic and does not rely on random seed initializations."),
            ("Name the library used for plotting the dendrogram tree.", "`scipy.cluster.hierarchy`.")
        ]
    },
    "11_DBSCAN": {
        "title": "DBSCAN",
        "filename": "DBSCAN.ipynb",
        "category": "clustering",
        "description": "Density-based spatial clustering that groups dense regions of points and identifies sparse points as outliers.",
        "analogy": "Identifying crowded cities (dense hubs) and isolating isolated wilderness homesteads (noise/outliers).",
        "math_formula": "\\text{Condition: } |N_\\epsilon(\\mathbf{p})| \\ge \\text{MinPts} \\quad \\text{where } N_\\epsilon(\\mathbf{p}) = \\{\\mathbf{q} \\in D \\mid d(\\mathbf{p}, \\mathbf{q}) \\le \\epsilon\\}",
        "worked_example": "Clustering crescent moon shapes. DBSCAN groups points that are close together. Outliers far from the crescent dense zone are labeled as noise class -1.",
        "dataset_code": """# Programmatic generation of concentric moons with noise
from sklearn.datasets import make_moons
np.random.seed(42)
X_raw, _ = make_moons(n_samples=400, noise=0.08, random_state=42)
# Add some uniform background noise points
outliers = np.random.uniform(-1.5, 2.5, size=(40, 2))
X_combined = np.vstack([X_raw, outliers])
df = pd.DataFrame({
    'X': np.round(X_combined[:, 0], 4),
    'Y': np.round(X_combined[:, 1], 4)
})
df.to_csv('dataset.csv', index=False)""",
        "model_build": """model = DBSCAN(eps=0.25, min_samples=5)
model.fit(X_scaled)
print("DBSCAN fitted.")""",
        "model_eval": """labels = model.labels_
# Guard: silhouette_score requires at least 2 distinct non-noise labels
unique_labels = set(labels)
if len(unique_labels - {-1}) >= 2:
    mask = labels != -1
    print('Silhouette Score:', silhouette_score(X_scaled[mask], labels[mask]))
else:
    print('Silhouette Score: N/A (fewer than 2 valid clusters found)')""",
        "model_vis": """plt.figure(figsize=(8, 6))
# Unique color palette for DBSCAN, with black/red reserved for noise
labels = model.labels_
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Outliers / Noise in Red
        col = [1, 0, 0, 1]
        marker = 'x'
        label = 'Noise / Outliers'
    else:
        marker = 'o'
        label = f'Cluster {k}'
        
    class_member_mask = (labels == k)
    xy = X_scaled[class_member_mask]
    plt.scatter(xy[:, 0], xy[:, 1], color=tuple(col), marker=marker, edgecolors='k', s=60, label=label)

plt.title("DBSCAN Clustering (eps=0.25, min_samples=5)")
plt.legend()
plt.show()""",
        "qa_interview": [
            ("What are Core, Border, and Noise points?", "1. Core: points with >= MinPts in their eps-neighborhood. 2. Border: points inside a Core's eps-neighborhood but with < MinPts. 3. Noise: all other points."),
            ("What are the advantages of DBSCAN over KMeans?", "1. Can discover clusters of arbitrary shapes (like moons). 2. Does not require pre-specifying cluster counts. 3. Explicitly detects noise and outliers."),
            ("How do you select parameters eps and min_samples?", "Use k-distance graph (sorting distance to k-th nearest neighbor). The optimal eps is the point of maximum curvature (knee) in the graph.")
        ],
        "qa_viva": [
            ("What value is assigned to noise points in scikit-learn's DBSCAN?", "Noise points are assigned the label `-1`."),
            ("What does 'eps' mean?", "Epsilon (eps) defines the radius of the neighborhood circle around any data point."),
            ("Is DBSCAN sensitive to scaling?", "Yes, because it relies on distance metrics to identify dense areas.")
        ]
    },
    "12_PCA": {
        "title": "Principal Component Analysis",
        "filename": "PCA.ipynb",
        "category": "pca",
        "description": "An unsupervised linear transformation that projects high-dimensional features onto lower-dimensional orthogonal principal components representing maximum variance.",
        "analogy": "Taking a 2D photograph of a 3D statue from the angle that captures the most detail.",
        "math_formula": "\\mathbf{\\Sigma} \\mathbf{v} = \\lambda \\mathbf{v} \\quad \\text{where } \\mathbf{\\Sigma} \\text{ is the Covariance Matrix, } \\mathbf{v} \\text{ are Eigenvectors, and } \\lambda \\text{ are Eigenvalues}",
        "worked_example": "Reducing 6-dimensional wine features to 2 principal components. PCA finds the first direction PC1 that captures 65% of data spread, and PC2 perpendicular to PC1 capturing 20% of spread. 85% variance is kept.",
        "dataset_code": """# Programmatic generation of high-dim wine features
from sklearn.datasets import make_blobs
np.random.seed(42)
# Generate 6 features with correlations representing wine attributes
X_raw, _ = make_blobs(n_samples=250, n_features=6, centers=3, cluster_std=1.5, random_state=42)
df = pd.DataFrame(X_raw, columns=['Alcohol', 'MalicAcid', 'Ash', 'Alcalinity', 'Magnesium', 'TotalPhenols'])
# Add correlation relationships
df['Alcohol'] = df['Alcohol'] * 0.4 + 13.0
df['MalicAcid'] = np.abs(df['MalicAcid'] * 0.5 + 2.0)
df['Alcalinity'] = df['Alcalinity'] * 2.0 + 19.0
df.to_csv('dataset.csv', index=False)""",
        "model_build": """model = PCA(n_components=2)
X_pca = model.fit_transform(X_scaled)
print("Explained Variance Ratio per PC:", model.explained_variance_ratio_)
print("PCs components mapping (eigenvectors):\\n", model.components_)""",
        "model_eval": """print('Explained Variance Ratio:', model.explained_variance_ratio_)
print('Cumulative Variance:', np.sum(model.explained_variance_ratio_))""",
        "model_vis": """fig, axes = plt.subplots(1, 2, figsize=(15, 6))
# Plot 1: Scree Plot
axes[0].bar(['PC1', 'PC2'], model.explained_variance_ratio_, color='indigo', alpha=0.7)
axes[0].set_title("Explained Variance Ratio per PC")
axes[0].set_ylabel("Variance Proportion")

# Plot 2: 2D Projected Scatter
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], color='purple', alpha=0.7)
axes[1].set_title("PCA Projection Scatter Space")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
plt.tight_layout()
plt.show()""",
        "qa_interview": [
            ("How does PCA determine the principal directions?", "By finding the eigenvectors of the covariance matrix. The eigenvectors represent directions of variance, and eigenvalues represent the magnitude of variance in those directions."),
            ("Why is feature standardization critical before PCA?", "PCA maximizes variance. If one feature is on a larger scale (e.g. salary), PCA will align PC1 along that feature axis simply due to variance magnitude, ignoring other features."),
            ("What is the difference between PCA and Autoencoders?", "PCA is a linear transformation, while Autoencoders can capture non-linear relationships using non-linear activation functions in neural network layers.")
        ],
        "qa_viva": [
            ("What does 'explained_variance_ratio_' show?", "The proportion of the dataset's total variance captured by each principal component."),
            ("Are principal components correlated?", "No, principal components are orthogonal (perpendicular) to each other, meaning they are completely uncorrelated."),
            ("How do you choose the number of PCs?", "By looking at cumulative explained variance (e.g., stopping when the components explain 90% or 95% of the total variance).")
        ]
    },
    "13_LDA": {
        "title": "Linear Discriminant Analysis",
        "filename": "LDA.ipynb",
        "category": "classification",
        "description": "A supervised dimensionality reduction and classification method that projects features onto directions that maximize class separation.",
        "analogy": "Rotating and viewing shadows of groups on a wall to find the angle where they overlap the least.",
        "math_formula": "J(\\mathbf{w}) = \\frac{\\mathbf{w}^T \\mathbf{S}_B \\mathbf{w}}{\\mathbf{w}^T \\mathbf{S}_W \\mathbf{w}} \\quad \\text{where } \\mathbf{S}_B \\text{ is Between-Class Scatter, and } \\mathbf{S}_W \\text{ is Within-Class Scatter}",
        "worked_example": "Classifying 3 crop classes using N, P, K. LDA finds the projection direction where crop clusters overlap the least, maximizing separation, then performs classification.",
        "dataset_code": """# Programmatic generation of agricultural crop features
from sklearn.datasets import make_classification
np.random.seed(42)
X_raw, y_raw = make_classification(n_samples=300, n_features=3, n_informative=3,
                                  n_redundant=0, n_classes=3, n_clusters_per_class=1,
                                  class_sep=1.5, random_state=42)
df = pd.DataFrame({
    'Nitrogen': np.round(X_raw[:, 0] * 15 + 50, 1),
    'Phosphorus': np.round(X_raw[:, 1] * 12 + 40, 1),
    'Potassium': np.round(X_raw[:, 2] * 10 + 30, 1),
    'CropType': y_raw
})
df.to_csv('dataset.csv', index=False)""",
        "model_build": """model = LinearDiscriminantAnalysis(n_components=2)
model.fit(X_train, y_train)
print("LDA transformation model completed.")""",
        "model_eval": """y_pred = model.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Classification Report:\\n", metrics.classification_report(y_test, y_pred))""",
        "model_vis": """# Visualizing LDA projections
X_test_lda = model.transform(X_test)
plt.figure(figsize=(8, 6))
plt.scatter(X_test_lda[:, 0], X_test_lda[:, 1], c=y_test, cmap='rainbow', edgecolors='k', s=60)
plt.title("LDA 2D Projection Space Scatter")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.show()""",
        "qa_interview": [
            ("What is the difference between PCA and LDA?", "PCA is unsupervised (ignores class labels and maximizes total variance). LDA is supervised (uses labels to maximize between-class variance while minimizing within-class variance)."),
            ("What is the maximum number of components LDA can output?", "Minimum of (Number of Features, Number of Classes - 1). For 3 classes, LDA can output at most 2 dimensions."),
            ("What are the assumptions of LDA?", "Normality of feature distributions, equal covariance matrices across all classes (homoscedasticity), and independence of sample observations.")
        ],
        "qa_viva": [
            ("Define LDA.", "Linear Discriminant Analysis: a supervised method that projects data to maximize class separability."),
            ("What does Between-Class Scatter represent?", "The variance between the mean coordinate points of different target classes."),
            ("What does Within-Class Scatter represent?", "The variance of data points around their respective class centroids.")
        ]
    },
    "14_AdaBoost": {
        "title": "AdaBoost Classifier",
        "filename": "AdaBoost.ipynb",
        "category": "classification",
        "description": "An adaptive boosting ensemble method that fits sequential weak learners (stumps) by updating weights on previously misclassified points.",
        "analogy": "A teacher giving extra homework and attention to topics that students got wrong on the last exam.",
        "math_formula": "H(\\mathbf{x}) = \\text{sign} \\left( \\sum_{t=1}^T \\alpha_t h_t(\mathbf{x}) \\right) \\quad \\text{where } \\alpha_t \\text{ is weak learner voting power}",
        "worked_example": "Telecom churn classification. AdaBoost trains a decision stump (depth 1 tree). Stumps that classify churn incorrectly are assigned higher weights. The next stump focuses on these hard cases.",
        "dataset_code": """# Programmatic generation of customer churn profiles
from sklearn.datasets import make_classification
np.random.seed(42)
X_raw, y_raw = make_classification(n_samples=500, n_features=3, n_informative=3,
                                  n_redundant=0, n_classes=2, weights=[0.75, 0.25],
                                  class_sep=1.2, random_state=42)
df = pd.DataFrame({
    'Tenure': np.round(np.clip(X_raw[:, 0] * 15 + 24, 1, 72), 0).astype(int),
    'MonthlyCharges': np.round(X_raw[:, 1] * 25 + 70, 2),
    'UsageVolume': np.round(np.clip(X_raw[:, 2] * 100 + 300, 10, 1000), 0),
    'Churned': y_raw
})
df.to_csv('dataset.csv', index=False)""",
        "model_build": """model = AdaBoostClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
print("AdaBoost ensemble trained successfully.")""",
        "model_eval": """y_pred = model.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Classification Report:\\n", metrics.classification_report(y_test, y_pred))""",
        "model_vis": """fig, axes = plt.subplots(1, 2, figsize=(15, 6))
# Plot 1: Feature Importances
importances = model.feature_importances_
axes[0].barh(X.columns, importances, color='maroon')
axes[0].set_title("AdaBoost Feature Importances")
axes[0].set_xlabel("Relative Importance Score")

# Plot 2: Confusion Matrix
cm = metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=axes[1])
axes[1].set_title("Confusion Matrix")
plt.tight_layout()
plt.show()""",
        "qa_interview": [
            ("What is a weak learner?", "A simple classifier that performs slightly better than random guessing (e.g., a Decision Stump — a decision tree with a depth of 1)."),
            ("How does AdaBoost compute learner weights (alpha)?", "Based on their classification error: alpha = 0.5 * ln((1 - error) / error). Stumps with lower error get higher voting weight."),
            ("Why is AdaBoost sensitive to noisy data and outliers?", "Because it dynamically increases the weights of misclassified points. Outliers will continuously get misclassified, forcing the model to focus heavily on fitting them (causing overfitting).")
        ],
        "qa_viva": [
            ("What does 'AdaBoost' stand for?", "Adaptive Boosting."),
            ("What is the default base estimator in scikit-learn's AdaBoost?", "DecisionTreeClassifier with `max_depth=1` (a Decision Stump)."),
            ("How are final predictions made?", "By combining the predictions of all stumps using a weighted majority vote.")
        ]
    },
    "15_Gradient_Boosting": {
        "title": "Gradient Boosting",
        "filename": "Gradient_Boosting.ipynb",
        "category": "classification",
        "description": "An ensemble boosting method that builds trees sequentially to predict the residual errors (gradients of loss function) of prior trees.",
        "analogy": "Playing golf: first stroke gets you close, second stroke corrects the remaining distance (error), and third corrects the residual.",
        "math_formula": "F_m(\\mathbf{x}) = F_{m-1}(\\mathbf{x}) + \\gamma_m h_m(\\mathbf{x}) \\quad \\text{where } h_m \\text{ fits the pseudo-residuals } r_{im} = -\\left[ \\frac{\\partial L(y_i, F(\\mathbf{x}_i))}{\\partial F(\\mathbf{x}_i)} \\right]_{F(\\mathbf{x}) = F_{m-1}(\\mathbf{x})}",
        "worked_example": "Predicting if house is expensive. First tree predicts average class probability (0.50). Error is y - 0.50. The next tree is trained to predict this error. Summing predictions corrects the classification boundary.",
        "dataset_code": """# Programmatic generation of house listings data
from sklearn.datasets import make_classification
np.random.seed(42)
X_raw, y_raw = make_classification(n_samples=500, n_features=4, n_informative=3,
                                  n_redundant=1, n_classes=2, weights=[0.6, 0.4],
                                  class_sep=1.2, random_state=42)
df = pd.DataFrame({
    'SqFt': np.round(X_raw[:, 0] * 500 + 2000, 0),
    'RoomsCount': np.round(np.clip(X_raw[:, 1] * 1.5 + 4, 2, 8), 0).astype(int),
    'Age': np.round(np.clip(X_raw[:, 2] * 15 + 30, 0, 100), 0).astype(int),
    'Income': np.round(np.clip(X_raw[:, 3] * 20000 + 80000, 20000, 200000), -2),
    'Expensive': y_raw
})
df.to_csv('dataset.csv', index=False)""",
        "model_build": """model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)
print("Gradient Boosting model trained.")""",
        "model_eval": """y_pred = model.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("F1-Score:", metrics.f1_score(y_test, y_pred))""",
        "model_vis": """fig, axes = plt.subplots(1, 2, figsize=(15, 6))
# Plot 1: Feature Importances
importances = model.feature_importances_
axes[0].barh(X.columns, importances, color='royalblue')
axes[0].set_title("Gradient Boosting Feature Importances")
axes[0].set_xlabel("Relative Gini Decrease")

# Plot 2: Confusion Matrix
cm = metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title("Confusion Matrix")
plt.tight_layout()
plt.show()""",
        "qa_interview": [
            ("How does Gradient Boosting build trees?", "Trees are fitted to the negative gradient (residuals) of the loss function. Each new tree corrects the residual errors of the existing ensemble."),
            ("What is the role of learning_rate (shrinkage)?", "It scales the contribution of each tree. A lower learning rate (e.g. 0.01) requires more trees but reduces overfitting and improves generalization."),
            ("What is Gradient Boosting's primary weakness?", "It is sequential (cannot be parallelized during training), making it slower to train than Random Forests.")
        ],
        "qa_viva": [
            ("What are residuals?", "The errors or differences between actual targets and predicted outputs."),
            ("What does the 'max_depth' parameter control?", "Limits the depth of individual trees (weak learners), typically set to 3-6 to prevent overfitting."),
            ("Does it support custom loss functions?", "Yes, Gradient Boosting can optimize any differentiable loss function.")
        ]
    },
    "16_XGBoost": {
        "title": "XGBoost",
        "filename": "XGBoost.ipynb",
        "category": "classification",
        "description": "An optimized, regularized Gradient Boosting framework designed for high speed and performance.",
        "analogy": "Building a custom golf cart engine that is supercharged and has built-in safety features to prevent crashing (overfitting).",
        "math_formula": "\\mathcal{L}^{(t)} = \\sum_{i=1}^n l\\left(y_i, \\hat{y}_i^{(t-1)} + f_t(\\mathbf{x}_i)\\right) + \\gamma T + \\frac{1}{2}\\lambda \\sum_{j=1}^T w_j^2",
        "worked_example": "Classifying loan default risk. XGBoost fits regularized trees, utilizing parallel CPU threads, handling missing values automatically, and pruning splits that don't improve gain above a threshold.",
        "dataset_code": """# Programmatic generation of loan default risk profiles
from sklearn.datasets import make_classification
np.random.seed(42)
X_raw, y_raw = make_classification(n_samples=600, n_features=4, n_informative=3,
                                  n_redundant=1, n_classes=2, weights=[0.85, 0.15],
                                  class_sep=1.3, random_state=42)
df = pd.DataFrame({
    'FICO': np.round(X_raw[:, 0] * 70 + 680, 0),
    'LTV': np.round(np.clip(X_raw[:, 1] * 0.12 + 0.75, 0.1, 1.2), 2),
    'Inquiries': np.round(np.clip(X_raw[:, 2] * 2 + 1, 0, 10), 0).astype(int),
    'Income': np.round(np.clip(X_raw[:, 3] * 25000 + 75000, 15000, 250000), -2),
    'Defaulted': y_raw
})
df.to_csv('dataset.csv', index=False)""",
        "model_build": """# Build XGBoost Classifier
model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                      eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)
print("XGBoost classifier trained.")""",
        "model_eval": """y_pred = model.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("ROC-AUC:", metrics.roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
print("Confusion Matrix:\\n", metrics.confusion_matrix(y_test, y_pred))""",
        "model_vis": """fig, axes = plt.subplots(1, 2, figsize=(15, 6))
# Plot 1: Feature Importances
importances = model.feature_importances_
axes[0].barh(X.columns, importances, color='darkorange')
axes[0].set_title("XGBoost Feature Importances")
axes[0].set_xlabel("F-Score Importance")

# Plot 2: Confusion Matrix
cm = metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=axes[1])
axes[1].set_title("Confusion Matrix")
plt.tight_layout()
plt.show()""",
        "qa_interview": [
            ("What makes XGBoost so fast?", "It implements parallel tree building, cache-aware access, block structure data layouts, and a weighted quantile sketch algorithm for fast split candidate selection."),
            ("How does XGBoost prevent overfitting?", "By adding L1 (Alpha) and L2 (Lambda) regularization terms directly to the tree objective function loss, and using gamma thresholds to prune splits."),
            ("How does XGBoost handle missing values?", "It automatically learns a default direction (left or right split) for missing values based on which direction reduces loss the most during training.")
        ],
        "qa_viva": [
            ("What does XGBoost stand for?", "Extreme Gradient Boosting."),
            ("Name a key regularization parameter in XGBoost.", "`reg_lambda` (L2 regularization) or `reg_alpha` (L1 regularization)."),
            ("What is the role of the 'gamma' parameter?", "Specifies the minimum loss reduction required to make a split on a leaf node.")
        ]
    }
}

def make_notebook_json(folder, data):
    cells = []
    
    # 1. Introduction
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# {data['title']}: A Comprehensive Guide\n",
            "\n",
            "## 1. Introduction\n",
            "\n",
            f"This notebook covers the step-by-step implementation of **{data['title']}** in Python.\n",
            "\n",
            "### Concept Overview\n",
            f"{data['description']}\n",
            "\n",
            "### Mathematical Formula\n",
            f"The core mathematical relationship or objective is defined as:\n",
            f"$$\n{data['math_formula']}\n$$\n",
            "\n",
            "### Real-World Analogy\n",
            f"**Analogy:** {data['analogy']}\n",
            "\n",
            "### Worked Example\n",
            f"**Worked Example:** {data['worked_example']}"
        ]
    })
    
    # 2. Import Libraries
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Import Libraries\n",
            "\n",
            "We import the standard data science and machine learning libraries."
        ]
    })
    
    # Library Imports mapping
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
        "10_Hierarchical_Clustering": "from sklearn.preprocessing import StandardScaler\nfrom sklearn.cluster import AgglomerativeClustering\nfrom sklearn.metrics import silhouette_score\nimport joblib",
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
    
    # 3. Create Dataset
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Create Synthetic Dataset\n",
            "\n",
            "We generate a realistic synthetic dataset to demonstrate the model's behavior and save it locally."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [data["dataset_code"]]
    })
    
    # 4. Load Dataset
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Load & Inspect Dataset\n",
            "\n",
            "We load the dataset using pandas to inspect the shape, variables, and summary statistics."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df = pd.read_csv('dataset.csv')\n",
            "print('Dataset Shape:', df.shape)\n",
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
            "Audit for missing values and duplicates."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('Null values:\\n', df.isnull().sum())\n",
            "print('Duplicate count:', df.duplicated().sum())"
        ]
    })
    
    # 6. EDA
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Exploratory Data Analysis (EDA)\n",
            "\n",
            "We perform visual analysis of the dataset, examining correlation heatmaps and target-colored feature distributions to understand the underlying boundaries."
        ]
    })
    
    # Custom EDA code based on category
    eda_code = "sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')\nplt.title('Correlation Matrix Heatmap')\nplt.show()\n\n"
    if data["category"] == "regression":
        eda_code += "# Histograms of features and target\nfig, axes = plt.subplots(1, len(df.columns), figsize=(5 * len(df.columns), 5))\nif len(df.columns) == 1:\n    sns.histplot(df.iloc[:, 0], kde=True, ax=axes, color='skyblue')\n    axes.set_title(f'Histogram of {df.columns[0]}')\nelse:\n    for idx, col in enumerate(df.columns):\n        sns.histplot(df[col], kde=True, ax=axes[idx], color='skyblue')\n        axes[idx].set_title(f'Histogram of {col}')\nplt.tight_layout()\nplt.show()"
    elif data["category"] == "classification":
        eda_code += "# 1. Feature Histograms (Continuous distributions colored by class)\nX_cols = df.columns[:-1]\ntarget_col = df.columns[-1]\nfig, axes = plt.subplots(1, len(X_cols), figsize=(5 * len(X_cols), 5))\nif len(X_cols) == 1:\n    sns.histplot(data=df, x=X_cols[0], hue=target_col, kde=True, ax=axes, multiple='stack', palette='coolwarm')\n    axes.set_title(f'Histogram of {X_cols[0]}')\nelse:\n    for idx, col in enumerate(X_cols):\n        sns.histplot(data=df, x=col, hue=target_col, kde=True, ax=axes[idx], multiple='stack', palette='coolwarm')\n        axes[idx].set_title(f'Histogram of {col}')\nplt.tight_layout()\nplt.show()\n\n# 2. Class Balance Pie Chart and Bar Graph\nfig, axes = plt.subplots(1, 2, figsize=(14, 6))\n# Pie chart\nclass_counts = df[target_col].value_counts()\naxes[0].pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff', '#99ff99'], startangle=90, wedgeprops={'edgecolor': 'black'})\naxes[0].set_title('Class Balance (Pie Chart)')\n# Bar chart\nsns.countplot(data=df, x=target_col, ax=axes[1], palette='Set2')\naxes[1].set_title('Class Counts (Bar Graph)')\nplt.tight_layout()\nplt.show()"
    else: # clustering and PCA
        eda_code += "# Histograms of features\nfig, axes = plt.subplots(1, len(df.columns), figsize=(5 * len(df.columns), 5))\nif len(df.columns) == 1:\n    sns.histplot(df.iloc[:, 0], kde=True, ax=axes, color='lightgreen')\n    axes.set_title(f'Histogram of {df.columns[0]}')\nelse:\n    for idx, col in enumerate(df.columns):\n        sns.histplot(df[col], kde=True, ax=axes[idx], color='lightgreen')\n        axes[idx].set_title(f'Histogram of {col}')\nplt.tight_layout()\nplt.show()"
        
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [eda_code]
    })
    
    # 7. Feature Engineering
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Feature Engineering & Scaling\n",
            "\n",
            "Prepare the features and apply standardization if required."
        ]
    })
    
    engineering_code = ""
    if data["category"] in ["clustering", "pca"]:
        engineering_code = "X = df.copy()\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\nprint('Standard scaling applied.')"
    else:
        if folder in ["03_Logistic_Regression", "04_KNN", "08_SVM", "13_LDA"]:
            engineering_code = "X = df.iloc[:, :-1]\ny = df.iloc[:, -1]\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\nprint('Scaling applied successfully.')"
        else:
            engineering_code = "X = df.iloc[:, :-1]\ny = df.iloc[:, -1]\nprint('Scaling not strictly needed for this model.')"
            
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
    if data["category"] in ["clustering", "pca"]:
        split_code = "# Unsupervised model - fitting directly on the scaled cohort\nprint('Skipping split step.')"
    else:
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
            "Instantiate and fit the model estimator."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [data["model_build"]]
    })
    
    # 10. Prediction & 11. Model Evaluation
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 10. Prediction & 11. Model Evaluation\n",
            "\n",
            "Check metrics to evaluate model performance."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [data["model_eval"]]
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
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [data["model_vis"]]
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
    
    # 14-17: Context Sections
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 14. Common Mistakes & Best Practices\n",
            "- Forgetting to apply the fitted scaler to new evaluation data points, leading to prediction errors.\n",
            "- Overfitting simple models with excessive features or polynomial terms.\n",
            "\n",
            "## 15. Advantages\n",
            "- Fast and easy to interpret baseline results.\n",
            "- Requires minimal parameter tuning compared to deep neural networks.\n",
            "\n",
            "## 16. Limitations\n",
            "- Assumes linear or simple spatial patterns that do not always match real-world anomalies.\n",
            "- Sensitive to noisy labels and extreme outliers.\n",
            "\n",
            "## 17. Real-World Applications\n",
            "- Predicting housing market price trends based on space features.\n",
            "- Classifying credit card transactions as fraudulent vs benign."
        ]
    })
    
    # 18. Interview Q&As
    qa_i_cells = []
    qa_i_cells.append("## 18. Algorithm-Specific Interview Questions & Answers\n")
    for idx, (q, a) in enumerate(data["qa_interview"], 1):
        qa_i_cells.append(f"**Q{idx}: {q}**\n\n*A: {a}*\n\n")
        
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": qa_i_cells
    })
    
    # 19. Viva Q&As
    qa_v_cells = []
    qa_v_cells.append("## 19. Algorithm-Specific Viva Voce Questions & Answers\n")
    for idx, (q, a) in enumerate(data["qa_viva"], 1):
        qa_v_cells.append(f"**Q{idx}: {q}**\n\n*A: {a}*\n\n")
        
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": qa_v_cells
    })
    
    # 20. Practice Exercises
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 20. Practice Exercises\n",
            "1. **Hyperparameter sweep**: Modify model parameters (such as `max_depth` or neighbor count `K`) and plot the test metric variance curve.\n",
            "2. **Manual metric calculation**: Compute precision and recall scores manually from the confusion matrix values and verify results using scikit-learn.\n",
            "3. **Outlier test**: Add an extreme value row to the dataset, re-train, and record the boundary displacement shift."
        ]
    })
    
    # 21. Conclusion
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 21. Conclusion\n",
            "\n",
            "In this notebook, we implemented the complete modeling cycle: generating datasets, performing scaling, training the model, evaluating predictions with multiple metrics, saving outputs, and studying typical interview Q&A profiles."
        ]
    })
    
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
            
        # Clean deprecated files
        existing_notebooks = glob.glob(os.path.join(folder, "*.ipynb"))
        for nb_path in existing_notebooks:
            basename = os.path.basename(nb_path)
            if basename != data["filename"]:
                try:
                    os.remove(nb_path)
                    print(f"Removed deprecated file: {nb_path}")
                except Exception as e:
                    print(f"Failed to remove {nb_path}: {e}")
                    
        if folder == "06_Decision_Tree":
            stray_path = os.path.join(folder, "Linear_Regression.ipynb")
            if os.path.exists(stray_path):
                try:
                    os.remove(stray_path)
                    print(f"Removed stray file: {stray_path}")
                except:
                    pass

        # Write dataset.csv by executing the generation code locally in the folder context
        original_cwd = os.getcwd()
        os.chdir(folder)
        try:
            exec_globals = {"np": np, "pd": pd, "make_classification": make_classification, "make_moons": make_moons, "make_blobs": make_blobs}
            exec(data["dataset_code"], exec_globals)
            print(f"Generated dataset for {folder}")
        except Exception as e:
            print(f"Failed to generate dataset for {folder}: {e}")
        finally:
            os.chdir(original_cwd)

        target_nb_path = os.path.join(folder, data["filename"])
        
        # Write .ipynb
        nb_dict = make_notebook_json(folder, data)
        with open(target_nb_path, "w", encoding="utf-8") as f:
            json.dump(nb_dict, f, indent=1)
        print(f"Created notebook: {target_nb_path}")
        
    print("\nGeneration finished successfully! Re-created correct files.")

if __name__ == "__main__":
    main()
