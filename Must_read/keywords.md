# Machine Learning Keywords Glossary & Interview Guide

> A comprehensive, interview-ready glossary of essential Machine Learning terms, concepts, algorithms, metrics, and production practices.

---

## Table of Contents
1. [Foundational & Data Engineering Terms](#1-foundational--data-engineering-terms)
2. [Model Training & Learning Dynamics](#2-model-training--learning-dynamics)
3. [Supervised Learning Algorithms](#3-supervised-learning-algorithms)
4. [Unsupervised Learning & Clustering](#4-unsupervised-learning--clustering)
5. [Evaluation Metrics & Performance Analysis](#5-evaluation-metrics--performance-analysis)
6. [Advanced Concepts, Deep Learning & MLOps](#6-advanced-concepts-deep-learning--mlops)

---

## 1. Foundational & Data Engineering Terms

### Features (Independent Variables) & Target (Dependent Variable)
- **Definition:** **Features ($X$)** are the input variables used to make predictions. The **Target ($y$)** is the outcome variable the model tries to predict.
- **Intuition:** In housing price prediction, square footage and number of bedrooms are features; the final sale price is the target.
- **Interview Tip:** *"Features are inputs ($X$) and targets are ground truth outputs ($y$). Feature engineering is the process of creating informative $X$ representation."*

---

### Data Leakage
- **Definition:** The accidental inclusion of information from outside the training dataset (such as the target variable or future test data) into the model training pipeline.
- **Why It Matters:** Causes unrealistically high training/validation scores that drop sharply in production.
- **Interview Tip:** *"Data leakage happens when future or unseen test information leaks into model training, usually via global preprocessing or target contamination."*

---

### Feature Scaling (Standardization vs. Normalization)
- **Standardization (Z-score Scaling):** Rescales data to have a mean of 0 and standard deviation of 1:
  $$x' = \frac{x - \mu}{\sigma}$$
- **Normalization (Min-Max Scaling):** Rescales data into a fixed range $[0, 1]$:
  $$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$
- **Interview Tip:** *"Standardization is preferred when data follows a Gaussian distribution or for algorithms sensitive to variance (e.g., SVM, Linear Regression); Min-Max is preferred when bounded intervals are required (e.g., neural network inputs, image pixels)."*

---

### Categorical Encoding (One-Hot, Label, Target Encoding)
- **One-Hot Encoding:** Creates binary columns for each unique category. Best for nominal data without intrinsic order.
- **Label / Ordinal Encoding:** Assigns integers ($0, 1, 2...$) to categories. Best for ordered categories (e.g., Low, Medium, High).
- **Target Encoding:** Replaces categories with the mean target value for that category.
- **Interview Tip:** *"Use One-Hot for nominal features with low cardinality, Ordinal for ranked features, and Out-of-Fold Target Encoding for high-cardinality features to avoid dimensionality explosions."*

---

### Imbalanced Data & Resampling (SMOTE, Class Weights)
- **SMOTE (Synthetic Minority Over-sampling Technique):** Synthesizes new minority instances by interpolating between nearest neighbors.
- **Class Weights:** Modifies the loss function to penalize misclassifications of minority samples heavily ($Loss = w \cdot L$).
- **Interview Tip:** *"SMOTE generates synthetic samples in feature space, while cost-sensitive learning adjusts loss gradients directly. Never apply SMOTE on test data!"*

---

### Curse of Dimensionality
- **Definition:** As the number of features (dimensions) increases, the volume of feature space grows exponentially, making data points sparse and distance metrics (e.g., Euclidean) less meaningful.
- **Solution:** Feature selection or dimensionality reduction (e.g., PCA).
- **Interview Tip:** *"In high-dimensional spaces, all data points become equidistant from each other, making distance-based models like KNN perform poorly."*

---

### Dimensionality Reduction (PCA, t-SNE, UMAP)
- **PCA (Principal Component Analysis):** Linear reduction technique that projects data onto orthogonal directions of maximum variance.
- **t-SNE / UMAP:** Non-linear techniques optimized for preserving local neighborhood structures, ideal for 2D/3D visualization.
- **Interview Tip:** *"PCA is linear, fast, and preserves global variance; t-SNE/UMAP are non-linear and excel at cluster visualization."*

---

## 2. Model Training & Learning Dynamics

### Bias-Variance Tradeoff
- **Bias:** Error introduced by approximating a real-world problem with an overly simple model (causes **Underfitting**).
- **Variance:** Sensitivity of the model to small fluctuations in the training set (causes **Overfitting**).
- **Total Error Equation:**
  $$\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$
- **Interview Tip:** *"High bias leads to underfitting; high variance leads to overfitting. Optimal ML design finds the sweet spot that minimizes total generalization error."*

---

### Overfitting vs. Underfitting
- **Overfitting:** High performance on training data, poor performance on test data (model memorized noise).
- **Underfitting:** Poor performance on both training and test data (model is too simple).
- **Fixes:**
  - *Fix Overfitting:* Add regularization, reduce feature count, get more data, use cross-validation, prune trees.
  - *Fix Underfitting:* Increase model complexity, add feature interactions, decrease regularization.

---

### Loss Function vs. Cost Function vs. Metric
- **Loss Function:** Measures error for a **single** sample (e.g., Binary Cross-Entropy loss for one row).
- **Cost Function:** The average loss over the **entire** dataset plus regularization terms.
- **Metric:** Human-readable score used to evaluate business performance (e.g., Accuracy, $F_1$-score, ROC-AUC).
- **Interview Tip:** *"Loss is optimized per sample, cost is optimized globally during gradient updates, and metrics assess business utility."*

---

### Gradient Descent (Batch, SGD, Mini-Batch)
- **Gradient Descent:** Optimization algorithm that iteratively updates model parameters ($\theta$) in the direction of steepest loss descent:
  $$\theta := \theta - \alpha \nabla_{\theta} J(\theta)$$
- **Batch GD:** Computes gradients using the entire dataset per update (Slow, exact).
- **Stochastic GD (SGD):** Updates parameters using 1 sample per step (Fast, noisy).
- **Mini-Batch GD:** Updates parameters using small batches (e.g., 32, 64, 128 samples). Best balance of speed and stability.

---

### Regularization ($L_1$ Lasso, $L_2$ Ridge, ElasticNet)
- **$L_1$ Regularization (Lasso):** Adds absolute magnitude penalty ($\lambda \sum |w_i|$). Shrinks uninformative weights strictly to **zero** (acts as feature selection).
- **$L_2$ Regularization (Ridge):** Adds squared magnitude penalty ($\lambda \sum w_i^2$). Shrinks weights smoothly toward zero without setting them strictly to zero.
- **ElasticNet:** Combines $L_1$ and $L_2$ penalties.
- **Interview Tip:** *"L1 produces sparse models by eliminating features; L2 handles multicollinearity by shrinking correlated feature weights together."*

---

### Cross-Validation ($K$-Fold, Stratified $K$-Fold, TimeSeriesSplit)
- **$K$-Fold CV:** Splits data into $K$ equal subsets, training on $K-1$ folds and testing on the remaining fold $K$ times.
- **Stratified $K$-Fold:** Ensures target class proportions are preserved inside each fold.
- **TimeSeriesSplit:** Expanding/rolling window split that respects chronological time order.

---

### Hyperparameter vs. Parameter
- **Parameter:** Learned automatically by the model during training (e.g., weights $w$, bias $b$ in Linear Regression).
- **Hyperparameter:** Configured manually by the engineer before training starts (e.g., learning rate $\alpha$, tree depth, $k$ in KNN).

---

## 3. Supervised Learning Algorithms

### Linear Regression & Ordinary Least Squares (OLS)
- **Definition:** Predicts a continuous output by fitting a linear equation to observed data.
- **OLS:** Method for finding optimal weights by minimizing the Sum of Squared Residuals (SSR):
  $$\text{SSR} = \sum (y_i - \hat{y}_i)^2$$
- **Assumptions:** Linearity, Independence of errors, Homoscedasticity (equal variance of residuals), Normality of residuals, No perfect Multicollinearity.

---

### Logistic Regression & Sigmoid Function
- **Definition:** Classification model that predicts the probability of a binary outcome using the **Sigmoid (Logit)** function:
  $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
- **Output:** Maps real-valued inputs to a probability range between $0$ and $1$.
- **Interview Tip:** *"Logistic Regression fits a linear decision boundary in log-odds space using Binary Cross-Entropy loss."*

---

### Decision Trees (Gini Impurity vs. Entropy)
- **Gini Impurity:** Measures frequency with which a randomly chosen element would be incorrectly labeled:
  $$\text{Gini} = 1 - \sum p_i^2$$
- **Entropy & Information Gain:** Measures disorder/uncertainty. Information Gain measures reduction in entropy after a split:
  $$\text{Entropy} = -\sum p_i \log_2(p_i)$$
- **Interview Tip:** *"Gini is computationally faster as it avoids log calculations; Entropy produces slightly more balanced trees."*

---

### Ensemble Methods: Bagging vs. Boosting
- **Bagging (Bootstrap Aggregating):** Trains multiple independent base models in **parallel** on bootstrapped data subsets and averages/votes predictions (e.g., Random Forest). Focuses on **reducing variance**.
- **Boosting:** Trains base models **sequentially**, where each new model focuses on correcting the errors of previous models (e.g., XGBoost, LightGBM). Focuses on **reducing bias**.

```
Bagging:   Data ──> [Bootstraps] ──> [Model 1, Model 2, Model 3] (Parallel) ──> Average
Boosting:  Data ──> [Model 1] ──> Error ──> [Model 2] ──> Error ──> [Model 3] (Sequential)
```

---

### Random Forest
- **Definition:** Bagging ensemble of decision trees with feature randomization (random subsample of features at each split).
- **Key Advantages:** Robust to overfitting, handles missing values well, provides built-in feature importance via Mean Decrease in Impurity.
- **Out-of-Bag (OOB) Error:** Evaluation score computed using samples not included in a tree's bootstrap sample.

---

### Gradient Boosting (GBM, XGBoost, LightGBM, CatBoost)
- **GBM:** Sequentially fits trees to the **pseudo-residuals** (gradients) of the loss function.
- **XGBoost (eXtreme Gradient Boosting):** Optimized GBM with second-order Taylor expansion (gradients + Hessians), built-in regularization, and parallel tree building.
- **LightGBM:** Leaf-wise tree growth algorithm using histogram-based feature binning (extremely fast for large data).
- **CatBoost:** Optimized out-of-the-box handling for categorical features without explicit pre-encoding.

---

### Support Vector Machines (SVM) & Kernel Trick
- **Hyperplane:** Decision boundary separating data classes in $N$-dimensional space.
- **Margin:** Distance between the decision hyperplane and the closest data points (**Support Vectors**).
- **Kernel Trick:** Projects non-linearly separable data into higher-dimensional space where it becomes linearly separable, without explicitly computing high-dimensional coordinates (e.g., RBF kernel).

---

### Naive Bayes & Bayes' Theorem
- **Bayes' Theorem:**
  $$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$
- **Why "Naive"?:** Assumes all features are **conditionally independent** given the class label.
- **Laplace Smoothing:** Adds a pseudo-count to prevent zero probability propagation for unseen categories.

---

## 4. Unsupervised Learning & Clustering

### $K$-Means Clustering
- **Definition:** Partitioning algorithm that clusters data into $K$ distinct clusters by minimizing Within-Cluster Sum of Squares (WCSS / Inertia).
- **Steps:** Initialize $K$ centroids $\rightarrow$ Assign points to nearest centroid $\rightarrow$ Recompute centroids $\rightarrow$ Repeat until convergence.
- **Determining $K$:** **Elbow Method** (inflection point of inertia curve) and **Silhouette Analysis**.

---

### Hierarchical Clustering (Agglomerative vs. Divisive)
- **Agglomerative (Bottom-Up):** Starts with every point as its own cluster and recursively merges nearest pairs.
- **Divisive (Top-Down):** Starts with one all-inclusive cluster and recursively splits.
- **Dendrogram:** Tree diagram visualizing cluster hierarchy and merge distances.

---

### DBSCAN (Density-Based Spatial Clustering)
- **Definition:** Group points based on spatial density threshold parameters: **$\varepsilon$ (Epsilon)** distance radius and **MinPts** (minimum density points).
- **Point Categories:**
  - *Core Point:* Has $\ge \text{MinPts}$ within distance $\varepsilon$.
  - *Border Point:* Within $\varepsilon$ of a Core Point, but has $<\text{MinPts}$.
  - *Noise Point:* Neither core nor border point (treated as outlier).
- **Key Advantage:** Discovers arbitrary shapes and naturally isolates noise without requiring $K$ upfront.

---

### Silhouette Score
- **Definition:** Evaluates cluster quality by measuring how similar a sample is to its own cluster ($a$) compared to neighboring clusters ($b$):
  $$s = \frac{b - a}{\max(a, b)}$$
- **Score Range:** $-1$ to $+1$. $+1$ means well-clustered; $0$ means overlapping clusters; $-1$ means wrong cluster assignment.

---

## 5. Evaluation Metrics & Performance Analysis

### Confusion Matrix
- Matrix layout detailing classification outcomes:

| | Predicted Positive | Predicted Negative |
| :--- | :--- | :--- |
| **Actual Positive** | **True Positive (TP)** | **False Negative (FN)** (Type II Error) |
| **Actual Negative** | **False Positive (FP)** (Type I Error) | **True Negative (TN)** |

---

### Precision, Recall, and $F_1$-Score
- **Precision:** Of all positive predictions, how many were actually positive?
  $$\text{Precision} = \frac{TP}{TP + FP}$$
- **Recall (Sensitivity / True Positive Rate):** Of all actual positive cases, how many did the model catch?
  $$\text{Recall} = \frac{TP}{TP + FN}$$
- **$F_1$-Score:** Harmonic mean of Precision and Recall:
  $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

### ROC Curve & ROC-AUC
- **ROC Curve:** Plots True Positive Rate ($\text{Recall}$) vs. False Positive Rate ($\text{FPR} = \frac{FP}{FP + TN}$) across all classification thresholds.
- **ROC-AUC:** Area Under the ROC Curve. Measures the probability that a randomly chosen positive sample is ranked higher than a randomly chosen negative sample. Range: $0.5$ (random guess) to $1.0$ (perfect separator).

---

### Precision-Recall Curve & PR-AUC
- **Definition:** Plots Precision vs. Recall across classification thresholds.
- **When to Use:** Preferred over ROC-AUC when evaluating heavily imbalanced datasets where the negative class dominates.

---

### Regression Metrics (MAE, MSE, RMSE, $R^2$, Adjusted $R^2$)
- **MAE (Mean Absolute Error):** Average of absolute residuals. Robust to outliers.
  $$\text{MAE} = \frac{1}{n} \sum |y_i - \hat{y}_i|$$
- **MSE (Mean Squared Error):** Average of squared residuals. Heavily penalizes large errors.
  $$\text{MSE} = \frac{1}{n} \sum (y_i - \hat{y}_i)^2$$
- **RMSE (Root Mean Squared Error):** Square root of MSE. Keeps error units identical to target variable.
  $$\text{RMSE} = \sqrt{\text{MSE}}$$
- **$R^2$ (Coefficient of Determination):** Proportion of target variance explained by model:
  $$R^2 = 1 - \frac{\text{SS}_{res}}{\text{SS}_{tot}}$$
- **Adjusted $R^2$:** Penalizes adding features that do not improve model predictive power.

---

## 6. Advanced Concepts, Deep Learning & MLOps

### Activation Functions (ReLU, Sigmoid, Softmax)
- **ReLU (Rectified Linear Unit):** $f(x) = \max(0, x)$. Standard for hidden layers in deep neural networks; avoids vanishing gradient for $x > 0$.
- **Leaky ReLU:** Fixes "Dying ReLU" problem by assigning a small non-zero slope for $x < 0$ ($f(x) = \max(0.01x, x)$).
- **Softmax:** Converts raw neural network output logits into normalized multi-class probabilities summing to 1.

---

### Vanishing & Exploding Gradients
- **Vanishing Gradients:** Gradients shrink exponentially toward zero during backpropagation through deep layers, stopping early weights from updating (common with Sigmoid/Tanh).
- **Exploding Gradients:** Gradients grow exponentially large, causing unstable weight updates and `NaN` loss (fixed via **Gradient Clipping**).

---

### Transfer Learning & Fine-Tuning
- **Transfer Learning:** Reusing pre-trained weights from a model trained on a large dataset (e.g., ImageNet, BERT) on a new related task.
- **Fine-Tuning:** Unfreezing selected top layers of a pre-trained network and updating weights on domain-specific data with a low learning rate.

---

### Data Drift vs. Concept Drift
- **Data Drift (Covariate Shift):** The input distribution changes over time ($P(X)$ changes), while the relationship to target remains the same ($P(y|X)$ unchanged). Example: Changes in user demographics.
- **Concept Drift:** The fundamental statistical relationship between input features and target changes ($P(y|X)$ changes). Example: Consumer purchasing behavior changes dramatically after a pandemic.
- **MLOps Action:** Triggers automated model retraining pipelines and data quality monitoring alerts.
