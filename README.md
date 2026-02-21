# Mutual Fund Performance & Rating Classification

A comprehensive data science project focused on predicting Morningstar ratings for European Mutual Funds using advanced feature engineering, dimensionality reduction, and gradient boosting techniques.

## Project Overview
The objective of this project is to analyze a dataset of over 30,000 mutual funds to predict their performance ratings (1-5). The project covers the entire machine learning pipeline: from complex data cleaning and multi-currency conversion to sophisticated feature engineering, handling class imbalance, and model optimization.

## Key Features
- **Extensive Data Cleaning**: Managed missing values via median/mode imputation and handled multi-currency fund sizes by converting all values to USD based on historical exchange rates.
- **Advanced Feature Engineering**: Created over 20+ custom features including fund age, risk-category interactions, regional diversity scores, and sector exposure ratios.
- **Principal Component Analysis (PCA)**: Applied PCA to highly correlated financial metrics (e.g., ROE vs. ROA, Management Fees vs. Ongoing Costs) to reduce multicollinearity while retaining 95%+ variance.
- **Imbalance Handling**: Implemented **SMOTE** and **SMOTEENN** (SMOTE + Edited Nearest Neighbors) to address the significant skew in rating distributions.
- **High-Performance Modeling**: Compared Random Forest, XGBoost, and LightGBM, achieving a peak accuracy of **~95%** with LightGBM.

## Technology Stack
- **Language**: Python 3.11
- **Libraries**: 
  - `Pandas`, `NumPy` (Data Manipulation)
  - `Scikit-Learn` (Preprocessing, PCA, Tuning)
  - `Imbalanced-Learn` (SMOTE, SMOTEENN)
  - `Matplotlib`, `Seaborn` (Visualization)
  - `XGBoost`, `LightGBM` (Gradient Boosting)

##  Dataset Analysis
The analysis utilizes the **Morningstar - European Mutual Funds** dataset, which includes:
- **Financial Metrics**: Returns (1yr, 3yr, 5yr, 10yr), Ratios (P/E, P/B, PCF), and ESG Scores.
- **Fund Metadata**: Inception dates, manager details, investment strategies, and benchmarks.
- **Target Variable**: `rating` (Morningstar 1-5 star rating).

## Workflow
1. **Exploratory Data Analysis (EDA)**: Identified high-cardinality categorical features and heavily skewed numerical distributions.
2. **Preprocessing**: 
   - Dropped redundant identifiers (`isin`, `ticker`).
   - Capped outliers using the **IQR method** to prevent model distortion.
   - Applied **Label Encoding** for high-cardinality categorical data.
3. **Dimensionality Reduction**: Used custom PCA components for feature pairs with correlation > 0.85.
4. **Resampling**: Used SMOTEENN to generate synthetic minority samples and clean noisy overlaps in the decision boundary.
5. **Model Evaluation**: Utilized 3-way splits (Train/Val/Test) to ensure robust generalization.

## Key Insights
1. **High Dimensionality & PCA**: The dataset contained a vast array of financial ratios and ESG factors. Feature selection and dimensionality reduction via **Principal Component Analysis (PCA)** were critical for effective modeling and managing multicollinearity.
2. **Class Imbalance Management**: The target `rating` variable was significantly imbalanced. While **SMOTE** helped generate minority samples, **SMOTEENN** proved superior by also removing "noise" (overlapping points), creating a more distinct decision boundary between rating classes.
3. **Non-Linear Interactions**: The model successfully captured complex interactions between fund categories, risk profiles, and sustainability variability, which are major drivers of non-linear rating distributions.
4. **Resampling Logic**: By combining synthetic oversampling with cleaning (ENN), we maintained high accuracy across both majority and minority rating classes.

## Algorithm Overview (Pseudocode)
The project implemented several sophisticated algorithms to ensure data balance and predictive accuracy:

### Data Balancing
- **SMOTE**: Synthetically generates new minority class instances by interpolating between existing $k$-nearest neighbors.
- **SMOTEENN**: A hybrid approach that first applies SMOTE to oversample and then uses **Edited Nearest Neighbors (ENN)** to prune samples that misclassify their neighbors, resulting in a cleaner dataset.

### Machine Learning Models
- **Random Forest**: An ensemble of decision trees using bagging and random feature selection.
- **XGBoost**: A scalable, end-to-end gradient boosting system that uses second-order Taylor expansion for the loss function.
- **LightGBM**: A fast, distributed gradient boosting framework that uses histogram-based algorithms and leaf-wise tree growth.
  - *Parameters used*: `n_estimators=230`, `max_depth=2`, `learning_rate=0.1`, `subsample=0.8`.

## Installation & Usage
1. Clone the repository.
2. Ensure the dataset `Morningstar - European Mutual Funds.csv` is in the root directory.
3. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm imbalanced-learn
   ```
4. Run the Jupyter Notebook: `Team06_classificationProject (1).ipynb`.

## References
- **Dataset Source**: [Kaggle - European Funds Dataset from Morningstar](https://www.kaggle.com/datasets/stefanoleone992/european-funds-dataset-from-morningstar)
- **Library Documentation**: [Scikit-Learn Official Site](https://scikit-learn.org)
- **Financial Research**:
  - [Trailing vs. Rolling Returns - ET Money](https://www.etmoney.com/learn/mutual-funds/annual-vs-trailing-vs-rolling-returns-meaning-calculation-importance/)
  - [Morningstar Sustainability Rating Methodology](https://www.sustainalytics.com/investor-solutions/analytic-reporting-solutions/morningstar-sustainability-rating-for-funds)
  - [Credit Rating Scale Interpretations](https://ratingagency.morningstar.com/PublicDocDisplay.aspx?i=%2b7cwsQ2XW8A%3d&m=i0Pyc%2bx7qZZ4%2bsXnymazBA%3d%3d&s=LviRtUKXqs8kml5dHt7FTeE2SZmY0Fvqd4iX49Mk%2f9UapyiFTEO6TA%3d%3d)

---
*Created as part of the Mutual Fund Classification Project - Team 06.*
