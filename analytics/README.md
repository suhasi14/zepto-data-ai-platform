
# Module 2: Titanic Data Analytics and Machine Learning

## 1. Project Overview

This module performs exploratory data analysis and predictive modelling using the Titanic dataset.

The project includes:
- Data loading and offline backup
- Missing-value analysis and treatment
- Data visualization
- Outlier detection
- Statistical analysis
- Correlation analysis
- Classification models
- Class imbalance handling
- Hyperparameter tuning
- Fare prediction using regression
- Model saving and reloading

## 2. Dataset

Dataset: Titanic

Source: Seaborn built-in Titanic dataset.

The dataset is loaded using:

    sns.load_dataset("titanic")

An offline CSV backup is immediately saved as:

    analytics/data/titanic.csv

## 3. Exploratory Data Analysis

### Missing Values

Missing-value counts and percentages are calculated for every column.

A threshold-based cleaning strategy is applied:
- Columns with more than 40% missing values are removed.
- Age is imputed using median values grouped by passenger class and sex.
- Fare is imputed using its median.
- Embarked is imputed using its mode.
- Remaining missing numerical and categorical values are handled using median and mode respectively.

### Distribution Analysis

Histograms and boxplots are generated for:
- Passenger Age
- Passenger Fare

### Outlier Detection

The Interquartile Range method is used.

    IQR = Q3 - Q1

    Lower Bound = Q1 - 1.5 × IQR

    Upper Bound = Q3 + 1.5 × IQR

Values outside these boundaries are identified as outliers.

### Fare Statistics

The following statistics are calculated:
- Mean
- Median
- Mode
- Skewness

Positive skewness indicates a longer right tail in the distribution.

### Survival Analysis

Survival rates are analyzed by:
- Sex
- Passenger class
- Combination of sex and passenger class

### Correlation Analysis

A six-variable correlation matrix is generated using:
- survived
- pclass
- age
- sibsp
- parch
- fare

A heatmap visualizes the relationships.

The two strongest off-diagonal correlations are identified.

### Standardization

Age and Fare are standardized using StandardScaler for an EDA diagnostic.

The modeling pipeline separately fits preprocessing only on training data to prevent data leakage.

## 4. Classification Models

The target variable is:

    survived

The dataset is divided into training and testing subsets using a stratified split.

The following models are implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

### Evaluation Metrics

Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

ROC curves are generated to compare classification performance.

A Decision Tree visualization is also generated.

## 5. Class Imbalance Handling

Three approaches are compared:

1. Baseline Logistic Regression
2. Class-weight balanced Random Forest
3. Logistic Regression with SMOTE

SMOTE is applied only to the training data through an imbalanced-learn pipeline.

The test dataset is not oversampled.

## 6. Hyperparameter Tuning

GridSearchCV is used to optimize the Random Forest classifier.

Parameters:
- n_estimators
- max_depth
- max_features

Three-fold cross-validation is used.

The Random Forest model uses:

    oob_score=True

The best model is evaluated on the held-out test set.

## 7. Regression Analysis

Random Forest Regression is used to predict passenger fare.

The regression model is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared
- Adjusted R-squared

A residual plot is generated.

Residual patterns are inspected for possible heteroscedasticity.

## 8. Model Persistence

The complete fitted classification pipeline is saved using Joblib.

It includes preprocessing and the trained model.

The saved pipeline is reloaded and tested using an unprocessed input row.

The fare regression pipeline is also saved.

## 9. Project Structure

- 01_eda.py: Exploratory data analysis.
- 02_modeling.py: Classification and regression.
- data/: Dataset and cleaned dataset.
- outputs/: Generated charts and analysis results.
- models/: Saved machine-learning pipelines.

## 10. Execution Order

Run the scripts in the following order:

    python 01_eda.py

    python 02_modeling.py

Install the project dependencies before execution.

## 11. Expected Outputs

EDA outputs:
- Missing-value report
- Age and Fare histograms
- Age and Fare boxplots
- Survival charts
- Correlation heatmap
- Standardized data

Modeling outputs:
- Classification metrics
- Confusion matrices
- ROC comparison
- Decision Tree diagram
- Class imbalance comparison
- GridSearchCV results
- Fare regression metrics
- Residual plot
- Saved model pipelines

All output files are generated during local execution.

## 12. Conclusion

The analysis explores passenger survival patterns and evaluates multiple classification approaches.

Regression is used to estimate passenger fares.

The model evaluation metrics provide a basis for selecting suitable predictive models while the use of training-only preprocessing helps prevent data leakage.
