import os
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "titanic.csv")

def load_dataset():

    print("\nSTEP 1: LOADING TITANIC DATASET")

    try:
        
        df = sns.load_dataset("titanic")

        df.to_csv(CSV_PATH, index=False)

        print("Dataset loaded successfully using Seaborn.")
        print("Offline CSV backup saved:", CSV_PATH)

    except Exception as error:

        print("Online dataset loading failed:", error)

        if os.path.exists(CSV_PATH):
            print("Loading previously saved offline CSV.")
            df = pd.read_csv(CSV_PATH)
        else:
            raise RuntimeError(
                "Titanic dataset unavailable and no CSV backup exists."
            )

    print("\nDataset shape:", df.shape)
    print("\nFirst five rows:")
    print(df.head())

    print("\nDataset information:")
    df.info()

    return df

def analyze_missing_values(df):

    print("\nSTEP 2: MISSING VALUE ANALYSIS")

    missing_count = df.isnull().sum()

    missing_percentage = (
        df.isnull().mean() * 100
    ).round(2)

    missing_report = pd.DataFrame({
        "Missing_Count": missing_count,
        "Missing_Percentage": missing_percentage
    })

    missing_report = missing_report.sort_values(
        by="Missing_Percentage",
        ascending=False
    )

    print("\nMissing value report:")
    print(missing_report)

    missing_report.to_csv(
        os.path.join(OUTPUT_DIR, "missing_values.csv")
    )

    return missing_report

def clean_dataset(df, missing_report):

    print("\nSTEP 3: DATA CLEANING")

    df = df.copy()
  
    threshold = 40

    columns_to_drop = missing_report[
        missing_report["Missing_Percentage"] > threshold
    ].index.tolist()

    print("\nColumns dropped due to excessive missing values:")
    print(columns_to_drop)

    df.drop(
        columns=columns_to_drop,
        inplace=True,
        errors="ignore"
    )
    if "age" in df.columns:

        df["age"] = df.groupby(
            ["pclass", "sex"]
        )["age"].transform(
            lambda group: group.fillna(group.median())
        )

        df["age"] = df["age"].fillna(df["age"].median())

    # Fill Fare using median.
    if "fare" in df.columns:
        df["fare"] = df["fare"].fillna(df["fare"].median())

    # Fill Embarked using mode.
    if "embarked" in df.columns:
        df["embarked"] = df["embarked"].fillna(
            df["embarked"].mode()[0]
        )

    # Fill remaining numerical missing values with median.
    for column in df.select_dtypes(include=np.number).columns:
        df[column] = df[column].fillna(df[column].median())

    # Fill remaining categorical missing values with mode.
    for column in df.select_dtypes(include="object").columns:
        if df[column].isnull().any():
            mode_values = df[column].mode()

            if not mode_values.empty:
                df[column] = df[column].fillna(mode_values[0])

    print("\nMissing values after cleaning:")
    print(df.isnull().sum())

    cleaned_path = os.path.join(
        OUTPUT_DIR,
        "titanic_cleaned.csv"
    )

    df.to_csv(cleaned_path, index=False)

    print("\nCleaned dataset saved:", cleaned_path)

    return df


# -------------------------------------------------
# 5. HISTOGRAMS AND BOXPLOTS
# -------------------------------------------------

def plot_distributions(df):

    print("\nSTEP 4: DISTRIBUTION ANALYSIS")

    # Age histogram
    plt.figure(figsize=(8, 5))

    sns.histplot(
        df["age"],
        bins=30,
        kde=True
    )

    plt.title("Distribution of Passenger Age")
    plt.xlabel("Age")
    plt.ylabel("Number of Passengers")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "age_histogram.png")
    )

    plt.close()

    # Fare histogram
    plt.figure(figsize=(8, 5))

    sns.histplot(
        df["fare"],
        bins=30,
        kde=True
    )

    plt.title("Distribution of Passenger Fare")
    plt.xlabel("Fare")
    plt.ylabel("Number of Passengers")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "fare_histogram.png")
    )

    plt.close()

    # Age boxplot
    plt.figure(figsize=(8, 4))

    sns.boxplot(x=df["age"])

    plt.title("Boxplot of Passenger Age")
    plt.xlabel("Age")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "age_boxplot.png")
    )

    plt.close()

    # Fare boxplot
    plt.figure(figsize=(8, 4))

    sns.boxplot(x=df["fare"])

    plt.title("Boxplot of Passenger Fare")
    plt.xlabel("Fare")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "fare_boxplot.png")
    )

    plt.close()

    print("Age and Fare histograms and boxplots saved.")


# -------------------------------------------------
# 6. IQR OUTLIER DETECTION
# -------------------------------------------------

def detect_outliers(df):

    print("\nSTEP 5: OUTLIER DETECTION USING IQR")

    for column in ["age", "fare"]:

        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)

        iqr = q3 - q1

        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr

        outliers = df[
            (df[column] < lower_limit) |
            (df[column] > upper_limit)
        ]

        print(f"\nColumn: {column}")
        print("Q1:", round(q1, 2))
        print("Q3:", round(q3, 2))
        print("IQR:", round(iqr, 2))
        print("Lower limit:", round(lower_limit, 2))
        print("Upper limit:", round(upper_limit, 2))
        print("Number of outliers:", len(outliers))


# -------------------------------------------------
# 7. FARE STATISTICAL ANALYSIS
# -------------------------------------------------

def fare_statistics(df):

    print("\nSTEP 6: FARE STATISTICS")

    fare_mean = df["fare"].mean()
    fare_median = df["fare"].median()
    fare_mode = df["fare"].mode()[0]
    fare_skewness = df["fare"].skew()

    print("Fare Mean:", round(fare_mean, 3))
    print("Fare Median:", round(fare_median, 3))
    print("Fare Mode:", round(fare_mode, 3))
    print("Fare Skewness:", round(fare_skewness, 3))

    if fare_skewness > 0:
        print("Interpretation: Fare distribution is positively skewed.")

    elif fare_skewness < 0:
        print("Interpretation: Fare distribution is negatively skewed.")

    else:
        print("Interpretation: Fare distribution is approximately symmetric.")


# -------------------------------------------------
# 8. SURVIVAL ANALYSIS
# -------------------------------------------------

def survival_analysis(df):

    print("\nSTEP 7: SURVIVAL ANALYSIS")

    # Overall survival
    print("\nOverall survival counts:")
    print(df["survived"].value_counts())

    print("\nOverall survival percentages:")
    print(
        (df["survived"].value_counts(normalize=True) * 100).round(2)
    )

    # Survival by sex
    sex_survival = df.groupby("sex")["survived"].mean() * 100

    print("\nSurvival percentage by sex:")
    print(sex_survival.round(2))

    # Survival by passenger class
    class_survival = df.groupby("pclass")["survived"].mean() * 100

    print("\nSurvival percentage by passenger class:")
    print(class_survival.round(2))

    # Survival by sex and passenger class
    combined_survival = df.pivot_table(
        values="survived",
        index="sex",
        columns="pclass",
        aggfunc="mean"
    ) * 100

    print("\nSurvival percentage by sex and passenger class:")
    print(combined_survival.round(2))

    # Chart 1: Survival by sex
    plt.figure(figsize=(7, 5))

    sns.barplot(
        data=df,
        x="sex",
        y="survived",
        errorbar=None
    )

    plt.title("Survival Rate by Sex")
    plt.xlabel("Sex")
    plt.ylabel("Survival Rate")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "survival_by_sex.png")
    )

    plt.close()

    # Chart 2: Survival by passenger class
    plt.figure(figsize=(7, 5))

    sns.barplot(
        data=df,
        x="pclass",
        y="survived",
        errorbar=None
    )

    plt.title("Survival Rate by Passenger Class")
    plt.xlabel("Passenger Class")
    plt.ylabel("Survival Rate")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "survival_by_class.png")
    )

    plt.close()

    # Chart 3: Survival by sex and passenger class
    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=df,
        x="pclass",
        y="survived",
        hue="sex",
        errorbar=None
    )

    plt.title("Survival Rate by Sex and Passenger Class")
    plt.xlabel("Passenger Class")
    plt.ylabel("Survival Rate")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "survival_by_sex_class.png")
    )

    plt.close()

    # Save survival tables.
    sex_survival.to_csv(
        os.path.join(OUTPUT_DIR, "survival_by_sex.csv")
    )

    class_survival.to_csv(
        os.path.join(OUTPUT_DIR, "survival_by_class.csv")
    )

    combined_survival.to_csv(
        os.path.join(OUTPUT_DIR, "survival_by_sex_class.csv")
    )

    print("\nChart interpretations:")
    print("1. Survival rates differ between male and female passengers.")
    print("2. Passenger class is associated with survival probability.")
    print("3. Survival rates vary across both sex and passenger class.")


# -------------------------------------------------
# 9. CORRELATION ANALYSIS
# -------------------------------------------------

def correlation_analysis(df):

    print("\nSTEP 8: CORRELATION ANALYSIS")

    required_columns = [
        "survived",
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare"
    ]

    # Exactly six columns are used.
    correlation_data = df[required_columns]

    correlation_matrix = correlation_data.corr()

    print("\nSix-column correlation matrix:")
    print(correlation_matrix.round(3))

    correlation_matrix.to_csv(
        os.path.join(OUTPUT_DIR, "correlation_matrix.csv")
    )

    # Heatmap
    plt.figure(figsize=(9, 7))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Titanic Dataset Correlation Heatmap")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    )

    plt.close()

    # Find the two strongest off-diagonal correlations.
    upper_triangle = correlation_matrix.where(
        np.triu(
            np.ones(correlation_matrix.shape),
            k=1
        ).astype(bool)
    )

    strongest_correlations = (
        upper_triangle.stack()
        .reindex(
            upper_triangle.stack().abs().sort_values(
                ascending=False
            ).index
        )
        .head(2)
    )

    print("\nTwo strongest off-diagonal correlations:")
    print(strongest_correlations.round(3))

    print(
        "\nInterpretation: Correlation indicates the strength "
        "and direction of a linear relationship. It does not "
        "prove causation."
    )


# -------------------------------------------------
# 10. STANDARDIZATION CHECK
# -------------------------------------------------

def standardization_check(df):

    print("\nSTEP 9: STANDARDIZATION CHECK")

    columns = ["age", "fare"]

    scaler = StandardScaler()

    standardized_data = scaler.fit_transform(
        df[columns]
    )

    standardized_df = pd.DataFrame(
        standardized_data,
        columns=["age_standardized", "fare_standardized"]
    )

    print("\nStandardized data preview:")
    print(standardized_df.head())

    print("\nStandardized means:")
    print(standardized_df.mean().round(6))

    print("\nStandardized standard deviations:")
    print(standardized_df.std(ddof=0).round(6))

    standardized_df.to_csv(
        os.path.join(OUTPUT_DIR, "standardized_age_fare.csv"),
        index=False
    )

    print(
        "\nInterpretation: Standardization transforms each "
        "feature to approximately zero mean and unit variance."
    )

    print(
        "This is an EDA diagnostic only. The modeling module "
        "will fit its scaler using training data only."
    )


# -------------------------------------------------
# 11. MAIN EXECUTION
# -------------------------------------------------

def main():

    print("=" * 65)
    print("ZEPTO DATA & AI PLATFORM - TITANIC EDA")
    print("=" * 65)

    # Load the dataset once.
    original_df = load_dataset()

    # Analyze missing values.
    missing_report = analyze_missing_values(original_df)

    # Clean the dataset.
    cleaned_df = clean_dataset(
        original_df,
        missing_report
    )

    # Distribution analysis.
    plot_distributions(cleaned_df)

    # Outlier analysis.
    detect_outliers(cleaned_df)

    # Statistical analysis.
    fare_statistics(cleaned_df)

    # Survival analysis.
    survival_analysis(cleaned_df)

    # Correlation analysis.
    correlation_analysis(cleaned_df)

    # Standardization check.
    standardization_check(cleaned_df)

    print("\n" + "=" * 65)
    print("EXPLORATORY DATA ANALYSIS COMPLETED")
    print("All results are saved in the analytics/outputs folder.")
    print("=" * 65)


if __name__ == "__main__":
    main()
