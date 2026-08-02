# --- Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_rel


def load_data(path):
    df = pd.read_csv(path, sep=';')
    return df

def eda(df):
    df.head()
    df.info()
    df.describe().drop(columns="id")
    df.isnull().sum()
    df["age_years"] = (df["age"] / 365).astype(int)
    plt.figure(figsize=(6,4))
    sns.histplot(df["age_years"], kde=True)
    plt.title("Age Distribution")
    plt.show()
    plt.figure(figsize=(6,4))
    sns.histplot(df["weight"], kde=True)
    plt.title("Weight Distribution")
    plt.show()
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df["ap_hi"])
    plt.title("Systolic Blood Pressure (ap_hi)")
    plt.show()
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df["ap_lo"])
    plt.title("Diastolic Blood Pressure (ap_lo)")
    plt.show()
    categorical_cols = ["gender", "cholesterol", "gluc", "smoke", "alco", "active", "cardio"]
    for col in categorical_cols:
        plt.figure(figsize=(5,4))
        sns.countplot(x=df[col])
        plt.title(f"{col} Distribution")
        plt.show()
    corr_with_target = df.corr()["cardio"].sort_values(ascending=False)
    plt.figure(figsize=(7,5))
    corr_with_target.drop("cardio").plot(kind="bar")
    plt.title("Correlation of Features with Target (cardio)")
    plt.show()
    df["age_bin"] = pd.cut(df["age_years"], bins=range(20, 80, 5))
    plt.figure(figsize=(12,5))
    sns.barplot(x=df["age_bin"], y=df["cardio"])
    plt.xticks(rotation=45)
    plt.title("Cardiovascular Disease Prevalence by Age Group")
    plt.show()
    def bp_category(row):
        if row["ap_hi"] < 120:
            return "normal"
        elif 120 <= row["ap_hi"] < 140:
            return "stage1"
        else:
            return "stage2"
    df["bp_category"] = df.apply(bp_category, axis=1)
    plt.figure(figsize=(6,4))
    sns.countplot(x=df["bp_category"])
    plt.title("Blood Pressure Categories")
    plt.show()
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df["weight"], y=df["ap_hi"], hue=df["cardio"])
    plt.title("Weight vs Systolic Blood Pressure")
    plt.show()
    fig = px.scatter(df, x="age_years", y="ap_hi", color="cardio", title="Interactive: Age vs Blood Pressure")
    fig.show()
    return df

def preprocess_data(df):
    df.drop(columns=["id", "age", "age_bin", "bp_category"], inplace = True)
    df.duplicated().sum()
    df.drop_duplicates(inplace= True)
    condition_keep = (
        (df['ap_hi'] <= 250) &
        (df['ap_lo'] <= 200) &
        (df['ap_hi'] >= 50)  &
        (df['ap_lo'] >= 30)  &
        (df['ap_hi'] > df['ap_lo'])
    )
    df_cleaned = df[condition_keep]
    numeric_cols = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo']
    for col in numeric_cols:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.85 * IQR
        upper_bound = Q3 + 1.85 * IQR
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
    df_cleaned.reset_index(drop=True, inplace=True)
    return df_cleaned

def split_data(df_cleaned):
    df_cleaned = df_cleaned.sort_values("age_years")
    train_end = int(len(df_cleaned) * 0.8)
    train = df_cleaned.iloc[:train_end].copy()
    test  = df_cleaned.iloc[train_end:].copy()
    X_train = train.drop(columns=["cardio"])
    y_train = train["cardio"]
    X_test  = test.drop(columns=["cardio"])
    y_test  = test["cardio"]
    return X_train, y_train, X_test, y_test

def resample_and_scale(X_train, y_train, X_test):
    sm = SMOTE()
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    numeric_robust = ["height","weight","ap_hi","ap_lo"]
    numeric_standard = ["age_years"]
    robust = RobustScaler()
    standard = StandardScaler()
    X_train_res[numeric_robust] = robust.fit_transform(X_train_res[numeric_robust])
    X_test[numeric_robust]      = robust.transform(X_test[numeric_robust])
    X_train_res[numeric_standard] = standard.fit_transform(X_train_res[numeric_standard])
    X_test[numeric_standard]      = standard.transform(X_test[numeric_standard])
    return X_train_res, y_train_res, X_test

def clustering(X_train_res, y_train):
    # KMeans
    kmeans = KMeans(n_clusters=6, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_train_res)
    df_kmeans = X_train_res.copy()
    df_kmeans["cluster"] = kmeans_labels
    df_kmeans["cardio"] = y_train.values
    cluster_profile = df_kmeans.groupby("cluster").agg(
        patients_count = ("cardio", "count"),
        cardio_rate    = ("cardio", "mean"),
        age_mean       = ("age_years", "mean"),
        ap_hi_mean     = ("ap_hi", "mean"),
        ap_lo_mean     = ("ap_lo", "mean"),
        chol_mean      = ("cholesterol", "mean"),
        gluc_mean      = ("gluc", "mean")
    )
    # DBSCAN
    dbscan = DBSCAN(eps=0.8, min_samples=20)
    dbscan_labels = dbscan.fit_predict(X_train_res)
    outliers_mask = dbscan_labels == -1
    outliers_count = outliers_mask.sum()
    total_count = len(dbscan_labels)
    outliers_ratio = outliers_count / total_count
    # Agglomerative + t-SNE
    X_hier = X_train_res.sample(n=3000, random_state=42)
    hc = AgglomerativeClustering(n_clusters=6, linkage="ward")
    hc_labels = hc.fit_predict(X_hier)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_hier_tsne = tsne.fit_transform(X_hier)
    # Visualization code can be added here if needed
    return cluster_profile, outliers_count, outliers_ratio

def deep_learning(X_train, y_train, X_test, y_test):
    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)
    y_pred_dnn = (model.predict(X_test) > 0.5).astype(int)
    return y_pred_dnn, model

def ensemble_evaluation(X_train, y_train, X_test, y_test, y_pred_dnn):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_pred_gb = gb.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    acc_gb = accuracy_score(y_test, y_pred_gb)
    acc_dnn = accuracy_score(y_test, y_pred_dnn)
    t_rf, p_rf = ttest_rel(y_pred_dnn.flatten(), y_pred_rf)
    t_gb, p_gb = ttest_rel(y_pred_dnn.flatten(), y_pred_gb)
    mean_diff_rf = np.mean(y_pred_dnn - y_pred_rf)
    mean_diff_gb = np.mean(y_pred_dnn - y_pred_gb)
    print(f"Random Forest Accuracy: {acc_rf:.4f}")
    print(f"Gradient Boosting Accuracy: {acc_gb:.4f}")
    print(f"DNN Accuracy: {acc_dnn:.4f}")
    print(f"Paired t-test DNN vs RF: t={t_rf:.3f}, p={p_rf:.3e}")
    print(f"Paired t-test DNN vs GB: t={t_gb:.3f}, p={p_gb:.3e}")
    print(f"Mean difference DNN-RF: {mean_diff_rf:.4f}")
    print(f"Mean difference DNN-GB: {mean_diff_gb:.4f}")

def main():
    path = r"D:\machine learning\Route\Project\cardio_train.csv"
    df = load_data(path)
    df = eda(df)
    df_cleaned = preprocess_data(df)
    X_train, y_train, X_test, y_test = split_data(df_cleaned)
    X_train_res, y_train_res, X_test_scaled = resample_and_scale(X_train, y_train, X_test)
    cluster_profile, outliers_count, outliers_ratio = clustering(X_train_res, y_train_res)
    print("Cluster profile:\n", cluster_profile)
    print(f"Outliers: {outliers_count}, Ratio: {outliers_ratio:.4f}")
    y_pred_dnn, dnn_model = deep_learning(X_train_res, y_train_res, X_test_scaled, y_test)
    ensemble_evaluation(X_train_res, y_train_res, X_test_scaled, y_test, y_pred_dnn)

if __name__ == "__main__":
    main()
