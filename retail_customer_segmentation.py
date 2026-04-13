"""
Retail Customer Segmentation for Promotion Optimisation
========================================================
Segments customers from a retail marketing campaign dataset using
KMeans and Agglomerative Clustering. Includes data cleaning, outlier
removal, feature engineering, PCA-based dimensionality reduction,
and cluster profiling to support targeted promotional campaigns.

Dataset: marketing_campaign.csv  (tab-separated)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering

warnings.filterwarnings("ignore")


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load the tab-separated marketing campaign dataset."""
    df = pd.read_csv(filepath, sep="\t")
    print(f"Loaded {len(df)} records with {df.shape[1]} columns.")
    return df


# ── Cleaning ──────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove null rows, parse Dt_Customer to datetime, encode
    Education and Marital_Status to numeric, remove outliers,
    and drop redundant columns.
    """
    df = df.dropna()
    print(f"After dropping nulls: {len(df)} rows")

    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)

    df["Partner_Count"] = df["Marital_Status"].replace(
        {"Married": 2, "Together": 2, "Absurd": 1,
         "Widow": 1, "YOLO": 1, "Divorced": 1, "Single": 1, "Alone": 1}
    )
    df["Partner_Count"] = pd.to_numeric(df["Partner_Count"])

    df["Education"] = df["Education"].replace(
        {"Basic": 1, "2n Cycle": 1, "Graduation": 2, "Master": 3, "PhD": 3}
    )
    df["Education"] = pd.to_numeric(df["Education"])

    df = df[df["Year_Birth"] > 1920]
    df = df[df["Income"] < 600_000]
    print(f"After outlier removal: {len(df)} rows")

    drop_cols = ["Marital_Status", "Z_CostContact", "Z_Revenue", "ID"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


# ── Feature Engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, reference_year: int = 2023) -> pd.DataFrame:
    """Derive Age, Children, Family_Size, Is_Parent, Spend, Customer_For."""
    df["Age"] = reference_year - df["Year_Birth"]

    df["Children"] = df["Kidhome"] + df["Teenhome"]

    df["Family_Size"] = df["Partner_Count"] + df["Children"]

    df["Is_Parent"] = np.where(df["Children"] > 0, 1, 0).astype("int32")

    df["Spend"] = (
        df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"]
        + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
    )

    now = datetime.now()
    df["Customer_For"] = (now - df["Dt_Customer"]).apply(lambda x: x.total_seconds())

    drop_cols = ["Dt_Customer", "Year_Birth"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> tuple:
    """Standardise features and reduce to 3 principal components."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    print(f"Explained variance (3 PCs): {pca.explained_variance_ratio_.sum():.2%}")

    return X_scaled, X_pca, scaler, pca


# ── Optimal Cluster Selection ─────────────────────────────────────────────────

def plot_elbow(X: np.ndarray, max_k: int = 10):
    """WCSS elbow plot for KMeans cluster selection."""
    wcss = []
    k_range = range(2, max_k + 1)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        wcss.append(km.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(k_range, wcss, marker="o")
    plt.xlabel("k")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")
    plt.tight_layout()
    plt.savefig("elbow.png", dpi=150)
    plt.show()
    print("Saved: elbow.png")


# ── Clustering ────────────────────────────────────────────────────────────────

def fit_kmeans(X: np.ndarray, n_clusters: int, random_state: int = 42) -> np.ndarray:
    """Fit KMeans and return labels."""
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return km.fit_predict(X)


def fit_agglomerative(X: np.ndarray, n_clusters: int) -> np.ndarray:
    """Fit Agglomerative Clustering and return labels."""
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    return agg.fit_predict(X)


# ── Profiling and Visualisation ───────────────────────────────────────────────

def plot_clusters_3d(X_pca: np.ndarray, labels: np.ndarray, title: str = "Clusters"):
    """3D PCA scatter plot coloured by cluster label."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
        c=labels, cmap="tab10", alpha=0.6, s=15,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    fname = title.lower().replace(" ", "_") + ".png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")


def profile_clusters(df: pd.DataFrame, labels: np.ndarray,
                     cols: list = None) -> pd.DataFrame:
    """Return mean values per cluster for interpretable profiling."""
    df_p = df.copy()
    df_p["Cluster"] = labels
    if cols is None:
        cols = ["Age", "Income", "Spend", "Children", "Family_Size", "Customer_For"]
    cols = [c for c in cols if c in df_p.columns]
    profile = df_p.groupby("Cluster")[cols].mean().round(2)
    print("\nCluster Profiles:")
    print(profile.to_string())
    return profile


def plot_correlation_matrix(df: pd.DataFrame):
    """Heatmap of feature correlations."""
    plt.figure(figsize=(20, 20))
    sns.heatmap(df.corr(), annot=True, center=0)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig("correlation_matrix.png", dpi=100)
    plt.show()
    print("Saved: correlation_matrix.png")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH  = "marketing_campaign.csv"
    N_CLUSTERS = 4

    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = engineer_features(df)

    plot_correlation_matrix(df)

    X_scaled, X_pca, scaler, pca = preprocess(df)

    plot_elbow(X_pca)

    km_labels  = fit_kmeans(X_pca, n_clusters=N_CLUSTERS)
    agg_labels = fit_agglomerative(X_pca, n_clusters=N_CLUSTERS)

    plot_clusters_3d(X_pca, km_labels,  title="KMeans Clusters")
    plot_clusters_3d(X_pca, agg_labels, title="Agglomerative Clusters")

    profile_clusters(df, km_labels,  cols=["Age", "Income", "Spend", "Children", "Family_Size"])
    profile_clusters(df, agg_labels, cols=["Age", "Income", "Spend", "Children", "Family_Size"])
