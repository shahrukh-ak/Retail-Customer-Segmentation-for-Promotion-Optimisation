# Retail Customer Segmentation for Promotion Optimisation

Applies KMeans and Agglomerative Clustering to identify distinct customer segments in a retail marketing dataset. PCA reduces the feature space to three principal components for 3D visualisation. Segments are profiled to support targeted promotional strategy.

## Business Context

A retail chain's marketing team needed a data-driven way to group customers so promotions could be tailored rather than applied uniformly. This project delivers labelled segments with interpretable profiles covering age, income, spending, and household composition.

## Dataset

`marketing_campaign.csv` is a tab-separated file containing 2,240 customer records with demographic, purchase, and campaign response attributes.

## Methodology

**Cleaning:** Null rows dropped, `Dt_Customer` parsed to datetime, `Marital_Status` mapped to numeric `Partner_Count`, `Education` mapped to ordinal levels. Outliers in `Year_Birth` (before 1920) and `Income` (above 600,000) removed.

**Feature Engineering:**
- Age from Year_Birth
- Children (Kidhome + Teenhome)
- Family_Size (Partner_Count + Children)
- Is_Parent binary flag
- Total Spend across all product categories
- Customer_For: tenure in seconds from join date to today

**Preprocessing:** StandardScaler followed by PCA reduction to 3 components.

**Cluster Selection:** Elbow plot (WCSS vs k) on the PCA-reduced space.

**Algorithms:**
- KMeans (n_clusters=4, n_init=10)
- Agglomerative Clustering (n_clusters=4, Ward linkage)

**Visualisation:** 3D scatter plot in PCA space for each algorithm. Cluster profiles via group mean comparison.

## Project Structure

```
09_retail_customer_segmentation/
├── retail_customer_segmentation.py  # Full pipeline
├── requirements.txt
└── README.md
```

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install with:

```bash
pip install -r requirements.txt
```

## Usage

Place `marketing_campaign.csv` in the same directory and run:

```bash
python retail_customer_segmentation.py
```

Outputs: `correlation_matrix.png`, `elbow.png`, `kmeans_clusters.png`, `agglomerative_clusters.png`.
