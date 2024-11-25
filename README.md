# Customer-Behavior-Analysis-Recommendation-System
# E-commerce Customer Behavior Analysis and Recommendation System

## Project Overview
E-commerce platforms generate vast amounts of customer data, which hold valuable insights into purchasing behaviors, preferences, and trends. This project leverages real-world transaction data to analyze customer behavior, identify key customer segments, and recommend products.

### Objectives:
1. Analyze sales trends and identify top-performing products and regions.
2. Segment customers based on their purchasing behaviors using the RFM (Recency, Frequency, Monetary) framework.
3. Build a recommendation system using collaborative filtering techniques to improve the personalization of product recommendations.
4. Demonstrate practical data science techniques such as data cleaning, feature engineering, machine learning, and data visualization.

### Value of the Project:
The insights and systems developed in this project can help businesses:
- Improve marketing strategies by targeting specific customer segments.
- Boost sales through personalized product recommendations.
- Retain valuable customers and re-engage at-risk ones.

---

## Table of Contents
1. [Dataset and Data Handling](#dataset-and-data-handling)
2. [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Feature Engineering](#feature-engineering)
5. [Customer Segmentation](#customer-segmentation)
6. [Product Recommendation System](#product-recommendation-system)
7. [Model Evaluation](#model-evaluation)
8. [Assumptions and Limitations](#assumptions-and-limitations)
9. [Installation and Usage](#installation-and-usage)
10. [Credits and License](#credits-and-license)

---

## Dataset and Data Handling
- **Dataset Name:** Online Retail Dataset
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail)
- **Description:** 
  - Transaction records from December 2010 to December 2011.
  - Includes details on invoices, products, customers, and regions.

### Data Preparation Steps:
1. Missing values were handled (135,080 rows with missing `CustomerID` removed).
2. Duplicate entries (5,225 rows) were removed.
3. Features such as total purchase amount and purchase dates were derived.

---

## Data Cleaning and Preprocessing
- Missing values were handled using `dropna`.
- Duplicate rows were identified and removed.
- Data types were corrected (e.g., converting `InvoiceDate` to datetime).
- Features were normalized for clustering.

---

## Exploratory Data Analysis (EDA)
### Key Analyses:
1. Sales trends over time were visualized using time-series plots.
2. Top 10 products and countries were analyzed to gain business insights.
3. Customer behavior was segmented using RFM metrics (Recency, Frequency, Monetary).

### Visualizations:
- Sales trends.
- Top-selling products and customer segments.
- RFM distribution histograms.

---

## Feature Engineering
### New Features Created:
1. Total purchase amount (quantity × unit price).
2. RFM metrics for customer segmentation.
3. Time-based features (month and year extracted from `InvoiceDate`).

---

## Customer Segmentation
### Segmentation Approach:
- K-means clustering was applied to normalized RFM metrics.
- The Elbow Method determined the optimal number of clusters (k=4).

### Cluster Insights:
1. **Cluster 0:** Loyal Customers.
2. **Cluster 1:** At-Risk Customers.
3. **Cluster 2:** Big Spenders (VIPs).
4. **Cluster 3:** Lost Customers.

### Visualizations:
- PCA scatterplot for cluster visualization.
- Bar chart of customer segment distribution.

---

## Product Recommendation System
### Methodology:
- A collaborative filtering-based recommendation system was built using the Surprise library.
- A user-item matrix was created to predict top products for each customer.

### Performance Metrics:
- **RMSE:** 80984.14
- **MAE:** 80983.79

### Example Output:
Top 5 recommendations for a sample customer:
```plaintext
1. Product A
2. Product B
3. Product C
4. Product D
5. Product E
