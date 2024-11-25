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
The product recommendation system was built using collaborative filtering techniques with the Surprise library. A user-item matrix was created to predict the top products for each customer.

#### Steps:
1. Created a user-item matrix where rows represent customers, columns represent products, and values represent quantities purchased.
2. Used collaborative filtering (Singular Value Decomposition - SVD) to predict product ratings for each customer.
3. Generated personalized product recommendations based on predicted ratings.

#### Performance Metrics:
- **Root Mean Squared Error (RMSE):** 80984.14  
  Measures the average magnitude of prediction error (lower is better).
- **Mean Absolute Error (MAE):** 80983.79  
  Indicates the average absolute difference between predicted and actual values.

#### Example Output:
Top 5 product recommendations for a sample customer:

```plaintext
1. Product A (Predicted Rating: 4.5)
2. Product B (Predicted Rating: 4.3)
3. Product C (Predicted Rating: 4.2)
4. Product D (Predicted Rating: 4.1)
5. Product E (Predicted Rating: 4.0)

```
---

## Model Evaluation
### Clustering Quality:
- **Silhouette Score:** 0.562  
  - Interpretation: This score indicates moderate clustering quality, suggesting that customer segments are reasonably well-separated but with some overlap. Further feature engineering or alternative clustering methods could improve separation.

### Recommendation System Metrics:
- **Root Mean Squared Error (RMSE):** 80984.14  
  - Interpretation: A high RMSE indicates that the model's predictions deviate significantly from actual values. Optimization or hybrid approaches may reduce error.
- **Mean Absolute Error (MAE):** 80983.79  
  - Interpretation: The MAE reflects the average magnitude of errors in predictions. While informative, the high value highlights the need for improvement.

---

## Assumptions and Limitations
### Assumptions:
1. **Data Quality:** The dataset is assumed to be accurate, complete, and representative of customer purchasing behavior during the given period (December 2010 to December 2011).
2. **RFM Metrics:** Equal weights were assigned to Recency, Frequency, and Monetary metrics for segmentation, assuming their equal importance to customer behavior.
3. **Collaborative Filtering:** The recommendation system assumes that customers with similar purchase histories have similar preferences.

### Limitations:
1. **Sparse Data:** The collaborative filtering approach struggles with sparsity in the user-item matrix, where many products have few or no interactions.
2. **Exclusion of External Factors:** The analysis does not consider external factors like promotions, seasonality, or customer demographics, which may influence purchasing behavior.
3. **High RMSE and MAE:** The high error metrics suggest limited accuracy in the recommendation system, likely due to sparsity and lack of implicit feedback.
4. **Moderate Clustering Quality:** The silhouette score (0.562) indicates some overlap between clusters, suggesting that customer behaviors might not be entirely distinct.

---

## Installation and Usage
### Environment Setup:
Install the required dependencies using the following command:
```bash
pip install -r requirements.txt
```
---

## Credits and License
### Acknowledgments:
This project was made possible by the following resources and tools:
- **Dataset:**  
  The dataset used in this project was sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail).  
- **Tools and Libraries:**  
  - [Pandas](https://pandas.pydata.org/) - For data manipulation and analysis.  
  - [NumPy](https://numpy.org/) - For numerical computations.  
  - [Matplotlib](https://matplotlib.org/) - For creating visualizations.  
  - [Seaborn](https://seaborn.pydata.org/) - For enhanced data visualization.  
  - [scikit-learn](https://scikit-learn.org/) - For machine learning and clustering.  
  - [Surprise](http://surpriselib.com/) - For building the collaborative filtering recommendation system.

### License:
This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute this software, provided proper attribution is given to the original author. See the `LICENSE` file in this repository for more details.

---

**Last Updated:** November 25, 2024

