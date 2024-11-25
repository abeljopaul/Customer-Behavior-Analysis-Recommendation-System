import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from mlxtend.frequent_patterns import apriori, association_rules

# Load your dataset
@st.cache_data
def load_data():
    file_path = 'OnlineRetail.xlsx'  # Ensure this file path is correct
    try:
        data = pd.read_excel(file_path)
        return data
    except FileNotFoundError:
        st.error(f"The file at {file_path} was not found.")
        return None

# Exploratory Data Analysis (EDA) Function
def show_eda(data):
    st.subheader("Exploratory Data Analysis")
    st.write("**Dataset Overview:**")
    st.dataframe(data.head())

    st.write(f"**Shape of the dataset:** {data.shape}")
    st.write("**Summary Statistics:**")
    st.write(data.describe())

    # Add visualizations
    st.subheader("Visualizations")
    if st.checkbox("Show Correlation Heatmap"):
        numeric_data = data.select_dtypes(include=[np.number])  # Select only numeric columns
        corr = numeric_data.corr()
        st.write("Correlation Matrix:")
        st.write(corr)
        fig, ax = plt.subplots()
        cax = ax.matshow(corr, cmap='coolwarm')
        fig.colorbar(cax)
        st.pyplot(fig)

# Customer-Based Recommendation System
def customer_based_recommendation(data, customer_id):
    customer_item_matrix = data.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', fill_value=0)

    if customer_id not in customer_item_matrix.index:
        st.error(f"Customer ID '{customer_id}' not found.")
        return []

    similarity_matrix = cosine_similarity(csr_matrix(customer_item_matrix))
    customer_index = customer_item_matrix.index.get_loc(customer_id)
    similarity_scores = similarity_matrix[customer_index]
    similar_customer_indices = similarity_scores.argsort()[::-1]

    recommended_items = []
    for idx in similar_customer_indices:
        similar_customer = customer_item_matrix.index[idx]
        items_bought = customer_item_matrix.loc[similar_customer]
        for product, quantity in items_bought.items():
            if quantity > 0 and product not in recommended_items:
                recommended_items.append(product)
        if len(recommended_items) >= 5:
            break

    return recommended_items[:5]

# Product-Based Recommendation System
def product_based_recommendation(data, product_id):
    transaction_data = data.groupby(['InvoiceNo', 'StockCode'])['Quantity'].sum().unstack().fillna(0)
    transaction_data = (transaction_data > 0).astype(int)  # Convert quantities to binary (purchased or not)
    frequent_itemsets = apriori(transaction_data, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    recommendations = rules[rules['antecedents'].apply(lambda x: product_id in x)]
    
    recommended_products = []
    for _, rule in recommendations.iterrows():
        recommended_products.extend(list(rule['consequents']))

    return list(set(recommended_products))[:5]

# Recommendation System Logic
def recommend_system(data):
    st.subheader("Recommendation System")
    rec_type = st.radio("Select Recommendation Type", ["Customer-Based", "Product-Based"])

    if rec_type == "Customer-Based":
        customer_id = st.text_input("Enter Customer ID:")
        if customer_id:
            recommended_items = customer_based_recommendation(data, customer_id)
            if recommended_items:
                st.write(f"Recommendations for Customer '{customer_id}':")
                st.write(recommended_items)

    elif rec_type == "Product-Based":
        product_id = st.text_input("Enter Product StockCode or Description:")
        if product_id:
            recommended_products = product_based_recommendation(data, product_id)
            st.write(f"Customers who bought '{product_id}' also bought:")
            st.write(recommended_products)

# Main Application
def main():
    st.title("Customer Behavior Analysis and Recommendation System")
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to:", ["EDA", "Recommendation System"])

    # Load data
    data = load_data()
    if data is not None:
        if choice == "EDA":
            show_eda(data)
        elif choice == "Recommendation System":
            recommend_system(data)
    else:
        st.error("There was an error loading the dataset.")

if __name__ == '__main__':
    main()
