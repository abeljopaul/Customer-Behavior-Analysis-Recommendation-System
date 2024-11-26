import streamlit as st
import pandas as pd
import pickle
import implicit
from scipy.sparse import csr_matrix
from surprise import SVD

# Load data
@st.cache_data
def load_data():
    data_path = "OnlineRetail.xlsx"  # Ensure this file is uploaded to the same directory
    data = pd.read_excel(data_path)
    data = data.rename(
        columns={
            "Customer ID": "CustomerID",
            "Stock Code": "StockCode",
            "Description": "Description",
            "Quantity": "Quantity",
        }
    )
    # Drop rows with missing CustomerID or StockCode
    data = data.dropna(subset=["CustomerID", "StockCode"])
    return data

# Preprocess data into sparse matrix
@st.cache_resource
def preprocess_data(data):
    # Map CustomerID and StockCode to integer indices
    customer_mapping = {id: idx for idx, id in enumerate(data["CustomerID"].unique())}
    product_mapping = {id: idx for idx, id in enumerate(data["StockCode"].unique())}
    data["CustomerIndex"] = data["CustomerID"].map(customer_mapping)
    data["ProductIndex"] = data["StockCode"].map(product_mapping)

    # Create sparse matrix
    sparse_matrix = csr_matrix(
        (data["Quantity"], (data["CustomerIndex"], data["ProductIndex"]))
    )
    return sparse_matrix, customer_mapping, product_mapping

# Train recommendation model
@st.cache_resource
def train_model(sparse_matrix):
    model = implicit.als.AlternatingLeastSquares(factors=20, iterations=10, regularization=0.1)
    # Fit the model on the sparse matrix
    model.fit(sparse_matrix)
    return model

# Get recommendations
def get_recommendations(model, customer_id, customer_mapping, product_mapping, reverse_product_mapping, n=5):
    if customer_id not in customer_mapping:
        return []
    customer_index = customer_mapping[customer_id]
    recommendations = model.recommend(
        customer_index,
        sparse_matrix,
        N=n,
        filter_already_liked_items=True,
    )
    return [(reverse_product_mapping[idx], score) for idx, score in recommendations]

# App UI
st.title("Product Recommendation System with Implicit")

# Sidebar inputs
st.sidebar.header("User Input")
customer_id_input = st.sidebar.text_input("Enter Customer ID:")
num_recommendations = st.sidebar.slider("Number of Recommendations:", 1, 10, 5)

# Load and preprocess data
data = load_data()
sparse_matrix, customer_mapping, product_mapping = preprocess_data(data)
reverse_product_mapping = {v: k for k, v in product_mapping.items()}

# Train or load model
model = train_model(sparse_matrix)

# Generate recommendations
if st.sidebar.button("Get Recommendations"):
    try:
        recommendations = get_recommendations(
            model,
            customer_id=int(customer_id_input),
            customer_mapping=customer_mapping,
            product_mapping=product_mapping,
            reverse_product_mapping=reverse_product_mapping,
            n=num_recommendations,
        )
        if recommendations:
            st.header(f"Top {num_recommendations} Recommendations for Customer {customer_id_input}")
            for idx, (product_id, score) in enumerate(recommendations, 1):
                description = data[data["StockCode"] == product_id]["Description"].iloc[0]
                st.write(f"**{idx}. Product ID:** {product_id}")
                st.write(f"**Description:** {description}")
                st.write(f"**Score:** {score:.2f}")
        else:
            st.warning("No recommendations available for this customer.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
