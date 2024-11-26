import streamlit as st
import pandas as pd
import pickle
from surprise import SVD
from surprise import Dataset
from surprise import Reader

# Load the pre-trained model
@st.cache_resource
def load_model():
    with open('recommendation_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Define the recommendation function
def get_recommendations(model, customer_id, user_item_triplets, n=5):
    # Get all unique product IDs and descriptions
    all_products = user_item_triplets[['StockCode', 'Description']].drop_duplicates()

    # Get products already purchased by the customer
    purchased_products = user_item_triplets[user_item_triplets['CustomerID'] == customer_id]['StockCode']

    # Filter products not yet purchased
    products_to_predict = all_products[~all_products['StockCode'].isin(purchased_products.values)]

    # Predict ratings for all products not purchased
    predictions = [
        (row['StockCode'], row['Description'], model.predict(customer_id, row['StockCode']).est)
        for _, row in products_to_predict.iterrows()
    ]

    # Sort predictions by estimated rating
    top_recommendations = sorted(predictions, key=lambda x: x[2], reverse=True)[:n]
    return top_recommendations

# App Title
st.title("Product Recommendation System")

# Input Section
st.sidebar.header("User Input")
customer_id = st.sidebar.text_input("Enter Customer ID:", value="12345")
num_recommendations = st.sidebar.slider("Number of Recommendations:", 1, 10, 5)

# Load necessary data
@st.cache_resource
def load_data():
    # Replace with the correct path to your .xlsx file
    data_path = 'OnlineRetail.xlsx'
    data = pd.read_excel(data_path)
    # Ensure column names match expectations
    data = data.rename(columns={"Customer ID": "CustomerID", "Stock Code": "StockCode", "Description": "Description"})
    return data

user_item_triplets = load_data()

# Load the model
model = load_model()

# Display Recommendations
if st.sidebar.button("Get Recommendations"):
    try:
        recommendations = get_recommendations(model, customer_id, user_item_triplets, n=num_recommendations)
        if recommendations:
            st.header(f"Top {num_recommendations} Recommendations for Customer {customer_id}")
            for idx, (product_id, description, rating) in enumerate(recommendations, 1):
                st.write(f"**{idx}. Product ID:** {product_id}")
                st.write(f"**Description:** {description}")
                st.write(f"**Estimated Rating:** {rating:.2f}")
        else:
            st.warning("No recommendations available for this customer.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
