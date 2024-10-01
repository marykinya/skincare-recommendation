# Skincare Product Recommendation System

## Overview

This project implements a **product recommendation system** that suggests similar skincare products based on their ingredients. It uses **cosine similarity** to compute the similarity between different products, making it easier for users to discover alternative skincare products.

The application is built using **Streamlit** for the user interface and **scikit-learn** for the machine learning components, such as calculating ingredient similarities.

## Features

- **Data Cleaning & Preprocessing**:
  - The dataset is cleaned and preprocessed by transforming the product ingredients from strings into lists, ensuring they are formatted correctly for the cosine similarity calculations.
  - Products include information such as name, URL, type, ingredient list, pricing, rating, and image URL.
  - Missing or inconsistent values in prices are handled, and columns are properly formatted.
  
- **Recommendation Engine**:
  - The core of the recommendation system is powered by **TfidfVectorizer** from `scikit-learn` to calculate the similarity between products based on their ingredient lists.
  - The user can select a product and the system will recommend similar items by analyzing the similarity scores.

- **Interactive Interface**:
  - The app allows users to interactively select a product type and a specific product from a dropdown menu, and receive personalized product recommendations.

## Data Cleaning

### Dataset

The dataset includes various skincare products with the following key columns:
- `product_name`: Name of the product.
- `product_url`: URL to the product page.
- `product_type`: Type/category of the product (e.g., Moisturiser, Cleanser).
- `ingredients`: List of ingredients in each product.
- `price`: Original price of the product.
- `updated_price`: Updated price, calculated to reflect price changes.
- `product_rating`: Average rating of the product.
- `image_url`: URL to the product's image.
- `price_change`: Difference between original and updated price.
- `brand`: The brand of the product.

### Data Preprocessing Steps

1. **Ingredients Cleaning**:
   - The ingredients column is converted from string representation of lists into actual Python lists using `pandas`.
   - This ensures that the **TfidfVectorizer** can process the data correctly when computing the cosine similarity.

2. **Price Adjustments**:
   - The prices are cleaned and compared. Any significant price changes are captured in a separate column called `price_change`.

3. **Handling Missing Data**:
   - Missing or inconsistent data is handled by checking for null values and removing rows with insufficient information.

4. **Brand Standardization**:
   - Brand names are standardized across the dataset for consistency in filtering and analysis.

## recommender.py

The `recommender.py` script is the core of the recommendation system and has several components:

### Key Components:

1. **Data Loading**:
   - The dataset is loaded using `pandas`. The ingredients are processed into lists for further analysis.

2. **Cosine Similarity**:
   - Using **TfidfVectorizer** from `scikit-learn`, the ingredients of each product are vectorized to compute the cosine similarity between products.
   - The cosine similarity allows us to compare products based on the overlap in their ingredients.

3. **Recommendation Function**:
   - A function calculates the cosine similarity of a selected product against all others in the dataset.
   - The user can define how many similar products they want to see (`top_n`), and a similarity threshold can be set to filter out less similar products.

4. **Streamlit Integration**:
   - The script integrates with **Streamlit** to create an interactive user interface, allowing users to select a product type and a specific product.
   - The app then displays similar products based on the cosine similarity score.
