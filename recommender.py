import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Project Documentation Section ---
def display_project_overview():

    st.header("Project Overview")
    st.markdown("""
    ### ✨ Skincare Product Recommendations: Discover Your Perfect Match ✨
    
    This project is designed to recommend similar skincare products based on their ingredients. By utilizing **cosine similarity**, the system calculates the degree of similarity between different products, allowing users to discover alternatives that align with their skincare preferences.

    With a comprehensive dataset containing:
    - **Product Names** 🏷️
    - **Product Types** 🛍️ (e.g., moisturizers, cleansers, and more)
    - **Ingredients** 🧪 (because we know every ingredient matters!)
    - **Ratings** ⭐ (so you can trust the quality)
    - **Prices** 💰 (for making budget-friendly choices)
    
    My system helps you discover similar products that might just be the next best thing for your skincare routine.

    """)
    st.markdown("---")

# --- Load and preprocess the skincare product dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv('skincare_products_2024_v2.csv')
    df['ingredients'] = df['ingredients'].apply(lambda x: x.strip("[]").replace("'", "").split(", "))
    return df

# --- Recommends similar products based on cosine similarity of ingredients within the same product type ---
def calculate_cosine_similarity(df, product_index, top_n=5, similarity_threshold=0.1):

    selected_product_type = df['product_type'].iloc[product_index]
    
    # Filter products by the same type and reset index
    df_same_type = df[df['product_type'] == selected_product_type].reset_index(drop=True)

    # Convert ingredients list back to strings to fit TfidfVectorizer
    df_same_type['ingredients_str'] = df_same_type['ingredients'].apply(lambda x: ' '.join(x))

    # Apply TfidfVectorizer
    vectorizer = TfidfVectorizer()
    ingredient_vectors = vectorizer.fit_transform(df_same_type['ingredients_str'])

    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(ingredient_vectors)
    target_similarities = cosine_similarities[product_index]

    # Sort by similarity score
    similar_indices = target_similarities.argsort()[::-1]
    
    # Recommend products based on similarity threshold
    recommended_products = []
    for idx in similar_indices:
        if len(recommended_products) >= top_n:
            break
        if idx != product_index and target_similarities[idx] >= similarity_threshold:
            recommended_products.append(df_same_type.iloc[idx])
    
    return pd.DataFrame(recommended_products)

# --- Main function to control the user interface and recommendation system ---
def main():

    df = load_data()
    
    # Display project documentation
    display_project_overview()

    # Create two columns for better layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # Dropdown to select a product type
        product_types = df['product_type'].unique().tolist()
        selected_product_type = st.selectbox('Select a product type:', product_types)

        # Filter product names based on selected product type and reset index
        filtered_products_df = df[df['product_type'] == selected_product_type].reset_index(drop=True)
        filtered_products = filtered_products_df['product_name'].tolist()

        # Dropdown to select a product
        selected_product = st.selectbox('Select a product:', filtered_products)

        # Get the index of the selected product
        product_index = filtered_products_df[filtered_products_df['product_name'] == selected_product].index[0]

    with col2:
        # Display the image of the selected product
        selected_product_image_url = filtered_products_df.loc[product_index, 'product_image_url']
        if selected_product_image_url:
            st.image(selected_product_image_url, width=150, caption=selected_product)

    # Input for number of recommendations and similarity threshold
    top_n = 5
    similarity_threshold = 0.1 

    # --- Recommend Products ---
    if st.button("Recommend"):
        recommended_products = calculate_cosine_similarity(filtered_products_df, product_index, top_n, similarity_threshold)

        if not recommended_products.empty:
            st.subheader("Recommended Products")
            # Display each recommended product with details and image
            for idx, row in recommended_products.iterrows():
                st.image(row['product_image_url'], width=150)
                st.markdown(f"**[{row['product_name']}]({row['product_url']})**")
                st.write(f"Brand: {row['brand']}")
                st.write(f"Rating: {row['product_rating']:.2f}")
                st.write(f"Price: ${row['updated_price']:,.2f}")
                st.write("---")
        else:
            st.warning("No similar products found. Try adjusting the similarity threshold or the number of recommendations.")

# Run the main function
if __name__ == "__main__":
    main()
