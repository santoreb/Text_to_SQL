import streamlit as st

# Page configuration
st.set_page_config(
    page_title="SQL to Text",
    layout="wide"
)

# Sidebar for settings
st.sidebar.title("Settings")
st.sidebar.write("Choose which model you want to use to tranform SQL to Text.")

# Transformation options
transform_option = st.sidebar.selectbox(
    "Select Model",
    ["Finetuned on spider", "Clustering based finetuned on spider","Finetuned on Bird"]
)

# Main title and description
st.title("SQL to Text Tranformation")
st.markdown(
    """
    Welcome to the **SQL to Text Transformer App**.  
    Enter your text below, choose a model from the sidebar, and view your SQL query instantly.
    """
)

# Input area
user_input = st.text_area(
    "Enter your text:",
    height=200,
    placeholder="Type or paste your text here..."
)

# Process text
if st.button("Transform"):
    if user_input.strip():
        if transform_option == "Finetuned on spider":
            output_text = "Returned SQL from Finetuned on spider"
        elif transform_option == "Clustering based finetuned on spider":
            output_text = "Returned SQL from Clustering based finetuned on spider"
        elif transform_option == "Finetuned on Bird":
            output_text = "Returned SQL from Finetuned on Bird"

        st.subheader("Transformed Output")
        st.success(output_text)
    else:
        st.warning("Please enter some text before transforming.")
