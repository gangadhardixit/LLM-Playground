#!/usr/bin/env python
# coding: utf-8

import streamlit as st

st.set_page_config(
    page_title="Book Recommender",
    layout="centered",
    initial_sidebar_state="auto"
)

import pandas as pd
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# --- Load the Books Dataset ---
books = pd.read_csv("C:/Users/gdixit/Notebooks/Recomendation_with_langchain/books_cleaned.csv")

# --- Write Descriptions to a File (only once) ---
# Optional: comment this after first run to avoid overwriting
books["tagged_description"].to_csv(
    "C:/Users/gdixit/Notebooks/Recomendation_with_langchain/tagged_description.txt",
    sep="\n",
    index=False,
    header=False
)

# --- Cache and Load Vector DB ---
@st.cache_resource(show_spinner="Loading book vector database...")
def load_vector_db():
    raw_documents = TextLoader(
        "C:/Users/gdixit/Notebooks/Recomendation_with_langchain/tagged_description.txt",
        encoding='utf-8'
    ).load()

    text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
    documents = text_splitter.split_documents(raw_documents)

    embeddings = OpenAIEmbeddings(api_key='xyz1')  
    return Chroma.from_documents(documents, embedding=embeddings)

# --- Load DB ---
db_books = load_vector_db()

# --- Define Recommendation Function ---
def retrieve_recommendations(query: str, db_books, top_k: int = 3) -> pd.DataFrame:
    recs = db_books.similarity_search(query, top_k)
    books_list = []

    for rec in recs:
        try:
            isbn_str = rec.page_content.strip('"').split()[0]
            books_list.append(int(isbn_str))
        except (ValueError, IndexError) as e:
            st.error(f"Error parsing recommendation: {rec.page_content}. Error: {e}")
            continue

    return books[books["isbn13"].isin(books_list)]

# --- Streamlit UI ---

st.title("Book Recommendation Engine")
st.markdown("Enter a query to get book recommendations from our site")

# Input widgets
query_input = st.text_input(
    "What kind of books are you looking for?",
    placeholder="e.g., 'classic novels', 'sci-fi adventures', 'cooking books'",
    help="Describe the genre, theme, or author you're interested in."
)

top_k_input = st.number_input(
    "How many recommendations?",
    min_value=1,
    max_value=10,
    value=3,
    step=1,
    help="Choose the number of top recommendations to display."
)

# Recommendation button
if st.button("Get Recommendations"):
    if query_input:
        with st.spinner("Searching for books..."):
            try:
                recommended_books = retrieve_recommendations(query_input, db_books, top_k_input)

                if not recommended_books.empty:
                    st.subheader(f"Top {len(recommended_books)} Recommendations for '{query_input}':")
                    st.dataframe(recommended_books, use_container_width=True)
                else:
                    st.info("No recommendations found. Try a different query!")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a query to get recommendations.")

st.markdown(
    """
    ---
    *This is a demo app using LangChain, ChromaDB, and OpenAI Embeddings.*
    """
)
