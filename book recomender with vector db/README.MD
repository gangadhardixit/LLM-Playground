Book Recommender Engine

This is a simple, self-contained book recommendation engine built with Streamlit, LangChain, and ChromaDB. It allows you to get book suggestions based on a natural language query.
 Features

    Natural Language Search: Find books by describing a genre, theme, or author in plain English.

    Vector-based Recommendations: The app uses a vector database to find books with descriptions semantically similar to your query.

    Self-contained: This version uses a mock dataset and a local embedding model, so it runs out-of-the-box without needing external files or API keys.

    Intuitive UI: A clean and simple user interface built with Streamlit for a great user experience.

💻 How It Works

The application uses a Retrieval-Augmented Generation (RAG) pipeline to provide recommendations:

    Data Preparation: A small dataset of book titles, authors, and summaries is loaded into a pandas DataFrame. The summaries are tagged with their unique ISBN.

    Embedding: An open-source HuggingFace embedding model is used to convert each book's tagged summary into a numerical vector.

    Vector Store: These vectors are stored in a ChromaDB vector database.

    Querying: When you enter a query, the same embedding model converts your query into a vector.

    Similarity Search: ChromaDB performs a similarity search, finding the book vectors that are most "similar" to your query vector.

    Recommendation: The app retrieves the corresponding book information from the DataFrame using the ISBN from the tagged summary and displays the results.

▶️ How to Run the App
Prerequisites

Make sure you have Python installed on your system.
Installation

First, install the necessary libraries using pip:

pip install streamlit pandas langchain-community langchain-chroma

Running the App

    Save the provided Python code as a file named app.py.

    Open your terminal or command prompt.

    Navigate to the directory where you saved app.py.

    Run the following command:

streamlit run app.py

The application will automatically open in your web browser.
📚 Customization

To use your own dataset and an OpenAI embedding model, you can modify the load_vector_db function and the mock_books section.

    Replace the mock_books DataFrame with code to load your books_cleaned.csv file.

    Uncomment or add the OpenAIEmbeddings line and replace the HuggingFace model.

    Remember to include your API key in the OpenAIEmbeddings call.
