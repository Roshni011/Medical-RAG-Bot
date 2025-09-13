<!-- You'll need a **Cohere API Key** to run this application. -->

⚕️ Medical RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit and Cohere API. I have used cohere because of 100 Free API calls. OpenAI does not have any free API, rest have already hit the limit.

<!--  The free tier has been used up and it might throw error 429 because of rate limits. -->

**Retrieval-Augmented Generation (RAG):** Combines the power of a custom knowledge base with a large language model.
**Vector Search:** Uses Cohere's `embed-english-v3.0` model to embed both the FAQ questions and user queries for efficient semantic search.
**Cohere Command R:** Leverages the `command-r` model for generating coherent and context-aware responses.
**Streamlit Interface:** Provides a user-friendly and interactive chat interface.
**Cached Embeddings:** Saves and loads embeddings to a local `.npy` file to prevent redundant API calls and speed up subsequent runs.

### Knowledge Base

The chatbot's knowledge base is a CSV file named `medical_faq.csv`.
I have used a medical Dataset from Kaggle, named medical_faq.csv with "Question" and "Answer" columns(at least).

▶️ Run the Application
streamlit run app.py


Steps:
1. Run the Streamlit application from your terminal:
`streamlit run app.py`
2. A new browser tab will open automatically with the Streamlit app. If it doesn't, navigate to http://localhost:8501.

3. Enter your Cohere API Key in the sidebar. The application will then generate or load the embeddings for the FAQ data.

4. Once the "Preparing knowledge base..." message disappears, you can start asking questions in the chat box!

5. I have a loom video attached. 

### LINK : 