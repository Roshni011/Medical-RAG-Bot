# import streamlit as st
# import pandas as pd
# import os
# import cohere

# # --- Page Configuration ---
# st.set_page_config(
#     page_title="Medical RAG Chatbot",
#     page_icon="⚕️",
#     layout="centered"
# )
# 
# # --- RAG Core Logic ---

# @st.cache_data
# def load_faq_data(file_path):
#     """
#     Loads the FAQ data from the specified CSV file.
#     Caches the data to avoid reloading on every interaction.
#     """
#     try:
#         if not os.path.exists(file_path):
#             st.error(f"Error: The file '{file_path}' was not found.")
#             st.info("Please make sure the 'medical_faq.csv' file is in the same directory as this script.")
#             return None
#         return pd.read_csv(file_path)
#     except Exception as e:
#         st.error(f"An error occurred while loading the CSV file: {e}")
#         return None

# def calculate_similarity(query, question):
#     """
#     Calculates a similarity score between the user's query and a question from the FAQ.
#     This uses a simple word overlap (Jaccard similarity) method.
#     """
#     query_words = set(str(query).lower().split())
#     question_words = set(str(question).lower().split())

#     if not query_words or not question_words:
#         return 0.0

#     intersection = query_words.intersection(question_words)
#     union = query_words.union(question_words)
    
#     return len(intersection) / len(union)

# def retrieve_and_generate(query, df, cohere_client):
#     """
#     Retrieves the most relevant context and uses Cohere to generate a final answer.
#     """
#     if df is None:
#         return "I'm sorry, my knowledge base is currently unavailable. Please check the data source."

#     # 1. Retrieval Step
#     best_match = {
#         "score": 0.0,
#         "context": "No relevant context found."
#     }

#     for index, row in df.iterrows():
#         question = row['Question']
#         score = calculate_similarity(query, question)
        
#         if score > best_match["score"] and score > 0.1:
#             best_match["score"] = score
#             best_match["context"] = row['Answer']
            
#     retrieved_context = best_match["context"]

#     # 2. Generation Step using Cohere
#     try:
#         # Construct a prompt for the language model
#         prompt = f"""
#         You are a helpful medical chatbot. Your tone should be informative and reassuring.
#         Use the following piece of context to answer the user's question. 
#         If the provided context is not sufficient or irrelevant to answer the question, 
#         politely state that you cannot find the information in your knowledge base. 
#         Do not make up information.

#         Context:
#         "{retrieved_context}"

#         Question:
#         "{query}"

#         Answer:
#         """
        
#         response = cohere_client.chat(
#             message=prompt,
#             model='command-r',
#             temperature=0.3
#         )
#         return response.text

#     except Exception as e:
#         st.error(f"An error occurred with the Cohere API: {e}")
#         return "Sorry, I'm having trouble connecting to my brain right now. Please try again later."


# # --- Streamlit UI ---

# st.title("⚕️ Medical RAG Chatbot")
# st.caption("Powered by Cohere and your local FAQ data.")

# with st.sidebar:
#     st.header("Configuration")
#     cohere_api_key = st.text_input("Cohere API Key", type="password", help="Get your API key from [Cohere's dashboard](https://dashboard.cohere.com/).")
#     st.markdown("---")
#     st.markdown("This chatbot uses a Retrieval-Augmented Generation (RAG) model.")
#     st.markdown("1. **Retrieval**: It finds the most relevant answer from your `medical_faq.csv` file.")
#     st.markdown("2. **Generation**: It uses the powerful Cohere model to give a conversational answer based on that information.")

# # Load the data
# faq_df = load_faq_data('medical_faq.csv')

# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = [
#         {"role": "assistant", "content": "Hello! Please enter your Cohere API key in the sidebar to begin. How can I help you today?"}
#     ]

# # Display prior chat messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Get user input
# if prompt := st.chat_input("Ask a question..."):
#     # Add user message to history and display it
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # Generate and display bot response
#     with st.chat_message("assistant"):
#         if not cohere_api_key:
#             st.warning("Please enter your Cohere API key in the sidebar to enable the chatbot.")
#             st.stop()
        
#         try:
#             co_client = cohere.Client(cohere_api_key)
#             with st.spinner("Thinking..."):
#                 response = retrieve_and_generate(prompt, faq_df, co_client)
#             st.markdown(response)
#         except Exception as e:
#             st.error(f"An error occurred. Please check your API key. Details: {e}")
#             st.stop()


#     # Add bot response to history
#     st.session_state.messages.append({"role": "assistant", "content": response})





import streamlit as st
import pandas as pd
import os
import cohere
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#  Page Configuration 
st.set_page_config(
    page_title="Medical RAG Chatbot",
    page_icon="⚕️",
    layout="centered"
)

# RAG 

@st.cache_data
def load_faq_data(file_path):
    """
    Loads the FAQ data from the specified CSV file.
    Caches the data to avoid reloading on every interaction.
    """
    try:
        if not os.path.exists(file_path):
            st.error(f"Error: The file '{file_path}' was not found.")
            st.info("Please make sure the 'medical_faq.csv' file is in the same directory as this script.")
            return None
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"An error occurred while loading the CSV file: {e}")
        return None
# Due to limited credits (for free-tier users), I have used Cohere API key for triall here..
@st.cache_data(show_spinner=False)
def get_or_create_embeddings(_df, _cohere_client):
    """
    Generates embeddings for the questions in the dataframe if they don't already exist,
    otherwise loads them from a local file. Uses Streamlit's cache for efficiency.
    """
    embeddings_file = 'faq_embeddings.npy'

    if os.path.exists(embeddings_file):
        st.info("Loading existing embeddings from file.")
        return np.load(embeddings_file)
    else:
        st.info("No existing embeddings found. Generating new ones...")
        questions = _df['Question'].tolist()
        #  embedding model and input type for retrieval
        response = _cohere_client.embed(
            texts=questions,
            model='embed-english-v3.0',
            input_type='search_document'
        )
        embeddings = np.array(response.embeddings)
        np.save(embeddings_file, embeddings)
        st.success("Embeddings generated and saved successfully!")
        return embeddings

def retrieve_and_generate(query, df, cohere_client, doc_embeddings):
    """
    Retrieves the most relevant context using vector search and uses Cohere
    to generate a final answer.
    """
    if df is None:
        return "I'm sorry, my knowledge base is currently unavailable. Please check the data source."

    # Embeddings + Vector Search
    retrieved_context = "No relevant context found."
    
    # Embed the user query
    query_embedding_response = cohere_client.embed(
        texts=[query],
        model='embed-english-v3.0',
        input_type='search_query'
    )
    query_embedding = np.array(query_embedding_response.embeddings)

    # Calculate cosine similarity between query and all document embeddings
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    
    # Find the best match
    best_match_index = np.argmax(similarities)
    best_match_score = similarities[best_match_index]

    # Set a threshold for relevance
    if best_match_score > 0.3: # You can tune this threshold
        retrieved_context = df.iloc[best_match_index]['Answer']

    # 2. Generation Step using Cohere
    try:
        # Construct a prompt for the language model
        prompt = f"""
        You are a helpful medical chatbot. Your tone should be informative and reassuring.
        Use the following piece of context to answer the user's question. 
        If the provided context is not sufficient or irrelevant to answer the question, 
        politely state that you cannot find the information in your knowledge base. 
        Do not make up information.

        Context:
        "{retrieved_context}"

        Question:
        "{query}"

        Answer:
        """
        
        response = cohere_client.chat(
            message=prompt,
            model='command-r',
            temperature=0.3
        )
        return response.text

    except Exception as e:
        st.error(f"An error occurred with the Cohere API: {e}")
        return "Sorry, I'm having trouble connecting to my brain right now. Please try again later."


# --- Streamlit UI ---

st.title("⚕️ Medical RAG Chatbot")
st.caption("Powered by Cohere Embeddings and Command R.")

with st.sidebar:
    st.header("Configuration")
    cohere_api_key = st.text_input("Cohere API Key", type="password", help="Get your API key from [Cohere's dashboard](https://dashboard.cohere.com/).")
    st.markdown("---")
    st.markdown("This chatbot uses a Retrieval-Augmented Generation (RAG) model:")
    st.markdown("1. **Retrieval**: It embeds your query and uses **vector search** to find the most semantically similar answer from `medical_faq.csv`.")
    st.markdown("2. **Generation**: It uses the powerful Cohere model to give a conversational answer based on that information.")

# Load the data
faq_df = load_faq_data('medical_faq.csv')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Please enter your Cohere API key in the sidebar to begin. How can I help you today?"}
    ]

# Initialize embeddings in session state
if "doc_embeddings" not in st.session_state:
    st.session_state.doc_embeddings = None

# Create embeddings if API key is provided
if cohere_api_key and faq_df is not None and st.session_state.doc_embeddings is None:
    try:
        co_client_for_embed = cohere.Client(cohere_api_key)
        with st.spinner("Preparing knowledge base... (this is a one-time setup)"):
            st.session_state.doc_embeddings = get_or_create_embeddings(faq_df, co_client_for_embed)
    except Exception as e:
        st.error(f"Failed to initialize or create embeddings. Please check your API key. Error: {e}")
        st.stop()


# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display bot response
    with st.chat_message("assistant"):
        if not cohere_api_key:
            st.warning("Please enter your Cohere API key in the sidebar to enable the chatbot.")
            st.stop()
        
        if faq_df is None or st.session_state.doc_embeddings is None:
            st.error("The knowledge base is not ready. Please check the CSV file and your API key.")
            st.stop()

        try:
            co_client = cohere.Client(cohere_api_key)
            with st.spinner("Thinking..."):
                response = retrieve_and_generate(prompt, faq_df, co_client, st.session_state.doc_embeddings)
            st.markdown(response)
            # Add bot response to history
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"An error occurred. Please check your API key. Details: {e}")
            st.stop()
