import pandas as pd
import faiss
import numpy as np
import cohere

class RAGPipeline:
    def __init__(self, cohere_api_key, index_path="faiss_index.bin"):
        self.co = cohere.Client(cohere_api_key)
        self.index_path = index_path
        self.index = None
        self.texts = []

    def load_data(self, filepath):
        df = pd.read_csv(filepath)
        if "question" not in df.columns or "answer" not in df.columns:
            raise ValueError("CSV must contain 'question' and 'answer' columns")
        self.texts = df["question"].tolist() + df["answer"].tolist()
        return df

    def create_embeddings(self, texts):
        # Cohere requires input_type
        response = self.co.embed(
            texts=texts,
            model="embed-english-v3.0",
            input_type="search_document"  # documents for FAISS
        )
        return np.array(response.embeddings, dtype="float32")

    def build_faiss_index(self):
        if not self.texts:
            raise ValueError("No texts loaded. Run load_data() first.")
        embeddings = self.create_embeddings(self.texts)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        faiss.write_index(self.index, self.index_path)
        return self.index

    def load_faiss_index(self):
        self.index = faiss.read_index(self.index_path)
        return self.index

    def retrieve(self, query, top_k=3):
        if self.index is None:
            raise ValueError("Index not built. Please load/build it first.")

        query_embedding = self.co.embed(
            texts=[query],
            model="embed-english-v3.0",
            input_type="search_query"  # queries for retrieval
        ).embeddings[0]

        query_vector = np.array([query_embedding], dtype="float32")
        distances, indices = self.index.search(query_vector, top_k)
        results = [self.texts[i] for i in indices[0]]
        return results
