import os
from typing import Optional
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.utilities import SQLDatabase
from .db import get_engine


load_dotenv()

CHROMA_COLLECTION = "products"

def get_chroma_client():
    return chromadb.PersistentClient(path="./chroma_db")

def get_embedding_fn():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

def get_or_create_collection():
    client = get_chroma_client()
    ef = get_embedding_fn()
    return client.get_or_create_collection(name=CHROMA_COLLECTION,embedding_function=ef,metadata={"hnsw:space":"cosine"})


def index_products():
    """
    Pull all products + reviews from MySQL and index them into ChromaDB.
    Run once on startup (skips if already indexed).
    """
    collection = get_or_create_collection()
 
    if collection.count() > 0:
        print(f"[VectorStore] Already indexed {collection.count()} documents. Skipping.")
        return
 
    engine = get_engine()
    from sqlalchemy import text as sql_text
 
    with engine.connect() as conn:
        rows = conn.execute(sql_text("""
            SELECT
                p.product_id,
                p.name,
                p.price,
                p.rating,
                p.description,
                c.category_name,
                GROUP_CONCAT(r.review_text SEPARATOR ' | ') AS reviews
            FROM products p
            LEFT JOIN categories c ON p.category_id = c.category_id
            LEFT JOIN reviews r ON p.product_id = r.product_id
            GROUP BY p.product_id
        """)).fetchall()
 
    docs, ids, metas = [], [], []
    for row in rows:
        doc_text = (
            f"Product: {row.name}. "
            f"Category: {row.category_name}. "
            f"Price: ₹{row.price}. "
            f"Rating: {row.rating}/5. "
            f"Description: {row.description}. "
            f"Customer reviews: {row.reviews or 'No reviews yet'}."
        )
        docs.append(doc_text)
        ids.append(str(row.product_id))
        metas.append({
            "product_id": row.product_id,
            "name": row.name,
            "price": float(row.price),
            "rating": float(row.rating) if row.rating else 0.0,
            "category": row.category_name,
        })
 
    collection.add(documents=docs, ids=ids, metadatas=metas)
    print(f"[VectorStore] Indexed {len(docs)} products.")
 
 

def dense_search(query:str , n_results : int = 5)->list[dict]:
    collection = get_or_create_collection()
    results = collection.query(query_texts=[query], n_results=n_results)
    output = []
    for i,doc in enumerate(results["documents"][0]):
        output.append({
            "source" : "vector",
            "text" : doc,
            "metadata" : results["metadatas"][0][i],
            "distance" : results["distances"][0][i]
        })
    return output


def hybrid_context(question : str , sql_results: list[dict] , top_k : int = 3)->str:
    vector_hits = dense_search(question, n_results=top_k)
    parts = []
    if sql_results:
        parts.append("SQL Query Results:\n")
        for row in sql_results[:10]:
            parts.append(str(row))
    if vector_hits:
        parts.append("Semantically relevant products:\n")
        for hit in vector_hits:
            parts.append(f"-{hit['text'][:300]}")
    return "\n".join(parts)