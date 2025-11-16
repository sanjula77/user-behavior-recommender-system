# TF-IDF + item-item similarity
# ml/phase2_hybrid/content_model.py
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import joblib
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_content_model(item_list):
    """
    item_list: list of item identifiers (e.g., normalized URL paths or titles)
    Returns TF-IDF vectorizer and item similarity matrix (dense or sparse)
    """
    corpus = [str(it) for it in item_list]
    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X = vectorizer.fit_transform(corpus)
    # compute cosine similarity matrix
    sim_matrix = linear_kernel(X, X)  # dense; small item sets OK
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    np.save(os.path.join(MODEL_DIR, "item_sim_matrix.npy"), sim_matrix)
    print("Saved TF-IDF vectorizer and item similarity matrix.")
    return vectorizer, sim_matrix
