# from transformers import AutoTokenizer, AutoModel
# import torch
# import numpy as np
# import faiss

# tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
# model = AutoModel.from_pretrained('facebook/contriever')

# def text_to_vector(text, max_length=512):
#     inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

# def retrieve_documents_with_dynamic(documents, queries, threshold=0.4):
#     if isinstance(queries, list):
#         query_vectors = np.array([text_to_vector(query) for query in queries])
#         average_query_vector = np.mean(query_vectors, axis=0)
#         query_vector = average_query_vector / np.linalg.norm(average_query_vector)
#         query_vector = query_vector.reshape(1, -1)
#     else:
#         query_vector = text_to_vector(queries)
#         query_vector = query_vector / np.linalg.norm(query_vector)
#         query_vector = query_vector.reshape(1, -1)

#     document_vectors = np.array([text_to_vector(doc) for doc in documents])
#     document_vectors = document_vectors / np.linalg.norm(document_vectors, axis=1, keepdims=True)
#     dimension = document_vectors.shape[1]
    
#     index = faiss.IndexFlatIP(dimension)
#     index.add(document_vectors)
#     lims, D, I = index.range_search(query_vector, threshold)
#     start = lims[0]
#     end = lims[1]
#     I = I[start:end]

#     if len(I) == 0:
#         top_documents = []
#         idx = []
#     else:
#         idx = I.tolist()
#         top_documents = [documents[i] for i in idx]

#     return top_documents, idx

import sys
import numpy as np
import sklearn
import numpy as np
import numpy.typing as npt

if not hasattr(npt, 'NDArray'):
    npt.NDArray = np.ndarray
from PIL import Image  

from sklearn.mixture import GaussianMixture
# ... rest of your original code ...import numpy as np
from rank_bm25 import BM25Okapi
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from rank_bm25 import BM25Okapi

def retrieve_documents_with_temporal_rankning(documents, queries, alpha=0.1, max_k=10):
    """
    Retrieve documents using GMM-based automatic top-k selection.
    
    Args:
        documents: List of documents
        queries: List of queries or a single query string
        alpha: Time decay factor
        max_k: Maximum number of components to try for GMM
    
    Returns:
        top_documents: List of retrieved documents
        idx: Indices of retrieved documents
        k_star: Optimal k determined by GMM+BIC
    """
    if len(documents) == 0:
        return [], [], 0
        
    # Tokenize documents for BM25
    tokenized_docs = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    
    # Handle different query formats
    if isinstance(queries, list):
        query_text = " ".join(queries)
    else:
        query_text = queries
    
    tokenized_query = query_text.split()
    bm25_scores = np.array(bm25.get_scores(tokenized_query))  # shape: [N]

    # Temporal decay weights: exp(-α * |T_q - T_i|)
    timestamps = np.array(timestamps)
    time_diffs = np.abs(timestamps - query_time)
    decay = np.exp(-alpha * time_diffs)  # shape: [N]

    # Apply Equation (5): re-weight BM25 by temporal decay
    numerator = bm25_scores * decay
    denominator = np.sum(numerator) + 1e-8  # Avoid divide by zero
    weights = numerator / denominator

    # Select Top-K segments based on reweighted scores (Equation 6)
    top_indices = np.argsort(-weights)[:top_k]
    top_documents = [documents[i] for i in top_indices]

    return top_documents, top_indices.tolist(), weights.tolist()


# def retrieve_documents_with_dynamic(documents, queries, alpha=0.1, max_k=10):
#     """
#     Retrieve documents using GMM-based automatic top-k selection.
    
#     Args:
#         documents: List of documents
#         queries: List of queries or a single query string
#         alpha: Time decay factor
#         max_k: Maximum number of components to try for GMM
    
#     Returns:
#         top_documents: List of retrieved documents
#         idx: Indices of retrieved documents
#         k_star: Optimal k determined by GMM+BIC
#     """
#     if len(documents) == 0:
#         return [], [], 0
        
#     # Tokenize documents for BM25
#     tokenized_docs = [doc.split() for doc in documents]
#     bm25 = BM25Okapi(tokenized_docs)
    
#     # Handle different query formats
#     if isinstance(queries, list):
#         query_text = " ".join(queries)
#     else:
#         query_text = queries
    
#     tokenized_query = query_text.split()
#     bm25_scores = np.array(bm25.get_scores(tokenized_query))
    
#     # Apply time-based weighting
#     timestamps = np.arange(len(documents))
#     query_time = timestamps[-1]
#     time_diffs = np.abs(query_time - timestamps)
#     time_weights = np.exp(-alpha * time_diffs)
    
#     weighted_scores = bm25_scores * time_weights
    
#     # If all scores are 0, return empty results
#     if np.all(weighted_scores == 0):
#         return [], [], 0
    
    

# def find_optimal_k_gmm(data, max_k=10):
#     """
#     Find optimal number of components (K) using GMM and BIC.
    
#     Args:
#         data: Data array (n_samples, n_features)
#         max_k: Maximum number of components to try
    
#     Returns:
#         k_star: Optimal number of components
#     """
#     n_samples = data.shape[0]
    
#     # Need at least 2 data points for GMM
#     if n_samples < 2:
#         return 1
    
#     # Try different values of K
#     max_k = min(max_k, n_samples)
#     bic_scores = []
#     log_likelihoods = []
#     valid_k_values = []
    
#     # Start from K=1
#     for k in range(1, max_k + 1):
#         try:
#             # Fit GMM with k components
#             gmm = GaussianMixture(
#                 n_components=k,
#                 covariance_type='full',
#                 random_state=42,
#                 reg_covar=1e-4,  # Add regularization to avoid singular covariance matrices
#                 max_iter=100
#             )
#             gmm.fit(data)
            
#             # Calculate log-likelihood
#             log_likelihood = gmm.score(data) * n_samples  # Multiply by n_samples to get total log-likelihood
#             log_likelihoods.append(log_likelihood)
            
#             # Calculate BIC as per formula: BIC_K = log(N)·d_K - 2·log(L_K)
#             # Where d_K = 3K - 1 for GMM with full covariance (as shown in the image)
#             d_k = 3 * k - 1
#             bic = np.log(n_samples) * d_k - 2 * log_likelihood
#             bic_scores.append(bic)
#             valid_k_values.append(k)
#         except:
#             # Skip if GMM fails to converge for this k
#             continue
    
#     # Find k_star (k that minimizes BIC)
#     if len(bic_scores) > 0:
#         min_bic_idx = np.argmin(bic_scores)
#         k_star = valid_k_values[min_bic_idx]
#         return k_star
#     else:
#         # Fallback if all GMM fits failed
#         return 0

