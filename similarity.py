from sentence_transformers import SentenceTransformer, SimilarityFunction
import numpy as np


def get_similarity(resume, jd):
    # Compute embeddings for both lists
    model = SentenceTransformer('bert-base-nli-mean-tokens', similarity_fn_name=SimilarityFunction.COSINE)

    # Encode the resume and job description
    resume_embedding = model.encode(resume)  # Encoding the resume
    job_description_embedding = model.encode(jd)  # Encoding the job description

    similarity_score = ([resume_embedding], [job_description_embedding])[0][0]

    # Rank and filter the top k most suitable jobs
    k = 3  # Number of top jobs to select
    top_k_indices = np.argsort(similarity_score)[::-1][:k]
    top_k_jobs = [similarity_score[i] for i in top_k_indices]
    top_k_scores = [similarity_score[i] for i in top_k_indices]
