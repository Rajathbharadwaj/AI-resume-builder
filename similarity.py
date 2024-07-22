from sentence_transformers import SentenceTransformer, SimilarityFunction
import numpy as np
from typing import List
import pandas as pd


def get_similarity(resume, jobs):
    jobs['description'] = jobs['description'].fillna("")

    def is_incomplete(description):
        # Check if the description is below a certain length threshold or contains placeholder text
        return len(description) < 100 or '...' in description

    incomplete_series = jobs['description'].apply(is_incomplete)
    complete_series = ~incomplete_series
    filtered_df = jobs[complete_series]
    jds = filtered_df['description'].to_list()
    titles = jobs['title'].to_list()
    companies = jobs['company'].to_list()
    urls = jobs['job_url'].to_list()
    direct_url = jobs['job_url_direct'].to_list()
    # Resume text extracted from the PDF
    # Load pre-trained Sentence Transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2', similarity_fn_name=SimilarityFunction.COSINE)

    # Encode the resume and job description
    resume_embedding = model.encode(resume)  # Encoding the resume
    # job_description_embedding = model.encode(job_description)  # Encoding the job description

    # Calculate cosine similarity
    # similarity_score = cosine_similarity([resume_embedding], [job_description_embedding])[0][0]

    # Display the similarity score

    # print(f"Similarity Score: {similarity_score:.4f}")

    # For multiple job descriptions, you can repeat the process and rank them by similarity score

    job_descriptions = jds  # Add more job descriptions as needed
    job_embeddings = model.encode(job_descriptions)  # Encoding all job descriptions

    # Calculate similarities for all job descriptions

    similarity_scores = model.similarity([resume_embedding], job_embeddings)[0]
    print(f"Similarity Score: {similarity_scores}")

    # Rank and filter the top k most suitable jobs

    k = 13  # Number of top jobs to select
    top_k_indices = np.argsort(np.array(similarity_scores))[::-1][:k]
    top_k_jobs = [job_descriptions[i] for i in top_k_indices]
    top_k_scores = [similarity_scores[i] for i in top_k_indices]
    top_k_titles = [titles[i] for i in top_k_indices]
    top_k_companies = [companies[i] for i in top_k_indices]
    top_k_urls = [urls[i] for i in top_k_indices]
    top_k_urls_direct = [direct_url[i] if direct_url[i] is not None else urls[i] for i in top_k_indices]


    # Display the top k most suitable jobs

    top_k_df = pd.DataFrame({
        'Job Description': top_k_jobs,
        'Title': top_k_titles,
        'Company': top_k_companies,
        'URL': top_k_urls_direct,
        "LinkedIn URL": top_k_urls,
        'Similarity Score': top_k_scores
    })
    top_k_df.drop_duplicates(subset=['Company'], inplace=True)
    return top_k_df
