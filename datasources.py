from serpapi import GoogleSearch
from cred import *


def get_jobs():
    params = {
        "api_key": api_key,
        "engine": "google_jobs",
        "google_domain": "google.ca",
        "q": "Machine Learning, ",
        "hl": "en",
        "gl": "ca",
        "location": "Canada"
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results


print(get_jobs()['jobs_results'])
