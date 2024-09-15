import time

from serpapi import GoogleSearch
import os

import secrets
import utils



def send_search_request(query):
    check = check_if_query_exist(query)
    if check[0]:
        return check[1]
    params = {
        "engine": "google",
        "q": query.lower(),
        "api_key": secrets.api_key_serp_api
    }
    try:
        # Start the timer
        start_time = time.time()

        # Your code to measure
        search = GoogleSearch(params)
        results = search.get_dict()

        # End the timer
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        print(f"The runtime of the code is {elapsed_time} seconds.")
    except:
        time.sleep(10)
        search = GoogleSearch(params)
        results = search.get_dict()

    return results


def check_if_query_exist(query):
    for pkl in os.listdir(utils.pkl_files_searches_path):
        if not pkl.endswith(".pkl"):
            continue
        dict_queries = utils.read_pkl_file(utils.pkl_files_searches_path + pkl)
        if query.lower() in list(map(str.lower,dict_queries.keys())):
            if query in dict_queries.keys():
                return True, dict_queries[query]
            if query.lower() in dict_queries.keys():
                return True, dict_queries[query.lower()]
    return False, None


def get_organic_search_snippets_from_search_results(result):
    if "organic_results" not in result:
        return list()
    organic_results = result["organic_results"]
    text_list = []
    for org_res in organic_results:
        text = ''
        if 'snippet' in org_res.keys():
            text = org_res['snippet']
        if 'title' in org_res.keys():
            text = text + org_res['title']
        text_list.append(text)
    return text_list