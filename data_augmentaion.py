import statistics

import search_api
from utils import *


def build_enriched_dict():
    dict_devices = read_pkl_file(pkl_file_csvs_data)
    results_dict = {
        hostnames_google_field: read_pkl_file(pkl_file_hostnames_search),
        domains_google_field: read_pkl_file(pkl_file_domains_search),
        user_agents_google_field: read_pkl_file(pkl_file_user_agents_search),
        tls_issuers_google_field: read_pkl_file(pkl_file_issuers_search),
        oui_vendors_google_field: read_pkl_file(pkl_file_oui_vendors_search)
    }

    fields = [hostname_field, dns_field, user_agents_field, tls_issuers_field, oui_field]
    data_lens = []
    for device in list(dict_devices.keys()):
        for field, google_field in zip(fields, results_dict.keys()):
            dict_devices[device].setdefault(google_field, [])
            for query in dict_devices[device][field]:
                if query in results_dict[google_field].keys():
                    relevance_texts = search_api.get_organic_search_snippets_from_search_results(
                        results_dict[google_field][query])
                    for i, text in enumerate(relevance_texts):
                        # i//2 will increase the index every 2 elements since we have title and description
                        dict_devices[device][google_field].append((query, i // 2 , text))
                elif query.lower() in results_dict[google_field].keys():
                    relevance_texts = search_api.get_organic_search_snippets_from_search_results(
                        results_dict[google_field][query.lower()])
                    for i, text in enumerate(relevance_texts):
                        # i//2 will increase the index every 2 elements since we have title and description
                        dict_devices[device][google_field].append((query.lower(), i // 2 , text))
        data_lens.append(len(dict_devices[device][field]))
        if len(dict_devices[device][dns_field]) == 0 and \
                len(dict_devices[device][hostname_field]) == 0 and \
                len(dict_devices[device][tls_issuers_field]) == 0:
            del dict_devices[device]
    #print(statistics.mean(data_lens))
    return dict_devices
    if False:
        for query in dict_devices[device][dns_field]:
            if query.lower() in domains_results.keys() and query.lower() not in noisy_domains_across_types:
                relevance_text = search_api.get_organic_search_snippets_from_search_results(domains_results[query.lower()])
                dict_devices[device][domains_google_filtered_noisy_domains_field].extend(relevance_text)

def save_enriched_data_pkl_file(file_to_save):
    save_dict_to_file_pkl(build_enriched_dict(),file_to_save)


def export_dict_to_json(outfile_path, dict_devices):
    import json

    # convert dictionary to JSON string
    json_string = json.dumps(dict_devices)

    # write JSON string to file
    with open(outfile_path, "w") as outfile:
        outfile.write(json_string)
