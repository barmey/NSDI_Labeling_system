import os
import subprocess
from collections import defaultdict

from matplotlib import pyplot as plt

import data_augmentaion
import utils

import seaborn as sns
import pandas as pd
import numpy as np


def compare_all_devices(devices_dict, devices_details, keys_to_compare):
    # Gather all device names
    all_devices = [device_name for device_list in devices_dict.values() for device_name in device_list]

    # Generate all pairs of devices for comparison
    for key in keys_to_compare:
        print(f"Computing Jaccard similarities for key: {key}")

        # Initialize a matrix to store the similarities
        similarity_matrix = pd.DataFrame(0.0, index=all_devices, columns=all_devices)

        for device_list in devices_dict.values():
            for i in range(len(device_list)-1):
                for j in range(i+1, len(device_list)):
                    print(f"Comparing {device_list[i]} and {device_list[j]}")

                    device1_details = devices_details[device_list[i]]
                    device2_details = devices_details[device_list[j]]

                    # Assuming the details are stored as lists in the dictionary
                    device1_key_details = set(device1_details.get(key, []))
                    device2_key_details = set(device2_details.get(key, []))

                    similarity = jaccard_similarity(device1_key_details, device2_key_details)
                    print(f"For key '{key}', Jaccard similarity is: {similarity:.2f}")

                    # Store the similarity in the matrix
                    similarity_matrix.loc[device_list[i], device_list[j]] = similarity
                    similarity_matrix.loc[device_list[j], device_list[i]] = similarity

        # Plot a heatmap of the similarities
        sns.heatmap(similarity_matrix.astype(float), annot=True, cmap="YlGnBu")
        plt.title(f'Jaccard Similarities for key {key}')
        plt.show()

# define your dictionaries and keys to compare, then call the function

def generate_csvs():
    tshark = "C:\\Program Files\\Wireshark\\tshark.exe"
    inputdir = 'G:\\Dropbox\\Dropbox\\IoT - Meyuhas\\IoT_lab\\pcaps\\data-imc19\\merged_pcaps_only\\uk_merged'
    for pcapFilename in os.listdir(inputdir):
        if not pcapFilename.endswith(".pcapng"):
            continue
        with open("G:\\Dropbox\\Dropbox\\IoT - Meyuhas\\IoT_lab\\pcaps\\data-imc19\\merged_pcaps_only\\uk_merged\\csv_find_names\\"+ pcapFilename+ ".csv","w") as outfile:
            subprocess.run([tshark, "-r",
                os.path.join(inputdir, pcapFilename),
                            "-T", "fields", "-e", "eth.src", "-e", "eth.dst",
                            "-e", "ip.src", "-e", "ip.dst",
                            "-e", "dhcp.option.hostname", "-e", "dns.ptr.domain_name", "-e", "http.user_agent", "-e",
                            "dns.qry.name", "-e",
                            "tls.handshake.extensions_server_name", "-e", "x509ce.dNSName", "-e",
                            "x509sat.printableString", "-e", "http.request.uri",
                            "-E", "header=y", "-E", "separator=,", "-E",
                            "quote=d", "-E", "occurrence=f"],
                stdout=outfile, check=True)


def csv_to_dict(gapminder_csv_url):
    #gapminder_csv_url = "G:\\Dropbox\\Dropbox\\IoT - Meyuhas\\IoT_lab\\pcaps\\data-imc19\\merged_pcaps_only\\uk_merged\\outfile.csv"
    fields = ["eth.src","eth.dst","ip.src","ip.dst","dhcp.option.hostname","dns.ptr.domain_name","http.user_agent","dns.qry.name","tls.handshake.extensions_server_name","x509ce.dNSName","http.request.uri","x509sat.printableString"]
    fields = ["dhcp.option.hostname","dns.ptr.domain_name","http.user_agent","dns.qry.name","tls.handshake.extensions_server_name","x509ce.dNSName","http.request.uri","x509sat.printableString"]
    record = pd.read_csv(gapminder_csv_url)
    details = {}
    for field in fields:
        details[field] = list(record[field].unique())
    details['device_mac'] = record[record['dhcp.option.hostname'].notna()]['eth.src'].iloc[0] if not record[record['dhcp.option.hostname'].notna()]['eth.src'].empty else ''
    #device_mac = record[record['dhcp.option.hostname'].notna()]['eth.src'].iloc[0]
    details['OUI'] = p.get_manuf(details['device_mac']) if details['device_mac'] != '' else ''
    details['filename'] = os.path.basename(gapminder_csv_url).split('.')[0]
    return details


def dict_to_file(dict,json_name):
    # load json module
    import json
    json = json.dumps(dict)
    f = open(json_name, "w")
    f.write(json)
    f.close()
def dict_to_csv(my_dictionary,csv_name):
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in my_dictionary.items()]))
    df.to_csv(csv_name)
    return df

def generate_json_files():
    inputdir = 'G:\\Dropbox\\Dropbox\\IoT - Meyuhas\\IoT_lab\\pcaps\\data-imc19\\merged_pcaps_only\\uk_merged\\csv_find_names'
    dataframes = []
    for csvFilename in os.listdir(inputdir):
        if not csvFilename.endswith(".csv"):
            continue
        dataframes.append(dict_to_csv(csv_to_dict(os.path.join(inputdir, csvFilename)),os.path.join(inputdir, csvFilename) + "1.csv"))
    return dataframes
#generate_csvs()
import math

def first_google_search(query):
    import requests
    from lxml import html
    from googlesearch import search
    from bs4 import BeautifulSoup


    ## Google Search query results as a Python List of URLs
    search_result_list = list(search(query, tld="co.in", num=10, stop=3, pause=1))
    if len(search_result_list) == 0:
        return 'Unknown'
    page = requests.get(search_result_list[0])

    tree = html.fromstring(page.content)

    soup = BeautifulSoup(page.content, features="lxml")
    return(soup.title.get_text())
def some_functionality():
    devices_data = generate_json_files()
    for device in devices_data:
        for host in device['dhcp.option.hostname'].values:
            if type(host) is not type('str'):
                continue
            print('Host: ',host,'\n Expected:',device['filename'].iloc[0],'\n Actual:',first_google_search(host)[:50],'\n')


def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def compare_cdf_similarity_all_devices(devices_dict, devices_details, keys_to_compare):
    # Gather device names that are in lists of length two or more
    all_devices = [device_name for device_list in devices_dict.values() if len(device_list) > 1 for device_name in device_list]

    # Generate all pairs of devices for comparison
    similarities = []
    for device_list in devices_dict.values():
        if len(device_list) > 1:
            for i in range(len(device_list)-1):
                for j in range(i+1, len(device_list)):
                    device1_details = devices_details[device_list[i]]
                    device2_details = devices_details[device_list[j]]

                    for key in keys_to_compare:
                        #print(f"Computing Jaccard similarities for key: {key}")

                        # Assuming the details are stored as lists in the dictionary
                        device1_key_details = set(device1_details.get(key, []))
                        device2_key_details = set(device2_details.get(key, []))

                        similarity = jaccard_similarity(device1_key_details, device2_key_details)
                        similarity = round(similarity, 2)  # round the similarity to two decimal places
                        if similarity >= 0.75:
                            print(f"For key '{key}' and devices '{device_list[i]}','{device_list[j]}', Jaccard similarity is: {similarity}")

                        # Store the pair, key and similarity in the list
                        similarities.append((f"{device_list[i]} vs {device_list[j]}", key, similarity))

    # Convert the list to a DataFrame for easier plotting
    df = pd.DataFrame(similarities, columns=['Pair', 'Key', 'Jaccard Similarity'])

    # For each key, plot the CDF
    for key in keys_to_compare:
        plt.figure(figsize=(8, 6))
        values = df[df['Key'] == key]['Jaccard Similarity']
        values.hist(cumulative=True, density=1, bins=100)
        plt.title(f'CDF of Jaccard Similarities for Key: {key}')
        plt.xlabel('Jaccard Similarity')
        plt.ylabel('Cumulative Probability')
        plt.grid()
        plt.show()


def compare_avg_similiarty_all_devices(devices_dict, devices_details, keys_to_compare):
    # Generate all pairs of devices for comparison
    similarities = []
    for device_list in devices_dict.values():
        if len(device_list) > 1:
            for i in range(len(device_list) - 1):
                for j in range(i + 1, len(device_list)):
                    device1_details = devices_details[device_list[i]]
                    device2_details = devices_details[device_list[j]]

                    avg_similarity = 0
                    for key in keys_to_compare:
                        #print(f"Computing Jaccard similarities for key: {key}")

                        # Assuming the details are stored as lists in the dictionary
                        device1_key_details = set(device1_details.get(key, []))
                        device2_key_details = set(device2_details.get(key, []))

                        similarity = jaccard_similarity(device1_key_details, device2_key_details)
                        avg_similarity += similarity  # accumulate the similarity

                    avg_similarity = avg_similarity / len(keys_to_compare)  # calculate the average
                    avg_similarity = round(avg_similarity, 2)  # round the similarity to two decimal places
                    #print(f"Average Jaccard similarity is: {avg_similarity}")

                    if avg_similarity > 0.5:
                        print(
                            f"The devices {device_list[i]} and {device_list[j]} have an average similarity of {avg_similarity}")

                        # Store the pair and average similarity in the list
                        similarities.append((f"{device_list[i]} vs {device_list[j]}", avg_similarity))

    # Convert the list to a DataFrame for easier plotting
    df = pd.DataFrame(similarities, columns=['Pair', 'Average Jaccard Similarity'])
    df.plot(kind='bar', x='Pair', y='Average Jaccard Similarity', figsize=(10, 5))  # Adjust as needed
    plt.title('Average Jaccard Similarities for Device Pairs')
    plt.ylabel('Average Jaccard Similarity')
    plt.xticks(rotation=60)  # Rotate x-axis labels for better visibility
    plt.show()


import numpy as np

from numpy import percentile
def compare_similiarties_all_devices(devices_dict, devices_details, keys_to_compare):
    # Define the lists to store the results
    stats_results = []
    similarity_results = []

    # Generate all pairs of devices for comparison
    for device_list in devices_dict.values():
        if len(device_list) > 1:
            for i in range(len(device_list) - 1):
                for j in range(i + 1, len(device_list)):
                    device1_details = devices_details[device_list[i]]
                    device2_details = devices_details[device_list[j]]
                    for key in keys_to_compare:
                        # Assuming the details are stored as lists in the dictionary
                        device1_key_details = set(device1_details.get(key, []))
                        device2_key_details = set(device2_details.get(key, []))

                        similarity = jaccard_similarity(device1_key_details, device2_key_details)

                        if key == 'dhcp.option.hostname' and 0.1 < similarity < 1.0:
                            print(f"For devices {device_list[i]} and {device_list[j]} with key '{key}', similarity is approximately '{similarity}")


                        # Append similarity to the results
                        similarity_results.append((key, similarity))

    # Convert the list to a DataFrame
    similarity_df = pd.DataFrame(similarity_results, columns=['Key', 'Similarity'])

    # Calculate statistics for each key
    for key in keys_to_compare:
        key_data = similarity_df[similarity_df['Key'] == key]['Similarity']
        avg_similarity = key_data.mean()
        sv = key_data.std()
        count_1 = (key_data == 1).sum()

        # Calculate percentiles
        quartiles = percentile(key_data, [25, 50, 75])

        stats_results.append((key, avg_similarity, sv))#, count_1, quartiles[0], quartiles[1], quartiles[2]))

    # Convert stats results to a DataFrame
    stats_df = pd.DataFrame(stats_results,
                            columns=['Key', 'Average Similarity', 'SV'])#, 'Count of 1.0 Similarity', '25th Percentile','Median', '75th Percentile'])

    # Print the DataFrames
    #print(df.to_string())
    print(stats_df.to_string())


def find_shared_domains(devices_details):
    # Create a dictionary where the keys are the domains and the values are sets of (vendor, type) tuples that use each domain
    domain_vendors_types = {}
    for device, details in devices_details.items():
        vendors = frozenset(details['vendor'])  # Treat the list of vendors as a set
        types = frozenset(details['type'])  # Treat the list of types as a set
        for domain in details['dns.qry.name']:
            if domain not in domain_vendors_types:
                domain_vendors_types[domain] = set()
            domain_vendors_types[domain].add((vendors, types))

    # Find the domains that are used by more than one (vendor, type) pair
    shared_domains = [(domain, pairs) for domain, pairs in domain_vendors_types.items() if len(pairs) > 1]
    #shared_domains_filtered = []

    return shared_domains


def cdf(data):
    data_sorted = np.sort(data)
    p = (1. * np.arange(len(data)) / (len(data) - 1))
    plt.plot(data_sorted, p)

def create_cdf_for_query_results(enriched_dict):
    google_fields = [utils.hostnames_google_field, utils.domains_google_field, utils.user_agents_google_field,
                     utils.tls_issuers_google_field, utils.oui_vendors_google_field]

    query_result_counts = []

    for google_field in google_fields:
        for device_data in enriched_dict.values():
            query_counts = defaultdict(int)

            for query_data in device_data[google_field]:
                query_counts[query_data[0]] += 1

            query_result_counts.extend(query_counts.values())

    # Plotting the CDF
    plt.figure(figsize=(5, 3.5))
    cdf(query_result_counts)

    #plt.title('CDF of Number of Results per Query')
    plt.xlabel('Number of Results Returned')
    plt.ylabel('Feature Values (%)')
    #plt.grid(True, which="both", ls="--")
    plt.show()
    plt.savefig(
        '/Users/barmey/Dropbox/IoT-Meyuhas/Labeling Paper/figures/' + f'cdf_queries_search_results_unique_dataset.png',
        bbox_inches='tight')
    plt.clf()

def create_cdf_for_device_queries(enriched_dict):
    fields = [utils.hostname_field, utils.dns_field, utils.user_agents_field, utils.tls_issuers_field, utils.oui_field]

    plt.figure(figsize=(10, 7))

    for field in fields:
        query_counts_per_device = []

        for device_data in enriched_dict.values():
            query_counts_per_device.append(len(set(device_data[field])))

        # Plotting the CDF for the current field
        data_sorted = np.sort(query_counts_per_device)
        p = 1. * np.arange(len(data_sorted)) / (len(data_sorted) - 1)
        plt.plot(data_sorted, p, label=field)

    plt.title('CDF of Number of Queries per Device Across Fields')
    plt.xlabel('Number of Queries per Device')
    plt.ylabel('Devices (%)')
    plt.legend()
    plt.grid(True, which="both", ls="--",bbox_inches='tight')
    plt.show()



def compute_avg_std_queries(enriched_dict):
    fields = [utils.hostname_field, utils.dns_field, utils.user_agents_field, utils.tls_issuers_field, utils.oui_field]

    avg_queries = {}
    std_queries = {}

    for field in fields:
        query_counts_per_device = [len(device_data[field]) for device_data in enriched_dict.values()]

        avg_queries[field] = round(np.mean(query_counts_per_device),2)
        std_queries[field] = round(np.std(query_counts_per_device),2)

    return avg_queries, std_queries
# define your dictionaries and keys to compare, then call the function
ext_dict = data_augmentaion.build_enriched_dict()
ext_dict_unique = {key: ext_dict[key] for key in utils.devices_unique_array if key in ext_dict}

create_cdf_for_query_results(ext_dict_unique)
avg_queries, std_queries = compute_avg_std_queries(ext_dict_unique)

# Display in a table format using pandas DataFrame
df = pd.DataFrame({
    'Field': [utils.friendly_names.get(key, key) for key in avg_queries.keys()],
    'Average Queries': list(avg_queries.values()),
    'Standard Deviation': list(std_queries.values())
})

print(df)

keys_to_compare = ['dns.qry.name', 'http.request.full_uri', utils.user_agents_field,'dhcp.option.hostname',utils.tls_issuers_field]
#compare_cdf_similarity_all_devices(utils.devices_dict_unique,ext_dict,keys_to_compare)
compare_avg_similiarty_all_devices(utils.devices_dict_unique,ext_dict,keys_to_compare)
compare_similiarties_all_devices(utils.devices_dict_unique,ext_dict,keys_to_compare)

# Call the function
shared_domains = find_shared_domains(ext_dict)
noisy_domains_across_types_and_vendors = []
# Print the domains and their associated vendors and types
for domain, pairs in shared_domains:
    vendors = []
    types = []
    for v, t in pairs:
        vendors.append(set(v))
        types.append(set(t))

    # Check if there is an intersection between any pair of vendor sets or type sets
    vendor_intersection = any(
        vendors[i].intersection(vendors[j]) for i in range(len(vendors)) for j in range(i + 1, len(vendors)))
    type_intersection = any(
        types[i].intersection(types[j]) for i in range(len(types)) for j in range(i + 1, len(types)))
    # Only print the domain if there is no intersection
    if not vendor_intersection and not type_intersection:
        vendors_str = 'Vendors: ' + ', '.join([", ".join(list(v)) for v in vendors])
        types_str = 'Types: ' + ', '.join([", ".join(list(t)) for t in types])
        noisy_domains_across_types_and_vendors.append(domain)
        print(f"Domain: {domain}, {vendors_str}, {types_str}")

domains_shared_across_types = []
# Print the domains and their associated types
for domain, pairs in shared_domains:
    types = []
    for _, t in pairs:
        types.append(set(t))

    # Check if there is an intersection between any pair of type sets
    type_intersection = any(
        types[i].intersection(types[j]) for i in range(len(types)) for j in range(i + 1, len(types)))

    # Only print the domain if there is no intersection
    if not type_intersection:
        types_str = 'Types: ' + ', '.join([", ".join(list(t)) for t in types])
        domains_shared_across_types.append(domain)
        print(f"Domain: {domain}, {types_str}")
print(f"noisy_domains_across_types = {domains_shared_across_types}")

print(f"noisy_domains_across_types_and_vendors = {noisy_domains_across_types_and_vendors}")
