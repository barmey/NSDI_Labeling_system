#import torch
import random
import statistics
import time

import numpy

import data_calssification_LLM
import decsion_tree
import fing_idenitfication
#import ner_frequencies
#import label_device
import parse_pcap
import utils
from utils import *
import data_enrichment
import data_augmentaion
import data_classification_RB
from transformers import pipeline
import data_classification_nlp_model
from tqdm import tqdm
import json
import numpy as np
def parse_pcaps():
    #parse_pcap.generate_csvs('/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/pcaps/Datasets/data-imc19/merged_pcaps_only/uk_merged',
    #                         'IMC19')
    #parse_pcap.generate_csvs('/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/pcaps/Datasets/data-imc19/merged_pcaps_only/us_merged',
    #                         'IMC19')
    #parse_pcap.generate_csvs('/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/pcaps/Datasets/captures_IoT-Sentinel/merged_devices',
    #                                                  'sentinel')
    #parse_pcap.generate_csvs('/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/pcaps/Datasets/Sivanathan/pcaps/',
    #                                                 'sivanathan')
    #parse_pcap.generate_csvs('/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/pcaps/Datasets/yourthings/pcaps/merged_devices',
    #                         'yourthings')
    #parse_pcap.generate_csvs('/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/pcaps/Datasets/OurLab/',
    #                        'our_lab')
    #                        'our_lab')

    #dict_devices = parse_pcap.read_devices_details()
    dict_devices = read_pkl_file(pkl_file_csvs_data)
    #dict_devices.update(parse_pcap.pre_proccessing_checkpoint('/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/pcaps/Datasets/CheckPoint_private/data_for_idc.csv'))
    dict_devices = parse_pcap.fill_with_vendors(dict_devices)
    dict_devices = parse_pcap.fill_with_types(dict_devices)
    dict_devices = parse_pcap.fill_with_categories(dict_devices)
    dict_devices = parse_pcap.add_filtered_domains_field(dict_devices,domains_filtered_field,utils.valid_external_domain)
    save_dict_to_file_pkl(dict_devices, pkl_file_csvs_data)

def data_enrichment_process():
    dict_devices = read_pkl_file(pkl_file_csvs_data)
    hostnames = get_all_field_from_devices_dict(dict_devices, 'dhcp.option.hostname')
    domains = get_all_field_from_devices_dict(dict_devices, 'dns.qry.name')
    user_agents = get_all_field_from_devices_dict(dict_devices, 'http.user_agent')
    tls_issuers = get_all_field_from_devices_dict(dict_devices, 'x509sat.printableString')
    oui_vendors = get_all_field_from_devices_dict(dict_devices, 'oui_vendor')

    user_agents = list(filter(lambda user_agent: valid_user_agent(user_agent), user_agents))
    tls_issuers = list(filter(lambda issuer: valid_issuer(issuer), tls_issuers))
    data_enrichment.get_and_save_search_tofile(pkl_file_user_agents_search, user_agents)
    data_enrichment.get_and_save_search_tofile(pkl_file_issuers_search, tls_issuers)
    data_enrichment.get_and_save_search_tofile(pkl_file_hostnames_search, hostnames)
    data_enrichment.get_and_save_search_tofile(pkl_file_oui_vendors_search, oui_vendors)

    domains_filtered = list(filter(lambda domain: valid_external_domain(domain), domains))
    domains_filtered = numpy.unique(list(map(str.lower, domains_filtered))).tolist()
    data_enrichment.get_and_save_search_tofile(pkl_file_domains_search, domains_filtered)

def ner_experiments_list(dict_enriched_devices,field_to_search_in):
    counter = 0
    list_of_vendors = []
    for device in tqdm(dict_enriched_devices.keys()):
        if device not in devices_unique_array:
            continue
        vendors = ner_frequencies.extract_orgs_from_device_using_gpt_ner(dict_enriched_devices[device], field_to_search_in)
        flag = False
        for vendor_real in dict_enriched_devices[device]['vendor']:
            for vendor_ner in vendors:
                if vendor_ner in vendor_real or vendor_real in vendor_ner:
                    counter += 1
                    flag = True
                    list_of_vendors.append(vendor_real)
                    break
            if flag:
                break
    print(f"NER identified {counter} vendors out of {len(devices_unique_array)}")
    counter = 0
    for device in tqdm(dict_enriched_devices.keys()):
        if device not in devices_unique_array:
            continue
        vendors = list_of_vendors
        for vendor_real in dict_enriched_devices[device]['vendor']:
            if vendor_real in vendors:
                counter += 1
                break
    print(f"NER identified {counter} vendors out of {len(devices_unique_array)}")

def ner_experiment_half_list_known(dict_devices,fields_to_search_in):
    devices_unique_vendor = utils.select_random_device_by_vendor(dict_devices)
    devices_half_list = random.sample(devices_unique_vendor,int(len(devices_unique_vendor)/2))
    print(f"size of vendor list {len(devices_unique_vendor)}")
    selected_devices_dict = {k:dict_devices[k] for k in dict_devices if k in devices_half_list}
    ner_experiments_list(selected_devices_dict,fields_to_search_in)
def data_classification_string_matching_vendor(dict_enriched_devices,field_to_search_in,data_fields,use_NER=False):
    vendors = utils.read_csv_single_column_to_list(vendors_path)
    vendors = utils.read_csv_single_column_to_list('verified_iot_manufacturers_no_categories_names_only.txt')
    catalog_types = utils.read_json('vendor_types_dict_vered.json')
    vendors = list(catalog_types.keys())
    dict_results = {}
    index = 0
    print("data_classification_string_matching_vendor")
    for device in tqdm(dict_enriched_devices.keys()):
        if use_NER:
            #vendors = ner_frequencies.extract_orgs_from_device(dict_enriched_devices[device],field_to_search_in)
            vendors = utils.acquired_by_gpt_vendors_list_gpt_filtered_one_by_one
            #print(f"NER vendors:{vendors} real:{dict_enriched_devices[device]['vendor']}")
        vec = data_classification_RB.calculate_vector_counter_word_in_text_fields_weighting(dict_enriched_devices[device], field_to_search_in, vendors,data_fields=data_fields)
        #vec = data_classification_RB.calculate_vector_counter_word_in_text_equal_fields(dict_enriched_devices[device], field_to_search_in, vendors)
        dict_results[device] = dict()
        dict_results[device]['classified_string_matching_vendor_vector'] = vec
        dict_results[device]['vendor'] = dict_enriched_devices[device]['vendor']
        index = index + 1
    if use_NER:
        res_path = utils.get_results_filename('rb-vendor-ner', field_to_search_in, path_classification_results,data_fields)
    else:
        res_path = utils.get_results_filename('rb-vendor-gpt-catalog', field_to_search_in, path_classification_results,data_fields)
    utils.save_dict_results_to_file_pkl(dict_results,res_path)
    count_matching_vendors(dict_results)
    return dict_results
    #utils.save_dict_to_file_pkl(dict_results, pkl_file_classified_vendor_rb)


# Function to count how many devices have a match in the classified_string_matching_vendor_vector list
def count_matching_vendors(devices_data):
    match_count = 0

    for device, data in devices_data.items():
        if device not in devices_unique_array:
            continue
        # Convert vendor values to lowercase for case-insensitive comparison
        vendor_values = set(v.lower() for v in data['vendor'])

        # Extract the set of vendors from 'classified_string_matching_vendor_vector' to lower case
        classified_vendors = set(item[0].lower() for item in data['classified_string_matching_vendor_vector'])

        # Check if any vendor value is in the classified list
        if vendor_values & classified_vendors:  # Set intersection to check for common elements
            match_count += 1
    print(f"Number of devices with a matching vendor: {match_count}")
    return match_count
def data_classification_oui_vendor(dict_enriched_devices):
    dict_results = {}
    print("data_classification_oui_vendor")
    for device in tqdm(dict_enriched_devices.keys()):
        vec = [dict_enriched_devices[device][oui_field]]
        dict_results[device] = dict()
        dict_results[device]['classified_oui_vector'] = vec
        dict_results[device]['vendor'] = dict_enriched_devices[device]['vendor']
    utils.save_dict_to_file_pkl(dict_results, pkl_file_classified_vendor_oui)

def data_classification_string_matching_type(dict_enriched_devices,fields_to_search,data_fields=utils.fields_data,use_list=False):
    field_to_search_in = fields_to_search #[dns_field, hostname_field, hostnames_google_field, domains_google_field]
    types = utils.read_csv_single_column_to_list(types_path)
    dict_results = {}
    index = -1
    dict_vendor_stringmatching_results = utils.read_pkl_file(utils.get_results_filename('rb-vendor', field_to_search_in, path_classification_results,data_fields))
    with open(vendor_type_list_chat_gpt4_path, 'r') as f:
        dict_vendor_type = json.load(f)
        dict_vendor_type = lowercase_keys(dict_vendor_type)
    result_filepath = utils.get_results_filename('rb-type',field_to_search_in,path_classification_results,data_fields)
    print("data_classification_string_matching_type")
    for device in tqdm(dict_enriched_devices.keys()):
        index = index + 1
        if use_list:
            result_filepath = utils.get_results_filename('rb-type-list-based', field_to_search_in,
                                                         path_classification_results,data_fields)
            #guessed_vendor = dict_vendor_stringmatching_results[device]['classified_string_matching_vendor_vector'][0][0].lower()
            guessed_vendor = dict_enriched_devices[device]['vendor'][0]
            if guessed_vendor in dict_vendor_type:
                types = dict_vendor_type[guessed_vendor]
            else:
                types = utils.read_csv_single_column_to_list(types_path)
        vec = data_classification_RB.calculate_vector_counter_word_in_text_equal_fields(dict_enriched_devices[device], field_to_search_in, types,1.0)
        dict_results[device] = dict()
        dict_results[device]['classified_string_matching_type_vector'] = vec
        dict_results[device]['type'] = dict_enriched_devices[device]['type']
        if len(types) == 1:
            dict_results[device]['classified_string_matching_type_vector'] = [(types[0],100)]
    utils.save_dict_results_to_file_pkl(dict_results,result_filepath)
    #utils.save_dict_to_file_pkl(dict_results, pkl_file_classified_type_rb)

def data_fing_identification(dict_enriched_devices,fields_to_search):
    field_to_search_in = fields_to_search #[dns_field, hostname_field, hostnames_google_field, domains_google_field]
    #types = utils.read_csv_single_column_to_list(types_path)
    dict_results = {}
    result_filepath = utils.get_results_filename('fing',field_to_search_in,path_classification_results,data_fields={})
    print("data_fing_identification")
    success_vendor = 0
    success_funtion = 0
    success_both = 0
    both_flag = False
    for device in tqdm(devices_unique_array):
        vendor,function,conf = fing_idenitfication.device_fing_query(dict_enriched_devices[device], field_to_search_in)
        dict_results[device] = dict()
        dict_results[device]['class_fing_vec'] = {'0':{'vendor':vendor,'type':function,'confidence':conf}}
        #dict_results[device]['fing_function'] = function
        #dict_results[device]['fing_confidence'] = conf
        dict_results[device]['type'] = dict_enriched_devices[device]['type']
        dict_results[device]['vendor'] = dict_enriched_devices[device]['vendor']
        if function == '':
            print("empty fing function result")
        if vendor.lower() in [x.lower() for x in dict_results[device]['vendor']]:
            success_vendor = success_vendor + 1
            both_flag = True
        else:
            #print(vendor,dict_results[device]['vendor'])
            both_flag = False
            #print(f'device: {device}, fing says: {vendor}, Real Vendor: {dict_results[device]["vendor"]}, OUI: {dict_enriched_devices[device][oui_field]}, hostname:{dict_enriched_devices[device][hostname_field]}')

        if function.lower() in [x.lower() for x in dict_results[device]['type']] or len([1 for x in dict_results[device]['type'] if (x.lower() in function.lower() or function.lower() in x.lower())]) > 0:
            success_funtion = success_funtion + 1
            if both_flag:
                success_both = success_both + 1
        else:
            print(f'device: {device}, fing says: {function}, Real function: {dict_results[device]["type"]}, OUI: {dict_enriched_devices[device][oui_field]}, hostname:{dict_enriched_devices[device][hostname_field]}')

    print(success_vendor,success_funtion,success_both,len(devices_unique_array))
    utils.save_dict_results_to_file_pkl(dict_results,result_filepath)

def calculate_baseline(dict_enriched_devices):
    with open(vendor_type_list_chat_gpt4_path, 'r') as f:
        dict_vendor_type = json.load(f)
        dict_vendor_type = lowercase_keys(dict_vendor_type)
    # Initialize variables to store probabilities
    chances_one_guess = []
    chances_two_guesses = []
    types_lens = []
    # Iterate over each device in the enriched devices dictionary
    for device, info in dict_enriched_devices.items():
        vendor = info['vendor'][0].lower()  # Assuming 'vendor' is a list and we need the first item
        types = dict_vendor_type.get(vendor, [])  # Get types or an empty list if vendor not found

        if types:  # Check if there are any types available for this vendor
            num_types = len(types)
            types_lens.append(num_types)
            prob_one_guess = 1 / num_types
            # Calculate the probability of guessing right with two tries, without repeating the first choice
            if num_types > 1:
                prob_two_guesses = prob_one_guess + (1 / (num_types - 1))
            else:
                prob_two_guesses = prob_one_guess  # if only one type, the second guess doesn't change the odds
        else:
            print("error!!")
            prob_one_guess = 0
            prob_two_guesses = 0

        chances_one_guess.append(prob_one_guess)
        chances_two_guesses.append(prob_two_guesses)

    # Calculate average probabilities
    average_one_guess = sum(chances_one_guess) / len(chances_one_guess) if chances_one_guess else 0
    average_two_guesses = sum(chances_two_guesses) / len(chances_two_guesses) if chances_two_guesses else 0
    print(f"The average amount of types per vendor is: {statistics.mean(types_lens)}")
    print(f"The random model accuracy (one guess): {average_one_guess}")
    print(f"The random model accuracy (two guesses): {average_two_guesses}")


def type_classification_nlp_model_dynamic_fields(dict_enriched_devices,classifier,fields_tosearch,fields_data,use_list=False,max_exp=False,search_results_number_exp=False):
    field_to_search_in = fields_tosearch
    types = utils.read_csv_single_column_to_list(types_path)
    dict_results = {}
    dict_table_results = {}
    index = 0
    #thresholdParam = ('_threshold'+str(round(threshold*10)))
    #field_to_search_in.append(thresholdParam)
    if max_exp:
        result_filepath = utils.get_results_filename(classifier.model.base_model_prefix+'-maxexp',field_to_search_in,path_classification_results,fields_data)
    elif search_results_number_exp:
        result_filepath = utils.get_results_filename(classifier.model.base_model_prefix+'-divide-search-results-exp',field_to_search_in,path_classification_results,fields_data)
    else:
        result_filepath = utils.get_results_filename(classifier.model.base_model_prefix,field_to_search_in,path_classification_results,fields_data)
    #field_to_search_in.remove(thresholdParam)
    #result_filepath = path_classification_results + utils.compress_and_encode(os.path.basename(result_filepath))
    dict_current_results = utils.read_pkl_file(result_filepath)
    #dict_vendor_stringmatching_results = utils.read_pkl_file(utils.get_results_filename('rb-vendor', field_to_search_in, path_classification_results))
    with open(vendor_type_list_chat_gpt4_path, 'r') as f:
        dict_vendor_type = json.load(f)
        dict_vendor_type = lowercase_keys(dict_vendor_type)
    print("type_classification_nlp_model")
    success_funtion=0
    miatakes = []
    for device in devices_unique_array:
        start_time = time.time()
        if use_list:
            result_filepath = utils.get_results_filename(classifier.model.base_model_prefix+'-list-based', field_to_search_in,
                                                         path_classification_results,fields_data)
            guessed_vendor = utils.dict_vendor_stringmatching_results[device]['classified_string_matching_vendor_vector'][0][0].lower()
            if guessed_vendor in dict_vendor_type:
                types = dict_vendor_type[guessed_vendor]
            else:
                types = utils.read_csv_single_column_to_list(types_path)
            #ignore guess, use real vendor (to find our maximum)
            types = dict_vendor_type[dict_enriched_devices[device]['vendor'][0].lower()]
            if len(numpy.intersect1d([x.lower() for x in dict_enriched_devices[device]['type']] ,[x.lower() for x in types])) == 0:
                print(f"device:{device},real vendor:{dict_enriched_devices[device]['vendor'][0].lower()},real type:{dict_enriched_devices[device]['type']},types:{types}")
        if max_exp:
            vec = data_classification_nlp_model.zeroshot_classification_max_confidence(dict_enriched_devices[device],field_to_search_in,types,classifier,fields_data)
        elif search_results_number_exp:
            vec,table = data_classification_nlp_model.zeroshot_classification_field_query_balance(dict_enriched_devices[device],field_to_search_in,types,classifier,fields_data)
        else:
            #print(device)
            vec,table = data_classification_nlp_model.zeroshot_classification_field_query_balance(dict_enriched_devices[device], field_to_search_in, types, classifier,fields_data)
        endtime = time.time()
        #print(f'function labeling time: {endtime-start_time}')
        dict_table_results[device] = table
        #top10_sentences = data_classification_nlp_model.get_top10_results_device(dict_enriched_devices[device], field_to_search_in, types, classifier,threshold)
        dict_results[device] = dict()
        dict_results[device]['classified_nlp_type_vector_' + classifier.model.base_model_prefix] = vec
        dict_results[device]['type'] = dict_enriched_devices[device]['type']
        if len(types) == 1:
            dict_results[device]['classified_nlp_type_vector_' + classifier.model.base_model_prefix] = [(types[0],1.0,1.0)]
            success_funtion = success_funtion + 1
            continue
        if len(vec) > 0:
            function=vec[0][0]
        else:
            function = ''
        if function.lower() in [x.lower() for x in dict_results[device]['type']] or len([1 for x in dict_results[device]['type'] if (x.lower() in function.lower() or function.lower() in x.lower())]) > 0:
            success_funtion = success_funtion + 1
        else:
            miatakes.append(device)
            print(f'device: {device}, model says: {function}, Real function: {dict_results[device]["type"]}, OUI: {dict_enriched_devices[device][oui_field]}, hostname:{dict_enriched_devices[device][hostname_field]}')

        #if dict_results[device]['type'][0].lower() not in [x.lower() for x in types]:
        #print('real vendor is: '+str(dict_enriched_devices[device]['vendor'])+' guessed vendor is' + guessed_vendor + ' types of this vendor are:'+str(types))
        #print(device, dict_results[device]['type'])
        #print(dict_results[device]['classified_nlp_type_vector_' + classifier.model.base_model_prefix])
        index = index + 1
    print(success_funtion,len(devices_unique_array))
    utils.save_dict_results_to_file_pkl(dict_results, result_filepath)
    utils.save_dict_to_file_pkl(dict_table_results,path_pkl_classification_table)
    find_max_availble_results(dict_results)
    return dict_results

def find_max_availble_results(dict_results):
    match_count = 0

    for device, data in dict_results.items():
        # Convert type values to lowercase for case-insensitive comparison
        type_values = set(t.lower() for t in data['type'])

        # Extract the set of items in 'classified_nlp_type_vector_roberta' to lower case
        classified_types = set(item[0].lower() for item in data['classified_nlp_type_vector_roberta'])

        # Check if any type value is in the classified list
        if type_values & classified_types:  # Set intersection to check for common elements
            match_count += 1
    print(f"Number of devices with a matching type: {match_count} out of {len(dict_results)} devices")
    return match_count
def train_decision_tree(dict_devices):
    dict_class_devices = utils.read_pkl_file(path_pkl_classification_table)
    df_data=decsion_tree.convert_dict_to_df(dict_class_devices)

    # Create a dictionary to map device names to labels
    device_to_label = {key: dict_devices[key]['type'][0].lower() for key in dict_class_devices.keys()}
    for function in device_to_label.values():
        if function.lower() not in [type.lower() for type in utils.read_csv_single_column_to_list(types_path)]:
            print(function)
    # Update the 'label' column with the corresponding label for each device name
    df_data['function'] = df_data['device_name'].map(device_to_label)
    decsion_tree.train_decision_tree(df_data)
    #decsion_tree.train_decision_tree_XGBClassifier(df_data)
def category_classification_nlp_model_dynamic_fields(dict_enriched_devices,classifier,fields_tosearch):
    field_to_search_in = fields_tosearch
    categories = utils.categories_dict.keys()
    dict_results = {}
    index = 0
    result_filepath = path_classification_results + 'class_categories_results_{}_{}.pkl'.format(''.join(fields_tosearch),classifier.model.base_model_prefix)
    dict_current_results = utils.read_pkl_file(result_filepath)

    print("categories_classification_nlp_model")
    for device in tqdm(dict_enriched_devices.keys()):
        if device in dict_current_results:
            dict_results[device] = dict_current_results[device]
    for device in tqdm(dict_enriched_devices.keys()):
        if device in dict_current_results:
            continue
        print(device)
        vec = data_classification_nlp_model.zeroshot_classification_equal_fields(dict_enriched_devices[device], field_to_search_in, list(categories), classifier)
        dict_results[device] = dict()
        dict_results[device]['classified_nlp_category_vector_' + classifier.model.base_model_prefix] = vec
        dict_results[device]['category'] = [dict_enriched_devices[device]['category']]
        print(device, dict_results[device]['category'])
        print(dict_results[device]['classified_nlp_category_vector_' + classifier.model.base_model_prefix])
        index = index + 1
        utils.save_dict_to_file_pkl(dict_results, result_filepath)

def type_classification_nlp_model(dict_enriched_devices,classifier):
    field_to_search_in = [hostnames_google_field,domains_google_field]
    types = utils.read_csv_single_column_to_list(types_path)
    dict_results = {}
    index = 0
    if classifier.model.base_model_prefix == 'roberta':
        result_filepath = pkl_file_classified_type_nlp_roberta_threshold05
    elif classifier.model.base_model_prefix == 'model':
        result_filepath = pkl_file_classified_type_nlp_facebook
    else:
        print("error, we didnt save calssification nlp file..")
        return
    dict_current_results = utils.read_pkl_file(result_filepath)

    print("type_classification_nlp_model")
    for device in tqdm(dict_enriched_devices.keys()):
        if device in dict_current_results:
            dict_results[device] = dict_current_results[device]
    for device in tqdm(dict_enriched_devices.keys()):
        if device in dict_current_results:
            continue
        print(device)
        vec = data_classification_nlp_model.zeroshot_classification_equal_fields(dict_enriched_devices[device], field_to_search_in, types, classifier)
        dict_results[device] = dict()
        dict_results[device]['classified_nlp_type_vector_' + classifier.model.base_model_prefix] = vec
        dict_results[device]['type'] = dict_enriched_devices[device]['type']
        print(device, dict_results[device]['type'])
        print(dict_results[device]['classified_nlp_type_vector_' + classifier.model.base_model_prefix])
        index = index + 1
        utils.save_dict_to_file_pkl(dict_results, result_filepath)


def type_classification_nlp_model_concat_fields(dict_enriched_devices, classifier):
    field_to_search_in = [hostnames_google_field,domains_google_field]
    types = utils.read_csv_single_column_to_list(types_path)
    dict_results = {}
    index = -1
    for device in dict_enriched_devices.keys():
        index = index + 1
        if index != 0:
            continue
        vec = data_classification_nlp_model.zeroshot_classification_concat_fields(dict_enriched_devices[device], field_to_search_in, types, classifier)
        dict_results[device] = dict()
        dict_results[device]['classified_nlp_type_vector_' + classifier.model.base_model_prefix] = vec
        dict_results[device]['type'] = dict_enriched_devices[device]['type']
        print(device, dict_results[device]['type'])
        print(dict_results[device]['classified_nlp_type_vector_' + classifier.model.base_model_prefix])
    counter_true = 0
    counter_true_two = 0
    for device in dict_results:
        if len(dict_results[device]['classified_nlp_type_vector_' + classifier.model.base_model_prefix]) <= 0:
            continue
        if dict_results[device]['type'] == dict_results[device]['classified_nlp_type_vector_' + classifier.model.base_model_prefix][0][0]:
            counter_true = counter_true + 1
        if len(dict_results[device]['classified_nlp_type_vector_' + classifier.model.base_model_prefix]) <= 1:
            continue
        if dict_results[device]['type'] == dict_results[device]['classified_nlp_type_vector_' + classifier.model.base_model_prefix][0][0] \
                or dict_results[device]['type'] == dict_results[device]['classified_nlp_type_vector_' + classifier.model.base_model_prefix][1][0]:
            counter_true_two = counter_true_two + 1
    print("results based on confidence only")
    print("Succeed: ", counter_true)
    print("Succeed 0/1: ", counter_true_two)
    if classifier.model.base_model_prefix == 'roberta':
        utils.save_dict_to_file_pkl(dict_results, pkl_file_classified_type_nlp_roberta_concat)
    elif classifier.model.base_model_prefix == 'model':
        utils.save_dict_to_file_pkl(dict_results, pkl_file_classified_type_nlp_facebook_concat)
    else:
        print("error, we didnt save calssification nlp file..")

def type_vendor_name_classification_by_gpt_model(dict_devices, model=utils.gpt_turbo_model):
    fields_to_search_in = experiment_fields
    result_filepath = get_results_filename(model,fields_to_search_in,path_classification_results)
    prompt_vendor = "Given the information provided by the user and extracted info from the internet, please classify the following IoT devices into device name, Vendor, and Type,\
             Please provide a confidence score for every type/vendor. If there are several options provide only the first two with the highest confidence level. \
            When you get the device vendor, use it to predict the type based on the knowledge you have about this vendor. \
            The device name is the device's official name, such as Amazon Alexa.\
            A vendor could be the name of a related company such as TP-Link, Amazon, Google, or Samsung.\
            The type could be  Television, Plug, Hub, Router, etc.\
            A confidence score is a number between 0% - 100%\
            Please add Explainability (max. 50 words) for your choice.\
            Your answer should be in a valid JSON format, up to 3 results, each result is your classification for device name, vendor, type, confidence, and explainability \
            Dont add any further information before or after your answer"

    for device in tqdm(dict_devices.keys()):
        dict_current_results = utils.read_pkl_file(result_filepath)
        dict_vendor_stringmatching_results = utils.read_pkl_file(utils.get_results_filename('rb-vendor', [hostname_field,hostnames_google_field,domains_google_field,dns_field,tls_issuers_field,tls_issuers_google_field], path_classification_results))

        guessed_vendor = dict_vendor_stringmatching_results[device]['classified_string_matching_vendor_vector'][0][0].lower()

        results = data_calssification_LLM.classify_device_by_fields_gpt(dict_devices[device],fields_to_search_in,model=model,prompt=prompt_vendor)
        dict_current_results[device]=results
        utils.save_dict_to_file_pkl(dict_current_results,result_filepath)

def type_classification_known_vendor_by_gpt_model(dict_devices, model):
    fields_to_search_in = experiment_fields
    fields_to_search_in.append(real_vendor_field)
    result_filepath = get_results_filename(model, fields_to_search_in, path_classification_results)

    prompt_vendor = "Given the information provided by the user and extracted info from the internet, please classify the following IoT devices into device name, Vendor, and Type,\
         Please provide a confidence score for every type/vendor. If there are several options provide only the first two with the highest confidence level. \
        When you get the device vendor, use it to predict the type based on the knowledge you have about this vendor. \
        The device name is the device's official name, such as Amazon Alexa.\
        A vendor could be the name of a related company such as TP-Link, Amazon, Google, or Samsung.\
        The type could be  Television, Plug, Hub, Router, etc.\
        A confidence score is a number between 0% - 100%\
        Please add Explainability (max. 50 words) for your choice.\
        Your answer should be in a valid JSON format, up to 3 results, each result is your classification for device name, vendor, type, confidence, and explainability \
        Dont add any further information before or after your answer"

    for device in tqdm(dict_devices.keys()):
        dict_current_results = utils.read_pkl_file(result_filepath)
        results = data_calssification_LLM.classify_device_by_fields_gpt(dict_devices[device], fields_to_search_in,
                                                                        model=model,prompt=prompt_vendor)
        dict_current_results[device] = results
        utils.save_dict_to_file_pkl(dict_current_results, result_filepath)

def type_vendor_name_classification_given_list_by_gpt_model(dict_devices, model):
    fields_to_search_in = experiment_fields
    result_filepath = get_results_filename(model+'list-based', fields_to_search_in, path_classification_results)

    prompt_vendor = "Given the information provided by the user and extracted info from the internet, please classify the following IoT devices into device name, Vendor, and Type, \
        Provided information includes a dictionary of IoT vendors and the types of IoT devices they produce, use it when you classify a device.\
        Please provide a confidence score for every type/vendor. If there are several options provide only the first two with the highest confidence level. \
        When you get the device vendor, use it to predict the type based on the knowledge you have about this vendor. \
        The device name is the device's official name, such as Amazon Alexa.\
        A vendor could be the name of a related company such as TP-Link, Amazon, Google, or Samsung.\
        The type could be  Television, Plug, Hub, Router, etc.\
        A confidence score is a number between 0% - 100%\
        Please add Explainability (max. 50 words) for your choice.\
        Your answer should be in a valid JSON format, up to 3 results, each result is your classification for device name, vendor, type, confidence, and explainability \
        Dont add any further information before or after your answer"

    for device in tqdm(dict_devices.keys()):
        dict_current_results = utils.read_pkl_file(result_filepath)
        results = data_calssification_LLM.classify_device_by_fields_gpt(dict_devices[device], fields_to_search_in,
                                                                        model=model,prompt=prompt_vendor,uselist=True)
        dict_current_results[device] = results
        utils.save_dict_to_file_pkl(dict_current_results, result_filepath)

def vendor_classification_nlp_model(dict_enriched_devices, classifier):
    field_to_search_in = [dns_field,hostname_field,hostnames_google_field,domains_google_field]
    vendors = utils.read_csv_single_column_to_list(vendors_path)
    dict_results = {}
    for device in dict_enriched_devices.keys():
        vec = data_classification_nlp_model.zeroshot_classification_equal_fields(dict_enriched_devices[device], field_to_search_in, vendors, classifier)
        dict_results[device] = dict()
        dict_results[device]['classified_nlp_vendor_vector_'+ classifier.model.base_model_prefix] = vec
        dict_results[device]['vendor'] = dict_enriched_devices[device]['vendor']
        print(dict_results[device])
    if classifier.model.base_model_prefix == 'roberta':
        utils.save_dict_to_file_pkl(dict_results, pkl_file_classified_vendor_nlp_roberta)
    elif classifier.model.base_model_prefix == 'facebook':
        utils.save_dict_to_file_pkl(dict_results, pkl_file_classified_vendor_nlp_facebook)
    else:
        print("error, we didnt save calssification nlp file..")

def convert_string_to_list_in_pkl(path):
    ext_dict = data_augmentaion.build_enriched_dict()
    with open(path,'rb') as f:
        dict_devices = pickle.load(f)
    for device in dict_devices:
        dict_devices[device][real_type_field] = ext_dict[device][real_type_field]
    save_dict_to_file_pkl(dict_devices, path)
    #return dict_devices

#debug
def rename_dict_results_names():
    curr = utils.read_pkl_file(utils.pkl_file_classified_type_nlp_roberta_threshold05)
    new_dict = dict()
    for device in curr:
        splited = device.split('_')
        if 'merged_us' in device:
            new_name = splited[0] + '_merged_us_IMC19' + splited[-1]
        elif 'merged_uk' in device:
            new_name = splited[0] + '_merged_uk_IMC19' + splited[-1]
        else:
            new_name = splited[0]+'_merged_'+splited[-1]
        new_dict[new_name] = curr[device]
        print(new_name)
    utils.save_dict_to_file_pkl(new_dict,utils.pkl_file_classified_type_nlp_roberta_threshold05)
    print(new_dict)
#convert_string_to_list_in_pkl(pkl_file_classified_type_nlp_roberta_threshold)
def sample_from_dict(d, sample=3):
    keys = random.sample(list(d), sample)
    values = [d[k] for k in keys]
    return dict(zip(keys, values))
def sample_datset_to_file(ext_dict):
    for x in ext_dict:
        del ext_dict[x][domains_google_field]
        del ext_dict[x][hostnames_google_field]
        del ext_dict[x][tls_issuers_google_field]
        del ext_dict[x][user_agents_google_field]
        del ext_dict[x]['http.request.uri']
    dict_here = sample_from_dict(ext_dict)
    data_augmentaion.export_dict_to_json(utils.dataset_json_no_searches, dict_here)

#parse_pcap.merge_all_pcaps_in_folders('G:\Dropbox\Dropbox\IoT-Meyuhas\IoT_lab\pcaps\Datasets\captures_IoT-Sentinel')

#parse_pcap.merge_all_pcaps_in_folders('/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/pcaps/Datasets/captures_IoT-Sentinel')
#parse_pcap.pre_proccessing_yourthings('/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/pcaps/Datasets/yourthings/pcaps/merged_devices','/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/pcaps/Datasets/yourthings/pcaps/devices_with_ip_mac.csv')
#parse_pcap.pre_proccessing_sivanthan('/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/pcaps/Datasets/Sivanathan/')
#parse_pcap.pre_proccessing_sivanthan_win('G:\Dropbox\Dropbox\IoT-Meyuhas\IoT_lab\pcaps\Datasets\Sivanathan')
#parse_pcap.pre_proccessing_ourlab('/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/pcaps/Datasets/OurLab/')
#parse_pcap.pre_proccessing_checkpoint('/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/pcaps/Datasets/CheckPoint_private/data_for_idc.csv')
def vendor_list_aquistion_gpt():
    path_file_results = '/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/classification_results/class_results_dns.qry.namedhcp.option.hostnamex509sat.printableStringoui_vendor_gpt-4-vendor.pkl'
    dict_current_results = utils.read_pkl_file(path_file_results)
    acquired_vendors_list = []
    real_vendor_list = []
    for device in dict_current_results:
        for result in dict_current_results[device]['classified_gpt_vector'].values():
            acquired_vendors_list.append(result['vendor'].split(' ')[0])
            acquired_vendors_list.append(result['vendor'].split(',')[0])
        real_vendor_list.append(dict_current_results[device]['vendor'])
    acquired_vendors_list = numpy.unique(acquired_vendors_list).tolist()
    acquired_vendors_list = [x.lower() for x in acquired_vendors_list]
    count = 0
    unknown_vendors = []
    for vendors in real_vendor_list:
        flag = False
        for vendor in vendors:
            if vendor.lower() in acquired_vendors_list:
                count = count + 1
                flag = True
                break
        if not flag:
            if type(vendors) == type([]):
                if vendor[0] not in unknown_vendors:
                    unknown_vendors.append(vendors[0])
    real_vendor_list = [tuple(x) for x in real_vendor_list]
    real_vendor_list = numpy.unique(real_vendor_list).tolist()
    unknown_vendors = numpy.unique(unknown_vendors).tolist()
    print(f"amount of unknown vendors is {len(unknown_vendors)} unknown_vendors: {unknown_vendors}")
    print(f"amount of unique vendors is {len(real_vendor_list)} real_vendors: {real_vendor_list}")
    print(acquired_vendors_list)


def label_classification_gpt(dict_devices,label,use_list,experiment_fields,data_fields=None,use_top_google_results=False,classifier=None ,model=utils.gpt_turbo_model):
    fields_to_search_in = experiment_fields
    list_based_ext = '-list-based' if use_list else ''
    both_labels = False
    rel_list = []
    #best vendor prediction we have
    dict_vendor_stringmatching_results = utils.read_pkl_file(utils.get_results_filename(best_vendor_file,
                                                                                        [hostname_field,
                                                                                         hostnames_google_field,
                                                                                         domains_google_field,
                                                                                         dns_field,
                                                                                         tls_issuers_field,
                                                                                         tls_issuers_google_field],
                                                                                         path_classification_results,data_fields=data_fields))
    if label == 'both':
        label = 'types and vendor'
        both_labels = True
    prompt_vendor = f"Given the information provided by the user and extracted info from the internet, please classify the following IoT devices into {label}.\
             Please provide a confidence score for every result. If there are several options provide only the first three with the highest confidence level. \
            When classifing {label}s please use label from the list of {label}s provided by the user\
            A confidence score is a number between 0% - 100%\
            Please add Explainability (max. 50 words) for your choice.\
            Your answer should be in a valid JSON format, up to 3 results, each result is your classification for {label}, confidence, and explainability \
            Dont add any further information before or after your answer"
    if both_labels or (use_list == True and label == 'type'):
        prompt_vendor = prompt_vendor + f'When you classify the device {label}s, predict a type produced by the {label}s using the provided list.'
        with open(vendor_type_list_chat_gpt4_path, 'r') as f:
            dict_vendor_type = json.load(f)
            dict_vendor_type = lowercase_keys(dict_vendor_type)
        rel_list = dict_vendor_type

    if label == 'vendor' and use_list == True:
        rel_list = utils.read_csv_single_column_to_list(vendors_path)
    if use_list == True and label == 'type':
        fields_to_search_in.append('Predicted Vendor')
    if use_list == False and label == 'type':
        rel_list = utils.read_csv_single_column_to_list(utils.types_path)
    if use_top_google_results:
        fields_to_search_in.append('Google Search Results')
    result_filepath = get_encoded_file_name(get_results_filename(model + '-' + label + list_based_ext, fields_to_search_in,
                                           path_classification_results,data_fields=data_fields))
    #result_filepath = '/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/classification_results/class_results_dhcp.option.hostnamedns.qry.namehttp.user_agentx509sat.printableStringoui_vendor_gpt-4-vendor-list-based.pkl'
    for device in tqdm(dict_devices.keys()):
        if use_list == True and label == 'type':
            #guessed_vendor = dict_vendor_stringmatching_results[device]['classified_string_matching_vendor_vector'][0][0].lower()
            guessed_vendor = dict_devices[device]['vendor']
            dict_devices[device]['Predicted Vendor'] = guessed_vendor
        if use_top_google_results:
            try:
                types = rel_list[guessed_vendor]
            except:
                types = utils.read_csv_single_column_to_list(utils.types_path)
            prompt_vendor = prompt_vendor + ' provided information contains google results for queries regarding the device info (for example, query can be a domain that the device uses)'
            top_10_results =data_classification_nlp_model.get_top10_results_device(dict_devices[device],fields_to_search_in,types,classifier,0.65)
            dict_devices[device]['Google Search Results'] = top_10_results

        dict_current_results = utils.read_pkl_file(result_filepath)
        results = data_calssification_LLM.classify_device_by_fields_and_lists_gpt(dict_devices[device], fields_to_search_in, list_labels=rel_list,label=label,model=model,prompt=prompt_vendor)
        dict_current_results[device] = results
        utils.save_dict_results_to_file_pkl(dict_current_results,get_results_filename(model + '-' + label + list_based_ext, fields_to_search_in,
                                           path_classification_results,data_fields=data_fields))

def vendor_expriments_for_paper(ext_dict,experiment_fields_filter):
    #only non-enriched fields
    all_experiment_fields = [oui_field,hostname_field,dns_field,user_agents_field,tls_issuers_field]
    data_classification_string_matching_vendor(ext_dict,all_experiment_fields,data_fields=update_data_fields(experiment_fields,utils.fields_data))
    all_experiment_fields = [x for x in all_experiment_fields if not 'user_age' in x]
    all_experiment_fields = [x for x in all_experiment_fields if not '509' in x]
    data_classification_string_matching_vendor(ext_dict,all_experiment_fields,data_fields=update_data_fields(experiment_fields,utils.fields_data))
    all_experiment_fields = [x for x in all_experiment_fields if not 'oui' in x]
    data_classification_string_matching_vendor(ext_dict,all_experiment_fields,data_fields=update_data_fields(experiment_fields,utils.fields_data))

    data_classification_string_matching_vendor(ext_dict,experiment_fields_filter,data_fields=update_data_fields(experiment_fields,utils.fields_data))
    experiment_fields_filter = [x for x in experiment_fields_filter if not 'oui' in x]
    data_classification_string_matching_vendor(ext_dict,experiment_fields_filter,data_fields=update_data_fields(experiment_fields,utils.fields_data))
    experiment_fields_filter = [x for x in experiment_fields_filter if not 'user_age' in x]
    data_classification_string_matching_vendor(ext_dict,experiment_fields_filter,data_fields=update_data_fields(experiment_fields,utils.fields_data))
    experiment_fields_filter = [x for x in experiment_fields_filter if not 'tls' in x]
    data_classification_string_matching_vendor(ext_dict,experiment_fields_filter,data_fields=update_data_fields(experiment_fields,utils.fields_data))
    data_classification_oui_vendor(ext_dict)


def function_experiments_for_paper(ext_dict,experiment_fields_filter,classifier):
    #array_mistakes = ['AugustDoorbellCam_yourthings.csv', 'BelkinWeMoCrockpot_yourthings.csv', 'BelkinWeMoLink_yourthings.csv', 'Belkin_Wemo_switch_Wireless_sivanathan.csv', 'CasetaWirelessHub_yourthings.csv', 'D-LinkDoorSensor_merged_sentinel.csv', 'D-LinkHomeHub_merged_sentinel.csv', 'EdimaxPlug1101W_merged_sentinel.csv', 'HueBridge_merged_sentinel.csv', 'iHome_Wireless_sivanathan.csv', 'iKettle2_merged_sentinel.csv', 'NestGuard_yourthings.csv', 'NestProtect_yourthings.csv', 'nest-tstat_merged_uk_IMC19.csv', 'NEST_Protect_smoke_alarm_Wireless_sivanathan.csv', 'Netatmo_weather_station_Wireless_sivanathan.csv', 't-philips-hub_merged_uk_IMC19.csv', 'Renpho_Humidifier_our_lab.csv', 'sengled-hub_merged_us_IMC19.csv', 'sousvide_merged_uk_IMC19.csv', 'WeMoInsightSwitch_merged_sentinel.csv', 'WeMoSwitch2_merged_sentinel.csv', 'WeMoLink_merged_sentinel.csv', 'Withings_Smart_scale_Wireless_sivanathan.csv', 'Withings_Smart_Baby_Monitor_Wired_sivanathan.csv', 'Xiaomi_light_bulb_our_lab.csv', 'SmarterCoffee_merged_sentinel.csv', 'Insteon_Camera_Wired_sivanathan.csv', 'Withings_merged_sentinel.csv', 'Insteon_Camera_Wireless_sivanathan.csv', 'WithingsHome_yourthings.csv', 'MAXGateway_merged_sentinel.csv']

    #ext_dict = {k:v for k,v in ext_dict.items() if k in utils.devices_unique_array}

    #all_experiment_fields = [oui_field,hostname_field,dns_field,user_agents_field,tls_issuers_field]
    type_classification_nlp_model_dynamic_fields(ext_dict, classifier, experiment_fields_filter,fields_data=utils.fields_data, use_list=False)
    #type_classification_nlp_model_dynamic_fields(ext_dict, classifier, all_experiment_fields,fields_data=utils.fields_data, use_list=True)

    #data_classification_string_matching_type(ext_dict, experiment_fields_filter,use_list=True)
    #data_classification_string_matching_type(ext_dict, experiment_fields_filter, use_list=False)
    #type_classification_nlp_model_dynamic_fields(ext_dict, classifier, experiment_fields_filter,fields_data=utils.fields_data, use_list=False)
    type_classification_nlp_model_dynamic_fields(ext_dict, classifier, experiment_fields_filter,fields_data=utils.fields_data_list, use_list=True)
    #experiment_fields_filter = [x for x in experiment_fields_filter if not 'oui' in x]
    #type_classification_nlp_model_dynamic_fields(ext_dict, classifier, experiment_fields_filter,fields_data=utils.fields_data, use_list=False)
    #type_classification_nlp_model_dynamic_fields(ext_dict, classifier, experiment_fields_filter,fields_data=utils.fields_data, use_list=True)
    #experiment_fields_filter = [x for x in experiment_fields_filter if not 'user_a' in x]
    #type_classification_nlp_model_dynamic_fields(ext_dict, classifier, experiment_fields_filter,fields_data=utils.fields_data, use_list=False)
    #type_classification_nlp_model_dynamic_fields(ext_dict, classifier, experiment_fields_filter,fields_data=utils.fields_data, use_list=True)
    #experiment_fields_filter = [x for x in experiment_fields_filter if not 'tls' in x]
    #type_classification_nlp_model_dynamic_fields(ext_dict, classifier, experiment_fields_filter,fields_data=utils.fields_data, use_list=False)
    #type_classification_nlp_model_dynamic_fields(ext_dict, classifier, experiment_fields_filter,fields_data=utils.fields_data, use_list=True)


def check_vendor_function_list(ext_dict):
    ext_dict_unique = {key: ext_dict[key] for key in utils.devices_unique_array if key in ext_dict}
    # File to save the results
    results_file = 'gpt_4o_vendor_functions.json'
    # Load existing results if the file exists
    with open(results_file, 'r') as file:
        saved_results = json.load(file)

    not_existing_vendors = []
    not_existing_functions = []
    for device in ext_dict_unique.values():
        vendor_exist = False
        function_exist = False
        for vendor in device['vendor']:
            if vendor.lower() in saved_results.keys():
                vendor_exist = True
            if vendor_exist:
                for function in device['type']:
                    if function.lower() in [x.lower() for x in saved_results[vendor.lower()]['types']]:
                        function_exist = True
                break
        if not vendor_exist:
            not_existing_vendors.append(device['vendor'])
        if not function_exist:
            not_existing_functions.append((device['vendor'],device['type']))
    print(not_existing_vendors)
    print(not_existing_functions)

def main():
    #parse_pcaps()
    #data_enrichment_process()
    ext_dict = data_augmentaion.build_enriched_dict()
    #ext_dict = assign_cpes()
    #timing_expriment(ext_dict)
    #results_number_experiment_string_matching(ext_dict)
    #threshold_and_results_number_experiment(ext_dict)

    ext_dict = filter_search_results(ext_dict,max_results_dict=utils.max_results_dict)
    #calculate_baseline(ext_dict)
    experiment_filtered_fields = update_experiment_fields(experiment_fields, utils.max_results_dict)
    #check_vendor_function_list(ext_dict)
    #vendor_expriments_for_paper(ext_dict,experiment_filtered_fields)
    #data_augmentaion.export_dict_to_json(utils.to_publish_json_file_path,ext_dict)
    #run SM and Roberta with each field alone:
    #ner_experiment_half_list_known(ext_dict,experiment_filtered_fields)
    #ner_experiments_list(ext_dict,experiment_filtered_fields)
    #data_classification_string_matching_vendor(ext_dict,experiment_filtered_fields,utils.fields_data,use_NER=False)
    #run_single_field_experiment(ext_dict)
    #run_single_field_experiment(ext_dict,threshold_exp=True)
    #train_decision_tree(ext_dict)
    #data_fing_identification(ext_dict,experiment_filtered_fields)
    #data_classification_string_matching_vendor(ext_dict,experiment_filtered_fields,data_fields=update_data_fields(experiment_fields,utils.fields_data))

    #classifier = pipeline("zero-shot-classification",
    #                      model="joeddav/xlm-roberta-large-xnli", use_auth_token=True)
        #classify using string matching
    #function_experiments_for_paper(ext_dict,experiment_filtered_fields,classifier)
    #data_classification_string_matching_type(ext_dict,experiment_filtered_fields,use_list=True)
        # classify using zero-shot-classification
    #calcualte_runtime_roberta(classifier)
    #type_classification_nlp_model_dynamic_fields(ext_dict, classifier, experiment_filtered_fields,fields_data=utils.fields_data, use_list=False)

    #data_classification_oui_vendor(ext_dict)
    #zero_thr_exp

    #type_classification_nlp_model_dynamic_fields(ext_dict, classifier, experiment_filtered_fields,fields_data=utils.fields_data_zero_thr, use_list=False)

    #type_classification_nlp_model_dynamic_fields(ext_dict, classifier, experiment_filtered_fields,fields_data=utils.fields_data, use_list=False)

    #vendor_classification_nlp_model_dynamic_fields(ext_dict,classifier,experiment_filtered_fields,use_list=False)
    #type_vendor_name_classification_by_gpt_model(ext_dict,model=gpt_turbo_model)
    #type_vendor_name_classification_by_gpt_model(ext_dict,model=gpt4_model)
    #ner_frequencies.vendor_list_acquistion(ext_dict,experiment_filtered_fields)
    #vendor_list_aquistion_gpt()
    #label_classification_gpt(ext_dict,'vendor',use_list=False,model=gpt4_model,data_fields=utils.fields_data)
    #label_classification_gpt(ext_dict,'vendor',experiment_fields=utils.experiment_fields,use_list=True,data_fields=utils.fields_data,model=gpt4_model)
    label_classification_gpt(ext_dict,'type',experiment_fields=utils.experiment_fields,use_list=False,data_fields=utils.fields_data,model=gpt4_model)
    #label_classification_gpt(ext_dict,'both',use_list=True,model=gpt4_model)
    #label_classification_gpt(ext_dict, 'type', use_top_google_results=False,use_list=True,model=gpt4_model)

    #label_classification_gpt(ext_dict, 'type', use_top_google_results=True,classifier=classifier,use_list=True,model=gpt4_model)
    #type_classification_known_vendor_by_gpt_model(ext_dict,model=gpt4_model)
    #type_vendor_name_classification_given_list_by_gpt_model(ext_dict,gpt4_model)
    #type_classification_nlp_model(ext_dict,classifier)
    #category_classification_nlp_model_dynamic_fields(ext_dict,classifier,experiment_fields)

    print(1)
def run_single_field_experiment(ext_dict,threshold_exp=False):
    all_experiment_fields = [oui_field,oui_vendors_google_field,hostname_field,hostnames_google_field,domains_google_field,dns_field,user_agents_field,user_agents_google_field,tls_issuers_field,tls_issuers_google_field]

    #only non-enriched fields
    all_experiment_fields = [oui_field,hostname_field,dns_field,user_agents_field,tls_issuers_field]
    #all_experiment_fields = [oui_vendors_google_field,hostnames_google_field,domains_google_field,user_agents_google_field,tls_issuers_google_field]


    classifier = pipeline("zero-shot-classification",
                          model="joeddav/xlm-roberta-large-xnli", use_auth_token=True)
    all_experiment_fields = update_experiment_fields(all_experiment_fields, utils.max_results_dict)
    for field in all_experiment_fields:
        #classify using string matching
        #data_classification_string_matching_type(ext_dict,[field])
        #for threshold in thresholds:
            # classify using zero-shot-classification
        #ext_dict_filtered = filter_search_results(ext_dict, max_results_dict=utils.max_results_dict)
        #data_classification_string_matching_vendor(ext_dict,[field],data_fields=utils.fields_data)
        type_classification_nlp_model_dynamic_fields(ext_dict, classifier, [field], use_list=False,
                                                     fields_data=utils.fields_data, max_exp=False,
                                                     search_results_number_exp=False)


def threshold_and_results_number_experiment(ext_dict):
    all_experiment_fields = [oui_vendors_google_field,hostnames_google_field,domains_google_field,user_agents_google_field,tls_issuers_google_field]
    all_experiment_fields = [hostnames_google_field,domains_google_field]
    all_experiment_fields = [[hostnames_google_field,domains_google_field,user_agents_google_field,tls_issuers_google_field,oui_vendors_google_field]]
    classifier = pipeline("zero-shot-classification",
                          model="joeddav/xlm-roberta-large-xnli", use_auth_token=True)
    #all_experiment_fields = update_experiment_fields(all_experiment_fields, utils.max_results_dict)
    thresholds = np.linspace(0.0, 1, 10, endpoint=False)
    for threshold in thresholds:
        #for field in all_experiment_fields:
        #for amount_of_max_results in range(1,11,1):
            #for instance in field:
            #    utils.max_results_dict[instance] = amount_of_max_results
            #utils.max_results_dict[field] = amount_of_max_results
            #exp_fields = update_experiment_fields(field, max_results_dict)
            #exp_fields = update_experiment_fields([field],max_results_dict)
            #ext_dict_filtered = filter_search_results(ext_dict,max_results_dict=utils.max_results_dict)
            # classify using zero-shot-classification
        type_classification_nlp_model_dynamic_fields(ext_dict_filtered, classifier, exp_fields, use_list=False,fields_data=utils.fields_data,max_exp=False,search_results_number_exp=False)

def calc_throsholds_times(dict_devices,groups_amount=5):
    discovery_times=[]
    for device, fields in dict_devices.items():
        for field, data in fields.items():
            if any(str_field in field for str_field in utils.enriched_field_to_field_dict.values()) and field.endswith('_with_time') and len(data) > 0:
                for value in data:
                    discovery_times.append(value['discovery_time'])
    sorted_times = sorted(discovery_times)
    border_values = np.linspace(0, len(sorted_times) - 1, groups_amount + 1, dtype=int)
    # Extract the border times
    border_times = [sorted_times[i] for i in border_values]
    return border_times[1:]
def timing_expriment(ext_dict):
    all_experiment_fields = [hostnames_google_field,domains_google_field,user_agents_google_field,tls_issuers_google_field,oui_vendors_google_field]
    amount_of_parts = 50
    thresholds = calc_throsholds_times(ext_dict,amount_of_parts)
    classifier = pipeline("zero-shot-classification",
                          model="joeddav/xlm-roberta-large-xnli", use_auth_token=True)
    for thr in thresholds:
        all_experiment_fields_updated = [entry + '_timed_filtered_'+utils.timedelta_to_short_str(thr) for entry in all_experiment_fields]
        ext_dict_filtered = filter_search_results_timing(ext_dict,thr)
        # classify using zero-shot-classification
        type_classification_nlp_model_dynamic_fields(ext_dict_filtered, classifier, all_experiment_fields_updated, use_list=True,fields_data=utils.fields_data,max_exp=False,search_results_number_exp=False)

def results_number_experiment_string_matching(ext_dict):
    all_experiment_fields = [oui_vendors_google_field,hostnames_google_field,domains_google_field,user_agents_google_field,tls_issuers_google_field]
    #all_experiment_fields = [hostnames_google_field,domains_google_field]
    #all_experiment_fields = [[hostnames_google_field,domains_google_field,user_agents_google_field,tls_issuers_google_field,oui_vendors_google_field]]
    #all_experiment_fields = update_experiment_fields(all_experiment_fields, utils.max_results_dict)
    for field in all_experiment_fields:
        for amount_of_max_results in range(1,11,1):
            #for instance in field:
            #    utils.max_results_dict[instance] = amount_of_max_results
            utils.max_results_dict[field] = amount_of_max_results
            #exp_fields = update_experiment_fields(field, max_results_dict)
            exp_fields = update_experiment_fields([field],max_results_dict)
            ext_dict_filtered = filter_search_results(ext_dict,max_results_dict=utils.max_results_dict)
            data_classification_string_matching_vendor(ext_dict_filtered,exp_fields)
def calcualte_runtime_roberta(classifer):
    # 180 chars (average result)
    to_classify = 'Fitbit homepage - www.fitbit.com You Fitness, Fitness Goals, Health Fitness More like this mikigo Miki Studiomiki 259 followers - www.fitbit.com | Fitness websites, Fitbit, Get fit'
    #4 labels (average number)
    labels=['bulb', 'light', 'sensor', 'Weight Scale']
    # Start the timer
    start_time = time.time()
    #classifer(to_classify,labels)
    data_classification_RB.count_words_per_word(to_classify,['Philips'],0.95)
    # End the timer
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print(f"The runtime of the code is {elapsed_time} seconds.")

if __name__ == "__main__":

    main()