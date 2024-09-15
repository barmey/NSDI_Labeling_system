import pandas as pd
import os
import requests
from lxml import html
#from googlesearch import search
from bs4 import BeautifulSoup
import time
import math
import json
import pickle
from difflib import SequenceMatcher
import re
import numpy
from urllib.error import URLError, HTTPError
from transformers import pipeline

#classifier = pipeline("zero-shot-classification",
#                      model="joeddav/xlm-roberta-large-xnli")
#classifier = pipeline("zero-shot-classification",
#                      model="facebook/bart-large-mnli")

pkl_file_searched_data = "/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/pcaps/data-imc19/merged_pcaps_only/uk_merged/csv_find_names/devices_details_labeling/all_devices_dictionary_google.pkl"
csvs_path = 'Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/pcaps/data-imc19/merged_pcaps_only/uk_merged/csv_find_names/devices_details_labeling'
vendors_path = "/Users/barmey/Dropbox/IoT-Meyuhas/gen_pcaps/labeling_project/vendors_list.csv"
types_path = "/Users/barmey/Dropbox/IoT-Meyuhas/gen_pcaps/labeling_project/types_list.csv"

def csv_to_dict(gapminder_csv_url):
    fields = ["dhcp.option.hostname","dns.ptr.domain_name","http.user_agent","dns.qry.name","tls.handshake.extensions_server_name","x509ce.dNSName","http.request.uri","x509sat.printableString"]
    record = pd.read_csv(gapminder_csv_url)
    details = {}
    for field in fields:
        details[field] = list(record[field].unique())
        details[field] = [x for x in details[field] if type(x) == type('')]
    return details

def list_devices_files_paths(dir_path):
    list = []
    for csvFilename in os.listdir(dir_path):
        if not csvFilename.endswith(".csv"):
            continue
        list.append(dir_path+'\\'+csvFilename)
    return list


def read_devices_details():
    devices_dict = {}
    for device_file_path in list_devices_files_paths(csvs_path):
        devices_dict[os.path.basename(device_file_path)] = csv_to_dict(device_file_path)
    #print(devices_dict)
    return devices_dict

def google_search(query):
    ## Google Search query results as a Python List of URLs
    search_result_list = list(search(query, tld="co.in", num=10, stop=3, pause=3))
    if len(search_result_list) == 0:
        return 'Unknown'
    text = []
    for url in search_result_list:
        try:
            page = requests.get(url)
            time.sleep(10)
            soup = BeautifulSoup(page.content, features="lxml")
            metas = soup.find_all('meta')
            text.extend([meta.attrs['content'] for meta in metas if 'name' in meta.attrs and 'description' in meta.attrs['name']])

            if soup.title != None:
                text.append(soup.title.get_text())
        except:
            continue
    return(text)


def collect_data_hostnames_from_google():
    devices_dict = read_pkl_file(pkl_file_searched_data)
    index = 0
    for device in devices_dict.values():
        index = index + 1
        if index < 12:
            continue

        #init device
        if 'hostname_google' not in device.keys():
            device['hostname_google'] = []
        if 'domains_google' not in device.keys():
            device['domains_google'] = []
        if 'dns_google' not in device.keys():
            device['dns_google'] = []

        #search google#
#        for hostname in device['dhcp.option.hostname']:
#            while True:
#                try:
#                    # do stuff
#                    device['hostname_google'].extend(list(dict.fromkeys(google_search(hostname))))
#                except HTTPError as e:
#                    print('went to sleep')
#                    time.sleep(60)
#                    continue
#                break

#        for domain in device['dns.qry.name']:
#            while True:
#                try:
#                    # do stuff
#                    device['dns_google'].extend(list(dict.fromkeys(google_search(domain))))
#                except HTTPError as e:
#                    print('went to sleep')
#                    time.sleep(60)
#                    continue
#                break

        #unique and save
        device['hostname_google'] = numpy.unique(device['hostname_google']).tolist()
        device['domains_google'] = numpy.unique(device['domains_google']).tolist()
        device['dns_google'] = numpy.unique(device['dns_google']).tolist()
        save_dict_to_file_pkl(devices_dict)
        print(device)
        print('\n\n')

    return devices_dict

def print_devices_list():
    devices_dict = read_devices_details()
    print([x.split('_')[0] for x in devices_dict.keys()])

def read_dict_from_file(file_path):
    #read csvs files
    devices_dict = read_devices_details()
    #extract names
    devices_names = ([x.split('_')[0] for x in devices_dict.keys()])

    # reading the searched data from files - google
    with open(file_path) as f:
        lines = [json.loads(line) for line in f]
    #create a dictionary with the searched data
    dict_devices = {devices_names[i]: lines[i] for i in range(len(lines))}

    save_dict_to_file_pkl(dict_devices)

    return(dict_devices)


def save_dict_to_file_pkl(dict):
    # save it to file so we can read it later with pickle load
    f = open(pkl_file_searched_data, 'wb')
    pickle.dump(dict, f)
    f.close()

def read_pkl_file(file_path):
    with open(file_path,'rb') as f:
        dict_devices = pickle.load(f)
    return dict_devices

def read_csv_vendors_file(path):
    fields = ['source','vendor']
    record = pd.read_csv(path)
    details = {}
    for field in fields:
        details[field] = list(record[field].unique())
        details[field] = [x for x in details[field] if type(x) == type('')]
    return details

def read_csv_types_file(path):
    fields = ['source','type']
    record = pd.read_csv(path)
    details = {}
    for field in fields:
        details[field] = list(record[field].unique())
        details[field] = [x for x in details[field] if type(x) == type('')]
    return details

def analyze_json_details(types,vendors):
    dict_devices = read_pkl_file(pkl_file_searched_data)
    start = time.time()
    for device in dict_devices.keys():
        ven = find_vendor(dict_devices[device], vendors)
        typ,conf = find_type(dict_devices[device], types)
        print('Device Name:',device,', Analyzed Type:',typ,' Confidence: ',round(conf,2),', Analyzed Vendor: ',ven)
        dict_devices[device]['analyzed_vendor_RB'] = ven
        dict_devices[device]['analyzed_type_model_roberta'] = typ
    save_dict_to_file_pkl(dict_devices)
    end = time.time()
    calculate_vendor_success_stats(dict_devices)
    print('Elapsed time is',end-start, 'seconds.' )


def calculate_vendor_success_stats(dict_devices):
    count_suc = 0
    count_gen = 0
    count_suc_type = 0
    count_gen_type = 0
    for device in dict_devices.keys():
        if match_strings(dict_devices[device]['vendor'],dict_devices[device]['analyzed_vendor_RB']):
            count_suc = count_suc + 1
        else:
            print('Failed Identification! Device Name:',dict_devices[device]['vendor'],', Analyzed Vendor:',dict_devices[device]['analyzed_vendor_RB'])
        if dict_devices[device]['analyzed_vendor_RB'] != 'unknown':
            count_gen = count_gen + 1

        if match_strings(dict_devices[device]['type'],dict_devices[device]['analyzed_type_model_roberta']):
            count_suc_type = count_suc_type + 1
        else:
            print('Failed Identification! Device Name:',dict_devices[device]['type'],', Analyzed type:',dict_devices[device]['analyzed_type_model_roberta'])
        if dict_devices[device]['analyzed_type_model_roberta'] != 'unknown':
            count_gen_type = count_gen_type + 1
    print('We found a vendor for ', count_gen,' out of ',len(dict_devices), ' devices - ',round(count_gen/len(dict_devices),2))
    print('We successfully found a vendor for ', count_suc,' out of ', len(dict_devices), ' devices - ',round(count_suc/len(dict_devices),2))
    print('We found a type for ', count_gen_type,' out of ',len(dict_devices), ' devices - ',round(count_gen_type/len(dict_devices),2))
    print('We successfully found a type for ', count_suc_type,' out of ', len(dict_devices), ' devices - ',round(count_suc_type/len(dict_devices),2))
    success_rates_vendors = {'Vendor':count_suc/len(dict_devices),'Type':count_suc_type/len(dict_devices)}
    print_bars_precentage_dict(success_rates_vendors)

def analyze_results(dict_devices,types,vendors):
    types_count = {x: 0 for x in types['type']}
    types_count_suc = {x: 0 for x in types['type']}
    types_rates = {x: 0 for x in types['type']}
    for device in dict_devices.keys():
        if match_strings(dict_devices[device]['type'],dict_devices[device]['analyzed_type_model_facebook']):
            types_count[dict_devices[device]['type']] = types_count[dict_devices[device]['type']] + 1
            types_count_suc[dict_devices[device]['type']] = types_count_suc[dict_devices[device]['type']] + 1
        else:
            types_count[dict_devices[device]['type']] = types_count[dict_devices[device]['type']] + 1
            print('Failed Identification! Device Name:',device,', Analyzed type:',dict_devices[device]['analyzed_type_model_facebook'])
    print('Identification percentages by types:')
    for type_name in types_count.keys():
        types_rates[type_name] = types_count_suc[type_name]/types_count[type_name]
        print(type_name,': ',round(types_rates[type_name],3)*100,'% success rate')
    print_bars_precentage_dict(types_rates)

def print_bars_precentage_dict(dict_precentages_categorties):
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter
    dict_precentages_categorties = dict(sorted(dict_precentages_categorties.items(),key=lambda item: item[1],reverse=True))
    plt.figure(figsize=(12,3))
    plt.bar(dict_precentages_categorties.keys(),[x for x in dict_precentages_categorties.values()],color='salmon',width=0.5)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1,decimals=0))
    plt.grid(axis='y')
    plt.ylabel('Success Rate (%)')
    plt.show()
def find_type(device, types):

    sequence_domains_to_classify = ''.join(device['dns_google'])
    sequence_hostname_to_classify = ''.join(device['hostname_google'])
    candidate_labels = types['type']
    type_host = ('unknown',0)
    type_domains = ('unknown', 0)
    try:
        if sequence_domains_to_classify != '':
            type_domains = (classifier(sequence_domains_to_classify, candidate_labels)['labels'][0],
                            classifier(sequence_domains_to_classify, candidate_labels)['scores'][0])
        if sequence_hostname_to_classify != '':
            type_host = (classifier(sequence_hostname_to_classify, candidate_labels)['labels'][0],
                        classifier(sequence_hostname_to_classify, candidate_labels)['scores'][0])
    except e:
        print('Error: ', e)
    finally:
        if type_host[0] != type_domains[0]:
            if type_host[1] > type_domains[1]:
                return type_host[0],type_host[1]
            else:
                return type_domains[0],type_domains[1]
        else:
            return type_host[0],max(type_host[1],type_domains[1])


def find_vendor(device,vendors):
    vendor_field = {}
    for field in device.keys():
        if len(device[field]) > 0:
            vendor_field[field] = find_vendor_in_text(device[field],vendors)

    fdist = dict(zip(*numpy.unique(list(vendor_field.values()), return_counts=True)))
    fdist.pop('unknown',None)
    #print("The elements with their counts are -", fdist)
    #print("The most common word is -", list(fdist)[0])
    fdist = list({k: v for k, v in sorted(fdist.items(), key=lambda item: item[1])})
    if len(fdist) > 0:
        return(fdist[-1])
    else:
        return('unknown')

def remove_words(list):
    to_remove = ['net','com','org','il','cn','co','gov']
    for remove in to_remove:
        if remove in list:
            list.remove(remove)
    return list


def find_vendor_in_text(text,vendors):
    splited_text = re.split(r"[-;,.\s]\s*",' '.join(text))
    splited_text = remove_words(splited_text)
    vendors_ratio = {}
    for vendor in vendors['vendor']:
        count = 0
        for word in splited_text:
            if match_strings(vendor,word):
                count = count + 1
        vendors_ratio[vendor] = count
    max_vendor = max(vendors_ratio, key=vendors_ratio.get)
    if vendors_ratio[max_vendor] == 0:
        return 'unknown'
    else:
        return max_vendor

def match_strings(vendor,word):
    return SequenceMatcher(None, vendor.lower(), word.lower()).ratio() >= 0.8

def fill_with_vendors():
    list_vendors = ['Apple','Blink','Blink','Amazon','Amazon','Amazon','Amazon','Google','Insteon','Lightify','Magichome',
                    'Nest','Ring','Roku','Samsung','Sengled','SmartThings','Sousvide','Philips','WeMo','TP-Link','TP-Link','Wansview','Roborock']

def fill_with_types():
    list_vendors = ['TV', 'Camera', 'Hub', 'Speaker', 'Speaker', 'Speaker', 'TV', 'Speaker', 'Hub',
                    'Hub', 'Light Bulb',
                    'Thermostat', 'Doorbell', 'TV', 'TV', 'Hub', 'Hub', 'Sous Vide', 'Light Bulb', 'Plug',
                    'Light Bulb', 'Plug', 'Camera', 'Vacuum']

    dict_devices = read_pkl_file(pkl_file_searched_data)
    for device,real_vendor in zip(dict_devices.keys(),list_vendors):
        dict_devices[device]['type'] = real_vendor
    save_dict_to_file_pkl(dict_devices)
#dict = collect_data_hostnames_from_google()
#print(dict)
#print_devices_list()
#read_dict_from_file("G:\\Dropbox\\Dropbox\\IoT - Meyuhas\\IoT_lab\\pcaps\\data-imc19\\merged_pcaps_only\\uk_merged\\csv_find_names\\devices_details_labeling\\google_all_devices.json")
#read_pkl_file(pkl_file_searched_data)
#fill_with_types()
#types = read_csv_types_file(types_path)
#vendors = read_csv_vendors_file(vendors_path)
#analyze_json_details(types,vendors)
#analyze_results(read_pkl_file(pkl_file_searched_data),types,vendors)
#calculate_vendor_success_stats(read_pkl_file(pkl_file_searched_data))