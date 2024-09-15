import pickle
import random

import pandas as pd
import numpy
import re
import os
from tqdm import tqdm

import utils

pkl_file_searched_data = "/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/pcaps/data-imc19/merged_pcaps_only/uk_merged/csv_find_names/devices_details_labeling/all_devices_dictionary_google.pkl"
#pkl_file_csvs_data = "/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/pcaps/data-imc19/merged_pcaps_only/uk_merged/csv_find_names/devices_details_labeling/all_devices_csvs_data_multiple_options.pkl"
pkl_file_csvs_data = "/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/csvs/devices_csvs_imc_sentinel.pkl"
pkl_files_searches_path = "/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/google_search_files/"

csvs_path = '/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/csvs/'
vendors_path = "/Users/barmey/Dropbox/IoT-Meyuhas/gen_pcaps/labeling_project/vendors_list_updated.csv"
types_path = "/Users/barmey/Dropbox/IoT-Meyuhas/gen_pcaps/labeling_project/types_list_updated.csv"
pkl_file_gpt_results = "/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/classification_results/gpt_classification_results.pkl"
pkl_file_fing_results = "/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/classification_results/class_fing_classification_results.pkl"
pkl_file_gpt_results_ner = "/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/classification_results/ner_based_gpt_classification_results.pkl"
pkl_file_string_matching_counter = "/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/classification_results/pkl_file_string_matching_counter.pkl"

metadata_file = "/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/classification_results/metadata_filenames.pkl"
pkl_file_hostnames_search = "/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/google_search_files/hostnames_google.pkl"
pkl_file_domains_search = "/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/google_search_files/domains_google.pkl"
pkl_file_user_agents_search = "/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/google_search_files/useragents_google.pkl"
pkl_file_issuers_search = "/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/google_search_files/issuers_google.pkl"
pkl_file_oui_vendors_search = "/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/google_search_files/oui_vendors_google.pkl"
path_classification_results = "/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/classification_results/"
pkl_file_classifications_sentences = "/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/classification_results/sentences_classification.pkl"
pkl_file_classified_vendor_rb = "/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/classification_results/classified_vendor_rb.pkl"
pkl_file_classified_vendor_oui = "/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/classification_results/classified_vendor_oui.pkl"
pkl_file_classified_type_rb = "/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/classification_results/classified_type_rb.pkl"
pkl_file_classified_vendor_nlp_roberta = "/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/classification_results/classified_vendor_nlp_rob.pkl"
pkl_file_classified_type_nlp_roberta = "/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/classification_results/classified_type_nlp_rob.pkl"
pkl_file_classified_type_nlp_roberta_threshold = "/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/classification_results/classified_type_nlp_rob_thr07.pkl"
pkl_file_classified_type_nlp_roberta_threshold05 = "/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/classification_results/classified_type_nlp_rob_thr05.pkl"
pkl_file_classified_vendor_nlp_facebook = "/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/classification_results/classified_vendor_nlp_face.pkl"
pkl_file_classified_type_nlp_facebook = "/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/classification_results/classified_type_nlp_face.pkl"
pkl_file_classified_type_nlp_roberta_concat = "/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/classification_results/classified_type_nlp_rob_concat.pkl"
pkl_file_classified_type_nlp_facebook_concat = "/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/classification_results/classified_type_nlp_face_concat.pkl"
path_pkl_classification_table = "/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/classification_results/devices_classification_table_roberta.pkl"
to_publish_json_file_path = "/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/classification_results/dict_devices_domains_hostnames.json"
dataset_json_no_searches = "/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/classification_results/dict_devices_no_searches.json"
vendor_type_list_chat_gpt4_path = '/Users/barmey/Dropbox/IoT-Meyuhas/gen_pcaps/labeling_project/vendor_types_dict.JSON'
real_vendor_field = 'vendor'
real_type_field = 'type'
real_category_field = 'category'

oui_field = 'oui_vendor'
hostname_field = 'dhcp.option.hostname'
dns_field = 'dns.qry.name'
hostnames_google_field = 'hostname_google'
domains_google_field = 'domains_google'
domains_google_filtered_noisy_domains_field = 'google_domains'+'_filtered_noise'
user_agents_google_field = 'user_agents_google'
user_agents_field = 'http.user_agent'
tls_issuers_google_field = 'tls_issuer_google'
oui_vendors_google_field = 'oui_vendors_google'
tls_issuers_field = 'x509sat.printableString'
domains_filtered_field = 'dns.qry.name_filtered'

field_to_enriched_field_dict = {
    oui_field: oui_vendors_google_field,
    hostname_field: hostnames_google_field,
    dns_field: domains_google_field,
    user_agents_field: user_agents_google_field,
    tls_issuers_field: tls_issuers_google_field
}
enriched_field_to_field_dict = {v: k for k, v in field_to_enriched_field_dict.items()}

full_uri_field = 'http.request.full_uri'
fields = ["dhcp.option.hostname", "dns.ptr.domain_name", "http.user_agent", "dns.qry.name",
          "tls.handshake.extensions_server_name", "x509ce.dNSName", "http.request.full_uri", "http.request.uri", "x509sat.printableString",domains_filtered_field]
categories_dict = {
'mobile devices': ["Generic", "Mobile", "Tablet", "MP3 Player", "eBook Reader", "Smart Watch", "Wearable", "Car"],
'audio video devices' :["VoIP Device","Media Player", 'Television', "Game Console", "Streamer", "Speaker/Amp", "Speaker", "AV Receiver", "Cable Box", "Disc Player", "Audio Player", "Remote Control", "Radio", "Photo Display", "Microphone", "Projector"],
'General it devices' :["Computer", "Laptop", "Printer", "Fax", "IP Phone", "Scanner"],
'home Automation devices' :["Crockpot","Humidifier","Siren","Vacuum","Gateway","Hub","Bridge","Echo","Camera", "Plug", "Switch", "Light","Bulb", "Voice Assistant", "Thermostat", "Smart Meter", "Fridge", "Garage Door", "Sprinkler", "Doorbell", "Lock", "Touch Panel", "Controller", "Weight Scale", "Weather Station", "Baby Monitor", "Motion Detector", "Smoke Detector", "Water Sensor", "Sensor"],
'network devices' : ["Router", "NAS", "Modem", "Firewall", "VPN", "PoE Switch", "USB", "Small Cell", "UPS", "Network Appliance"],
'server devices' :["Virtual Machine", "Server", "Terminal", "Mail Server", "File Server", "Web Server", "Domain Server", "Communication", "Database"],
'Board devices' : ["Raspberry", "Arduino", "Processing", "Circuit Board", "RFID Tag"]
}
gpt_turbo_model= 'gpt-3.5-turbo'
gpt4_model = 'gpt-4'
#experiment definition: what are the fields to search in, remove comment of desired experiment
#only hostname
#experiment_fields = [hostname_field]
#only hostname and hostname google
#experiment_fields = [hostname_field,hostnames_google_field]
#experiment_fields = [hostnames_google_field]
#only domains
#experiment_fields = [dns_field]
#only domains and domains google
#experiment_fields = [dns_field,domains_google_field]
#experiment_fields = [domains_google_field]
#dhcp and dns fields without their google results
#experiment_fields = [dns_field,hostname_field]
#experiment_fields = [domains_filtered_field,hostname_field,tls_issuers_field]
#experiment_fields = [dns_field,hostname_field,tls_issuers_field]
#experiment_fields = [dns_field,hostname_field,tls_issuers_field,full_uri_field]

#experiment_fields = [dns_field,hostname_field,tls_issuers_field,oui_field]

#experiment_fields = [dns_field,domains_google_field,hostname_field,hostnames_google_field,tls_issuers_field,tls_issuers_google_field]
#experiment_fields = [dns_field,domains_google_field,hostname_field,hostnames_google_field,tls_issuers_field,tls_issuers_google_field,user_agents_field,user_agents_google_field]

#2 top fields and their correspond google (domains, dhcp, user_agent, tls issuer)
#experiment_fields = [hostname_field,hostnames_google_field,domains_google_field,dns_field]

#all 4 fields and their correspond google (domains, dhcp, user_agent, tls issuer)
#experiment_fields = [hostname_field,hostnames_google_field,domains_google_field,dns_field,user_agents_field,user_agents_google_field,tls_issuers_field,tls_issuers_google_field]
#experiment_fields = [hostname_field,hostnames_google_field,domains_google_field,dns_field,user_agents_field,user_agents_google_field]
#experiment_fields = [hostname_field,hostnames_google_field,domains_google_field,dns_field,tls_issuers_field,tls_issuers_google_field]
#experiment_fields = [hostname_field,tls_issuers_field,dns_field]
#experiment_fields = [user_agents_field]
#experiment_fields = [user_agents_field,user_agents_google_field]
#experiment_fields = [user_agents_google_field]
#experiment_fields = [tls_issuers_field]
#experiment_fields = [tls_issuers_field,tls_issuers_google_field]
#experiment_fields = [tls_issuers_google_field]
#experiment_fields = [tls_issuers_field,dns_field]

#experiment_fields = [hostname_field,hostnames_google_field,domains_google_field,dns_field,user_agents_field,user_agents_google_field,tls_issuers_field,tls_issuers_google_field,oui_field,oui_vendors_google_field]

#experiment_fields = [tls_issuers_field,dns_field]
experiment_fields = [oui_vendors_google_field, hostnames_google_field, domains_google_field,
                         user_agents_google_field, tls_issuers_google_field]
#experiment_fields = [hostname_field,dns_field,user_agents_field,tls_issuers_field,oui_field]

max_results_dict = {
    hostnames_google_field: 10,
    domains_google_field: 10,
    user_agents_google_field: 10,
    tls_issuers_google_field: 10,
    oui_vendors_google_field: 10
}
fields_data_zero_thr = {
    hostnames_google_field: {
        "threshold": 0.3,
        "weight": 0.26
    },
    domains_google_field: {
        "threshold": 0.3,
        "weight": 0.24
    },
    oui_vendors_google_field: {
        "threshold": 0.3,
        "weight": 0.19
    },
    user_agents_google_field: {
        "threshold": 0.3,
        "weight": 0.2
    },
    tls_issuers_google_field: {
        "threshold": 0.3,
        "weight": 0.1
    }
}

fields_data = {
    hostnames_google_field: {
        "threshold": 0.55,
        "weight": 0.26
    },
    domains_google_field: {
        "threshold": 0.57,
        "weight": 0.24
    },
    oui_vendors_google_field: {
        "threshold": 0.62,
        "weight": 0.19
    },
    user_agents_google_field: {
        "threshold": 0.57,
        "weight": 0.2
    },
    tls_issuers_google_field: {
        "threshold": 0.49,
        "weight": 0.1
    }
}
#optimized - for with list experiment
fields_data_list = {
    hostnames_google_field: {
        "threshold": 0.55,
        "weight": 0.33
    },
    domains_google_field: {
        "threshold": 0.44,
        "weight": 0.22
    },
    oui_vendors_google_field: {
        "threshold": 0.47,
        "weight": 0.17
    },
    user_agents_google_field: {
        "threshold": 0.44,
        "weight": 0.14
    },
    tls_issuers_google_field: {
        "threshold": 0.5,
        "weight": 0.14
    }
}

def update_data_fields(field_to_search_in, fields_data):
    return {key: fields_data[key] for key in field_to_search_in if key in fields_data}


friendly_names = {
    'oui_vendors_google': 'OUI+',
    'oui_vendor': 'OUI',
    'dhcp.option.hostname': 'Hostname',
    'dns.qry.name': 'Domains',
    'hostname_google': 'Hostname+',
    'domains_google': 'Domains+',
    'user_agents_google': 'User-agents+',
    'http.user_agent': 'User-agents',
    'tls_issuer_google': 'TLS Issuer+',
    'x509sat.printableString': 'TLS Issuer',
}
def update_experiment_fields(experiment_fields, max_results_dict):
    return [field + '_filtered_'+str(max_results_dict[field]) if field in max_results_dict.keys() else field for field in experiment_fields]

def filter_search_results(dict_devices, max_results_dict):
    import copy
    dict_devices_up = copy.deepcopy(dict_devices)
    for device in list(dict_devices_up.keys()):
        for field in list(dict_devices_up[device].keys()):
            if 'google' in field:  # assuming all the search results fields contain 'google'
                max_results = max_results_dict[field]
                filtered_results = [result[2] for result in dict_devices_up[device][field] if result[1] < max_results]
                dict_devices_up[device][field + '_filtered_'+str(max_results)] = filtered_results
    return dict_devices_up
def filter_search_results_timing(dict_devices,threshold):
    import copy
    dict_devices_up = copy.deepcopy(dict_devices)
    for device in list(dict_devices_up.keys()):
        for field in list(dict_devices_up[device].keys()):
            if 'google' in field:  # assuming all the search results fields contain 'google'
                if enriched_field_to_field_dict[field]+'_with_time' not in dict_devices[device].keys():
                    dict_devices_up[device][field + '_timed_filtered_'+timedelta_to_short_str(threshold)] = [entry[2] for entry in dict_devices[device][field]]
                    continue
                # Create a set of values from the first array with discovery time less than the threshold
                values_within_threshold = [entry['value'] for entry in dict_devices[device][enriched_field_to_field_dict[field]+'_with_time'] if entry['discovery_time'] < threshold]
                # Filter the second array based on the set
                filtered_enriched_values = [entry[2] for entry in dict_devices[device][field] if entry[0] in values_within_threshold]
                dict_devices_up[device][field + '_timed_filtered_'+timedelta_to_short_str(threshold)] = filtered_enriched_values
    return dict_devices_up
#experiment_fields = [hostname_field,hostnames_google_field,domains_google_filtered_noisy_domains_field,dns_field,user_agents_field,user_agents_google_field,tls_issuers_field,tls_issuers_google_field]

best_vendor_file = 'rb-vendor'

def select_random_device_by_vendor(devices):
    selected_devices = {}
    vendor_device_map = {}

    # Group devices by vendor
    for device_id, device_info in devices.items():
        vendor = tuple(device_info['vendor'])  # Convert list to tuple to use as a key
        if vendor not in vendor_device_map:
            vendor_device_map[vendor] = []
        vendor_device_map[vendor].append(device_id)

    # Randomly select one device per vendor
    for vendor, device_ids in vendor_device_map.items():
        selected_device_id = random.choice(device_ids)
        selected_devices[selected_device_id] = devices[selected_device_id]
    print(selected_devices.keys())
    return selected_devices.keys()
def save_dict_to_file_pkl(dict,path_pkl_to_save):
    # save it to file so we can read it later with pickle load
    f = open(path_pkl_to_save, 'wb')
    pickle.dump(dict, f)
    f.close()

def read_pkl_file(file_path):
    dict_devices = {}
    if os.path.exists(file_path):
        with open(file_path,'rb') as f:
            dict_devices = pickle.load(f)
    return dict_devices


import hashlib
import json
import os
import pickle


def generate_filename_from_hash(data_string):
    """Generate a filename using md5 hash of the data string."""
    return hashlib.md5(data_string.encode()).hexdigest()


def save_metadata(filename_hash, original_data, metadata_file_path):
    """Save a mapping from hashed filename to original data."""
    try:
        with open(metadata_file_path, 'r') as f:
            metadata = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        metadata = {}

    metadata[filename_hash] = original_data

    with open(metadata_file_path, 'w') as f:
        json.dump(metadata, f)

def get_encoded_file_name(base_path_pkl_to_save):
    filename_hash = generate_filename_from_hash(base_path_pkl_to_save)
    hashed_filepath = f"{os.path.dirname(base_path_pkl_to_save)}/{filename_hash}.pkl"
    return hashed_filepath
def save_dict_results_to_file_pkl(dict_data, base_path_pkl_to_save, metadata_file_path=metadata_file):
    """Save dictionary to a file with a hashed name and record its metadata."""
    filename_hash = generate_filename_from_hash(base_path_pkl_to_save)
    hashed_filepath = f"{os.path.dirname(base_path_pkl_to_save)}/{filename_hash}.pkl"

    with open(hashed_filepath, 'wb') as f:
        pickle.dump(dict_data, f)

    save_metadata(filename_hash, base_path_pkl_to_save, metadata_file_path)

    return hashed_filepath  # Return the actual path where the file was saved, useful for subsequent operations


def read_dict_results_file(file_path, metadata_file_path=metadata_file):
    """Read the file based on either hashed or original filename and return the original filename."""
    # Check if the file directly exists (hashed name provided)
    try:
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            # Fetch the original filename from metadata
            with open(metadata_file_path, 'r') as meta_f:
                metadata = json.load(meta_f)
                for hashed, original in metadata.items():
                    if hashed in file_path:
                        return data, original
            return data, None
        else:
            return {}, None
    except:
        print(file_path)
        return {}, None



def get_all_field_from_devices_dict(dict_devices,field):
    hostnames = []
    for device in dict_devices.keys():
        if field in dict_devices[device].keys() and dict_devices[device][field] != [None]:
            hostnames.extend(dict_devices[device][field])
    return numpy.unique(hostnames).tolist()

import zlib
import base64

def valid_external_domain(domain):
    global_domains = ['google.com','facebook.com','qq.com','example.com']
    if any(item in domain for item in global_domains):
        return False
    parts = domain.split('.')
    if 'arpa' in parts or 'in-addr' in parts or 'wpad' in parts or domain.endswith('.lab'):
        return False
    return True

def valid_issuer(issuer):
    not_issuer = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA",
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY","Alaska", "Alabama", "Arkansas", "American Samoa", "Arizona", "California", "Colorado",
                   "Connecticut", "District ", "of Columbia", "Delaware", "Florida", "Georgia", "Guam", "Hawaii",
                   "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts",
                   "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina",
                   "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York",
                   "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Puerto Rico", "Rhode Island", "South Carolina",
                   "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Virgin Islands", "Vermont", "Washington",
                   "Wisconsin", "West Virginia", "Wyoming",'Ontario','Paris']
    if any(item in issuer for item in not_issuer):
        return False
    return True
def valid_user_agent(user_agent):
    not_user_agents = ['index','include','html','echo','%','=','#',',,','<script>','whatever']
    if any(item in user_agent for item in not_user_agents):
        return False
    return True

def read_json(path):
    # Load and standardize the final list of IoT manufacturers from the JSON file
    with open(path, 'r') as file:
        dict_json = json.load(file)
    return dict_json

def read_csv_single_column_to_list(path):
    record = pd.read_csv(path)
    list_vendors = [i[0] for i in record.values.tolist()]
    return list_vendors





def decode_fields_data(encoded_string):
    field_sections = encoded_string.split("__")

    data = {}
    for section in field_sections:
        parts = section.split("_")
        field_name = parts[0]
        threshold = float(parts[2][1:])
        weight = float(parts[4][1:])

        data[field_name] = {
            "threshold": threshold,
            "weight": weight
        }
    return data
def encode_fields_data(fields_data):
    name_parts = []
    for field, data in fields_data.items():
        name_parts.append(f"{field}_t{data['threshold']}_w{data['weight']}")
    return "__".join(name_parts)
def get_results_filename(modelname,fields_to_search,path_dir,data_fields):
    encoded_data = encode_fields_data(data_fields)
    return path_dir + 'class_results_{}_{}_{}.pkl'.format(''.join(fields_to_search),encoded_data,modelname)

def timedelta_to_short_str(td):
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Include days in hours if present
    if td.days > 0:
        hours += td.days * 24

    return f"{hours:02}_{minutes:02}_{seconds:02}"
def extract_info_from_filename(filename):
    # Extract the directory and the base name
    # Regular expression pattern to extract field data sections
    pattern = r"([\w]+)_t([\d\.]+)_w([\d\.]+)"

    path_dir, basename = os.path.split(filename)
    path_dir += '/'  # Ensure there's a trailing slash

    # Split the basename using double underscores to separate fields_to_search from the encoded data
    parts = basename.split("__")

    # Concatenate all parts up until the last one
    encoded_data_str = "__".join(parts[:-1])

    matches = re.findall(pattern, filename)

    # Assuming all fields have both threshold and weight
    data_fields = {}

    fields_curr = fields.copy()
    # fields_curr.append('vendor')
    fields_curr.append(domains_google_field)
    fields_curr.append(hostnames_google_field)
    fields_curr.append(tls_issuers_google_field)
    fields_curr.append(user_agents_google_field)
    fields_curr.append(domains_google_filtered_noisy_domains_field)
    fields_curr.append('Predicted Vendor')
    fields_curr.append('Google Search Results')
    fields_curr.append(oui_vendors_google_field)
    fields_to_search = []


    for match in matches:
        field_tmp, threshold, weight = match
        for field in fields_curr:
            if field in field_tmp:
                data_fields[field] = {"threshold": float(threshold), "weight": float(weight)}


    # Extract fields_to_search
    field_numbers = {}
    for field in fields_curr:
        # Create a pattern with the field name and look for it in fields_and_filter_str
        pattern = field + r"_filtered_([0-9]+)"
        match = re.search(pattern, filename)
        if match:
            fields_to_search.append(field)
            field_numbers[field] = int(match.group(1))

    for field in utils.field_to_enriched_field_dict.keys():
        if field == oui_field and (field + '_') not in filename:
            continue
        if (field) in filename and field not in fields_to_search:
            fields_to_search.append(field)

    # Extract the model name
    modelname = parts[-1].split("_")[-1].split(".")[0]

    pattern = r"timed_filtered_(\d+_\d{2}_\d{2})"
    match = re.search(pattern, filename)
    if match:
        # Convert the time format from 'HH_MM_SS' to 'HH:MM:SS'
        time_thershold=match.group(1).replace('_', ':')
    else:
        time_thershold='0'

    return modelname, fields_to_search, field_numbers, data_fields,time_thershold

def relative_luminance(rgb):
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def lowercase_keys(input_dict):
    if isinstance(input_dict, dict):
        return {k.lower(): lowercase_keys(v) for k, v in input_dict.items()}
    elif isinstance(input_dict, list):
        return [lowercase_keys(element) for element in input_dict]
    else:
        return input_dict


from urllib.parse import urlparse
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = numpy.exp(x - numpy.max(x))
    return e_x / e_x.sum()


def smart_shorten_url(url, end_length):
    parsed = urlparse(url)

    # Extract the FQDN
    fqdn = parsed.netloc

    # Extract the remainder of the URL after the FQDN
    remainder = parsed.path + parsed.params + parsed.query + parsed.fragment

    # Create the shortened URL
    if len(remainder) > end_length:
        return fqdn + "..." + remainder[-end_length:]
    else:
        return fqdn + remainder

#defined as a domain that presents in more than a single device type
noisy_domains_across_types = ['nat.xbcs.net', 'api.xbcs.net', 'fw.xbcs.net', 'time.stdtime.gov.tw', 'fs.xbcs.net', 'time.nest.com', 'frontdoor.nest.com', 'wpad.unctrl.dada.lab', 'wpad.dada.lab', 'spectrum.s3.amazonaws.com', 'arcus-uswest.amazon.com', 'dcape-na.amazon.com', 'dp-gw-na.amazon.com', 'msh.amazon.com', 'device-messaging-na.amazon.com', 'todo-ta-g7g.amazon.com', 'api.amazonalexa.com', 'device-metrics-us.amazon.com', 'wl.amazon-dss.com', 'unagi-na.amazon.com', 'prod.amazoncrl.com', 'd3h5bk8iotgjvw.cloudfront.net', 'wpad.unctrl.moniotr.lab', 'wpad.moniotr.lab', 'd3gjecg2uu2eaq.cloudfront.net', 'gw', 'analytics.localytics.com', 'xbcs.net', 'api.samsungosp.com', 'mtalk4.google.com', 'devices.xbcs.net', 'alt7-mtalk.google.com', 'tunnel.xbcs.net', 'alt4-mtalk.google.com', 'alt5-mtalk.google.com', 'msh.amazon.co.uk', 'www.googleapis.com', 'redirector.gvt1.com', 'cdn.ravenjs.com', 'www.google-analytics.com', 'pubsub.pubnub.com', 'time1.google.com', 'v2.broker.lifx.co', 's3-eu-west-1.amazonaws.com', 'devs.tplinkcloud.com', 'use1-api.tplinkra.com', 'client-api.itunes.apple.com', 'mesu.apple.com', 'ocsp.digicert.com', 'cs9.wac.phicdn.net', 'pd.itunes.apple.com', 'radio.itunes.apple.com', 'e673.dsce9.akamaiedge.net', 'pancake.apple.com', 'xp.itunes-apple.com.akadns.net', 'init.ess.apple.com', 'time-ios.apple.com', 'init.itunes.apple.com', 'gsa.apple.com', 'gdmf.apple.com', 'xp.apple.com', 'guzzoni.apple.com', 'ocsp.apple.com', 'init-p01md.apple.com', 'www.icloud.com', 'play.itunes.apple.com', 'setup.icloud.com', 'bookkeeper.itunes.apple.com', 'upp.itunes.apple.com', 'init-p01st.push.apple.com', '20-courier.push.apple.com', 'e4478.a.akamaiedge.net', 'mesu.g.aaplimg.com', '13-courier.push.apple.com', 'www-cdn.icloud.com.akadns.net', 'play.googleapis.com', 'dp-rsm-prod.amazon.com', 'softwareupdates.amazon.com', 'drive.amazonaws.com', 'ntp-g7g.amazon.com', 'kindle-time.amazon.com', '_companion-link._tcp.local', '_homekit._tcp.local', 'gateway.icloud.com', '33-courier.push.apple.com', 'time-ios.g.aaplimg.com', '37-courier.push.apple.com', 'time-b.nist.gov', 'euw1-api.tplinkra.com', 'euw1-events.tplinkra.com', 'deventry.tplinkcloud.com', 'msmetrics.ws.sonos.com', 'us-auth2.samsungosp.com', 'alt3-mtalk.google.com', 'alt8-mtalk.google.com', 'alt1-mtalk.google.com', 'api.mixpanel.com', 'decide.mixpanel.com', 'graph.facebook.com', 'sstream.flipboard.com', 'settings.crashlytics.com', 's-usc1c-nss-117.firebaseio.com', 'login.jsp', 'index.php', 'perl.exe', 'Help.action', 'www.facebook.com', '_http._tcp.local', 'us.ot.io.mi.com', 'download.tplinkcloud.com', 'events.tplinkra.com', 'connect.insteon.com', 'de.ot.io.mi.com', '0.ubuntu.pool.ntp.org', 'i.ytimg.com', 'itunes.apple.com.edgekey.net', 'iphonesubmissions.apple.com', '_raop._tcp.local', 'e673.e9.akamaiedge.net', 'gspe35-ssl.ls.apple.com', 'gsp64-ssl.ls.apple.com', 'gspe1-ssl.ls.apple.com', 'keyvalueservice.icloud.com', 'cl2.apple.com', 'configuration.apple.com.edgekey.net', 'sr.symcd.com', 'ocsp-ds.ws.symantec.com.edgekey.net', 'origin.guzzoni-apple.com.akadns.net', '18-courier.push.apple.com', 'clientflow.apple.com', 'configuration.apple.com', 'gsp-ssl.ls.apple.com', 'gsp64-ssl.ls-apple.com.akadns.net', 'g.symcd.com', 'clientflow.apple.com.edgekey.net', 'apple.com', 'ocsp.entrust.net', '31-courier.push.apple.com', '7-courier.push.apple.com', 'gspe21-ssl.ls.apple.com', '12-courier.push.apple.com', '34-courier.push.apple.com', 'www.apple.com', '21-courier.push.apple.com', '41-courier.push.apple.com', '2-courier.push.apple.com', 'pancake.g.aaplimg.com', 'sync.itunes.apple.com', '9-courier.push.apple.com', '5-courier.push.apple.com', '1-courier.push.apple.com', 'Astra Lagos._companion-link._tcp.local', 's2.symcb.com', 'gsas.apple.com', 'profile.ess.apple.com', 'static.ess.apple.com', 'identity.ess.apple.com', 'BD09082C-F0E9-54D4-9A9F-37FA985F91C4._homekit._tcp.local', 'world-gen.g.aaplimg.com', '43217C19-3360-5E41-8A8C-5A3E676A9E97._homekit._tcp.local', '11-courier.push.apple.com', '40-courier.push.apple.com', '45-courier.push.apple.com', 'cl2-cdn.origin-apple.com.akadns.net', 'C397334C-BEB7-5AF5-83BC-8E9A8C6E5836._homekit._tcp.local', '43-courier.push.apple.com', '8-courier.push.apple.com', 'push.apple.com', '14-courier.push.apple.com', '25-courier.push.apple.com', '15-courier.push.apple.com', '20.courier-push-apple.com.akadns.net', '27-courier.push.apple.com', '46-courier.push.apple.com', '22-courier.push.apple.com', '49-courier.push.apple.com', 'etc', 'login', 'perl', 'idcplg', 'about', 'admincp', 'surgeweb', 'gs-loc.apple.com', 'googleads.g.doubleclick.net', 'adservice.google.com', 'www.googletagmanager.com', 'www.googleadservices.com', 'cdn.awsusor0.fds.api.mi-img.com', 'scalews.withings.net', 'safebrowsing.googleapis.com', 'xmpp.withings.net']
#defined as a domain that presents in more than a single device type and vendor (both critireas must be met)
#noisy_domains_across_types_and_vendors = ['wpad.unctrl.dada.lab', 'wpad.dada.lab', 'wpad.unctrl.moniotr.lab', 'wpad.moniotr.lab', 'alt5-mtalk.google.com', 'www.googleapis.com', 'redirector.gvt1.com', 'cdn.ravenjs.com', 'www.google-analytics.com', 'pubsub.pubnub.com', 'time1.google.com', 'v2.broker.lifx.co', 's3-eu-west-1.amazonaws.com', 'play.googleapis.com', 'time-b.nist.gov', 'msmetrics.ws.sonos.com', 'us-auth2.samsungosp.com', 'alt1-mtalk.google.com', 'settings.crashlytics.com', 'www.facebook.com', '_http._tcp.local', 'us.ot.io.mi.com', 'de.ot.io.mi.com', '0.ubuntu.pool.ntp.org', 'i.ytimg.com', 'gs-loc.apple.com', 'googleads.g.doubleclick.net', 'adservice.google.com', 'www.googletagmanager.com', 'www.googleadservices.com', 'cdn.awsusor0.fds.api.mi-img.com', 'safebrowsing.googleapis.com']

devices_dict_unique = {
    'Aria': ['Aria_merged_sentinel.csv'],
    #'AppleEcho': ['Apple_Echo_Wireless_sivanathan.csv','AppleHomePod_yourthings.csv'],
    'AppleTV': ['AppleTV(4thGen)_yourthings.csv', 'appletv_merged_uk_IMC19.csv', 'appletv_merged_us_IMC19.csv'],
    'AugustDoorbellCam': ['AugustDoorbellCam_yourthings.csv'],
    'BelkinNetcam': ['BelkinNetcam_yourthings.csv'],
    'BelkinWeMoCrockpot': ['BelkinWeMoCrockpot_yourthings.csv'],
    'BelkinWeMoLink': ['BelkinWeMoLink_yourthings.csv'],
    'BelkinWeMoMotionSensor': ['BelkinWeMoMotionSensor_yourthings.csv', 'Belkin_wemo_motion_sensor_Wireless_sivanathan.csv'],
    'BelkinWeMoSwitch': ['BelkinWeMoSwitch_yourthings.csv','Belkin_Wemo_switch_Wireless_sivanathan.csv'],
    'BlinkCamera': ['blink-camera_merged_us_IMC19.csv', 'blink-camera_merged_uk_IMC19.csv'],
    'BlinkSecurityHub': ['blink-security-hub_merged_us_IMC19.csv', 'blink-security-hub_merged_uk_IMC19.csv'],
    'BoseSoundTouch10': ['BoseSoundTouch10_yourthings.csv'],
    'Canary': ['Canary_yourthings.csv'],
    'CasetaWirelessHub': ['CasetaWirelessHub_yourthings.csv'],
    'ChineseWebcam': ['ChineseWebcam_yourthings.csv'],
    'D-LinkCam': ['D-LinkCam_merged_sentinel.csv','D-LinkDCS-5009LCamera_yourthings.csv','D-LinkDayCam_merged_sentinel.csv'],
    'D-LinkDoorSensor': ['D-LinkDoorSensor_merged_sentinel.csv'],
    'D-LinkHomeHub': ['D-LinkHomeHub_merged_sentinel.csv'],
    'D-LinkSiren': ['D-LinkSiren_merged_sentinel.csv'],
    'D-LinkSwitch': ['D-LinkSwitch_merged_sentinel.csv'],
    'D-LinkSensor': ['D-LinkSensor_merged_sentinel.csv'],
    'D-LinkWaterSensor': ['D-LinkWaterSensor_merged_sentinel.csv'],
    'Dropcam': ['Dropcam_Wireless_sivanathan.csv'],
    'EdimaxCam': ['EdimaxCam1_merged_sentinel.csv', 'EdimaxCam2_merged_sentinel.csv'],
    'EdimaxPlug': ['EdimaxPlug2101W_merged_sentinel.csv', 'EdimaxPlug1101W_merged_sentinel.csv'],
    'EdnetGateway': ['EdnetGateway_merged_sentinel.csv'],
    'EchoPlus':['echoplus_merged_uk_IMC19.csv','echoplus_merged_us_IMC19.csv'],
    'EchoSpot':['echospot_merged_uk_IMC19.csv','echospot_merged_us_IMC19.csv'],
    'EchoDot':['echodot_merged_uk_IMC19.csv','echodot_merged_us_IMC19.csv'],
    'Echo':['Amazon_Echo_Wireless_sivanathan.csv','AmazonEcho_our_lab.csv','AmazonEchoGen1_yourthings.csv'],
    'FireTV': ['firetv_merged_us_IMC19.csv', 'firetv_merged_uk_IMC19.csv', 'AmazonFireTV_yourthings.csv'],
    'GoogleHome': ['GoogleHome_yourthings.csv','GoogleHomeMini_yourthings.csv', 'google-home-mini_merged_uk_IMC19.csv', 'google-home-mini_merged_us_IMC19.csv', 'Google_Home_mini_our_lab.csv','Google_SmartSpeaker_our_lab.csv'],
    'HarmonKardonInvoke': ['HarmonKardonInvoke_yourthings.csv'],
    'HP Printer': ['HP_Printer_Wireless_sivanathan.csv'],
    'HueBridge': ['HueBridge_merged_sentinel.csv'],
    'InsteonHub': ['insteon-hub_merged_us_IMC19.csv', 'insteon-hub_merged_uk_IMC19.csv', 'InsteonHub_yourthings.csv'],
    'iHome': ['iHome_Wireless_sivanathan.csv'],
    'iKettle2': ['iKettle2_merged_sentinel.csv'],
    'KoogeekPlug': ['Koogeek_plug_our_lab.csv'],
    'LightBulbsLiFXSmartBulb': ['Light_Bulbs_LiFX_Smart_Bulb_Wireless_sivanathan.csv','Lifx_bulb_our_lab.csv','LIFXVirtualBulb_yourthings.csv'],
    'Lightify': ['Lightify_merged_sentinel.csv', 'lightify-hub_merged_uk_IMC19.csv', 'lightify-hub_merged_us_IMC19.csv'],
    'MagicHomeStrip': ['magichome-strip_merged_us_IMC19.csv', 'magichome-strip_merged_uk_IMC19.csv'],
    'MiCasaVerdeVeraLite': ['MiCasaVerdeVeraLite_yourthings.csv'],
    'nVidiaShield': ['nVidiaShield_yourthings.csv'],
    'NestCamIQ': ['NestCamIQ_yourthings.csv'],
    'NestDropcam': ['Nest_Dropcam_Wireless_sivanathan.csv'],
    'NestGuard': ['NestGuard_yourthings.csv'],
    'NestProtect': ['NestProtect_yourthings.csv'],
    'NestThermostat': ['nest-tstat_merged_uk_IMC19.csv', 'nest-tstat_merged_us_IMC19.csv'],
    'NESTProtectSmokeAlarm': ['NEST_Protect_smoke_alarm_Wireless_sivanathan.csv'],
    'NetatmoWelcome': ['Netatmo_Welcome_Wireless_sivanathan.csv'],
    'NetatmoWeatherStation': ['Netatmo_weather_station_Wireless_sivanathan.csv'],
    'NetgearArloCamera': ['NetgearArloCamera_yourthings.csv'],
    'PhilipsHUEHub': ['PhilipsHUEHub_yourthings.csv', 't-philips-hub_merged_us_IMC19.csv', 't-philips-hub_merged_uk_IMC19.csv'],
    'PiperCam': ['PiperCam_our_lab.csv'],
    'PiperNV': ['PiperNV_yourthings.csv'],
    'PIX-STARPhotoFrame': ['PIX-STAR_Photo-frame_Wireless_sivanathan.csv'],
    'RenphoHumidifier': ['Renpho_Humidifier_our_lab.csv'],
    'RingDoorbell': ['RingDoorbell_yourthings.csv', 'ring-doorbell_merged_us_IMC19.csv', 'ring-doorbell_merged_uk_IMC19.csv'],
    'Roomba': ['Roomba_yourthings.csv'],
    'Roku': ['Roku4_yourthings.csv', 'RokuTV_yourthings.csv', 'roku-tv_merged_us_IMC19.csv', 'roku-tv_merged_uk_IMC19.csv'],
    'SamsungSmartCam': ['Samsung_SmartCam_Wireless_sivanathan.csv','Samsung_smart_camera_our_lab.csv'],
    'SamsungSmartTV': ['samsungtv-wired_merged_uk_IMC19.csv', 'samsungtv-wired_merged_us_IMC19.csv', 'SamsungSmartTV_yourthings.csv'],
    'SecurifiAlmond': ['SecurifiAlmond_yourthings.csv'],
    'SengledHub': ['sengled-hub_merged_uk_IMC19.csv', 'sengled-hub_merged_us_IMC19.csv'],
    'SmartThingsHub': ['Smart_Things_Wired_sivanathan.csv', 'smartthings-hub_merged_us_IMC19.csv', 'smartthings-hub_merged_uk_IMC19.csv', 'SamsungSmartThingsHub_yourthings.csv'],
    'Sonos': ['Sonos_yourthings.csv'],
    'SousVide': ['sousvide_merged_us_IMC19.csv', 'sousvide_merged_uk_IMC19.csv'],
    'SwitcherPlug': ['Switcher_plug_our_lab.csv'],
    'TP-LinkBulb': ['tplink-bulb_merged_uk_IMC19.csv', 'tplink-bulb_merged_us_IMC19.csv','TP-LinkSmartWiFiLEDBulb_yourthings.csv'],
    'TP-LinkCamera': ['TP-Link_Day_Night_Cloud_camera_Wireless_sivanathan.csv'],
    'TP-LinkPlug': ['tplink-plug_merged_us_IMC19.csv', 'tplink-plug_merged_uk_IMC19.csv', 'TP-LinkWiFiPlug_yourthings.csv', 'TP_Link_plug_our_lab.csv', 'TP_Link_plug_HS100_our_lab.csv', 'TP-LinkPlugHS100_merged_sentinel.csv', 'TP-LinkPlugHS110_merged_sentinel.csv','TP_Link_plug_our_lab.csv', 'TP-Link_Smart_plug_Wireless_sivanathan.csv'],
    't-wemo-plug':['t-wemo-plug_merged_uk_IMC19.csv','t-wemo-plug_merged_us_IMC19.csv'],
    'WansView':['wansview-cam-wired_merged_uk_IMC19.csv','wansview-cam-wired_merged_us_IMC19.csv'],
    'WeMoInsightSwitch': ['WeMoInsightSwitch2_merged_sentinel.csv', 'WeMoInsightSwitch_merged_sentinel.csv'],
    'WeMoSwitch': ['WeMoSwitch2_merged_sentinel.csv', 'WeMoSwitch_merged_sentinel.csv'],
    'WeMoLink':['WeMoLink_merged_sentinel.csv'],
    'WinkHub': ['Wink2Hub_yourthings.csv', 'WinkHub_yourthings.csv'],
    'Withings_scale': ['Withings_Smart_scale_Wireless_sivanathan.csv'],
    'Withings_cam':['Withings_Smart_Baby_Monitor_Wired_sivanathan.csv'],
    'Withings_sleep':['Withings_Aura_smart_sleep_sensor_Wireless_sivanathan.csv'],
    'WyzeIPCam': ['Wyze_IPCam_our_lab.csv'],
    'XiaomiCleaner': ['xiaomi-cleaner_merged_us_IMC19.csv', 'xiaomi-cleaner_merged_uk_IMC19.csv'],
    'XiaomiHub': ['xiaomi-hub_merged_uk_IMC19.csv', 'xiaomi-hub_merged_us_IMC19.csv'],
    'XiaomiImilabCamera': ['Xiaomi_imilab_camera_our_lab.csv'],
    'XiaomiLightBulb': ['Xiaomi_light_bulb_our_lab.csv'],
    'XiaoyiYICamera': ['Xiaoyi_YI_camera_our_lab.csv','yi-camera_merged_uk_IMC19.csv', 'yi-camera_merged_us_IMC19.csv'],
    'TribySpeaker': ['Triby_Speaker_Wireless_sivanathan.csv'],
    'SmarterCoffee': ['SmarterCoffee_merged_sentinel.csv'],
    'InsteonCameraWired': ['Insteon_Camera_Wired_sivanathan.csv'],
    'NestCam': ['NestCam_our_lab.csv'],
    'WithingsMerged': ['Withings_merged_sentinel.csv'],
    'KoogeekLightbulb': ['KoogeekLightbulb_yourthings.csv'],
    'InsteonCameraWireless': ['Insteon_Camera_Wireless_sivanathan.csv'],
    'LogitechHarmonyHub': ['LogitechHarmonyHub_yourthings.csv'],
    'BlipcareBloodPressureMeter': ['Blipcare_Blood_Pressure_meter_Wireless_sivanathan.csv'],
    'WithingsHome': ['WithingsHome_yourthings.csv'],
    'MAXGateway': ['MAXGateway_merged_sentinel.csv'],
    'ChamberlainmyQGarageOpener': ['ChamberlainmyQGarageOpener_yourthings.csv']
}
devices_unique_array = ['Aria_merged_sentinel.csv', 'appletv_merged_us_IMC19.csv', 'AugustDoorbellCam_yourthings.csv', 'BelkinNetcam_yourthings.csv', 'BelkinWeMoCrockpot_yourthings.csv', 'BelkinWeMoLink_yourthings.csv', 'Belkin_wemo_motion_sensor_Wireless_sivanathan.csv', 'Belkin_Wemo_switch_Wireless_sivanathan.csv', 'blink-camera_merged_us_IMC19.csv', 'blink-security-hub_merged_uk_IMC19.csv', 'BoseSoundTouch10_yourthings.csv', 'Canary_yourthings.csv', 'CasetaWirelessHub_yourthings.csv', 'D-LinkDayCam_merged_sentinel.csv', 'D-LinkDoorSensor_merged_sentinel.csv', 'D-LinkHomeHub_merged_sentinel.csv', 'D-LinkSiren_merged_sentinel.csv', 'D-LinkSwitch_merged_sentinel.csv', 'D-LinkSensor_merged_sentinel.csv', 'D-LinkWaterSensor_merged_sentinel.csv', 'Dropcam_Wireless_sivanathan.csv', 'EdimaxCam1_merged_sentinel.csv', 'EdimaxPlug1101W_merged_sentinel.csv', 'EdnetGateway_merged_sentinel.csv', 'echoplus_merged_uk_IMC19.csv', 'echospot_merged_us_IMC19.csv', 'echodot_merged_us_IMC19.csv', 'AmazonEchoGen1_yourthings.csv', 'AmazonFireTV_yourthings.csv', 'GoogleHome_yourthings.csv', 'HarmonKardonInvoke_yourthings.csv', 'HP_Printer_Wireless_sivanathan.csv', 'HueBridge_merged_sentinel.csv', 'insteon-hub_merged_us_IMC19.csv', 'iHome_Wireless_sivanathan.csv', 'iKettle2_merged_sentinel.csv', 'LIFXVirtualBulb_yourthings.csv', 'Lightify_merged_sentinel.csv', 'magichome-strip_merged_us_IMC19.csv', 'MiCasaVerdeVeraLite_yourthings.csv', 'nVidiaShield_yourthings.csv', 'NestCamIQ_yourthings.csv', 'Nest_Dropcam_Wireless_sivanathan.csv', 'NestGuard_yourthings.csv', 'NestProtect_yourthings.csv', 'nest-tstat_merged_uk_IMC19.csv', 'NEST_Protect_smoke_alarm_Wireless_sivanathan.csv', 'Netatmo_Welcome_Wireless_sivanathan.csv', 'Netatmo_weather_station_Wireless_sivanathan.csv', 'NetgearArloCamera_yourthings.csv', 't-philips-hub_merged_uk_IMC19.csv', 'PiperCam_our_lab.csv', 'PiperNV_yourthings.csv', 'PIX-STAR_Photo-frame_Wireless_sivanathan.csv', 'Renpho_Humidifier_our_lab.csv', 'RingDoorbell_yourthings.csv', 'Roomba_yourthings.csv', 'roku-tv_merged_uk_IMC19.csv', 'Samsung_smart_camera_our_lab.csv', 'SamsungSmartTV_yourthings.csv', 'SecurifiAlmond_yourthings.csv', 'sengled-hub_merged_us_IMC19.csv', 'smartthings-hub_merged_uk_IMC19.csv', 'Sonos_yourthings.csv', 'sousvide_merged_uk_IMC19.csv', 'Switcher_plug_our_lab.csv', 'tplink-bulb_merged_us_IMC19.csv', 'TP-Link_Day_Night_Cloud_camera_Wireless_sivanathan.csv', 'TP-LinkWiFiPlug_yourthings.csv', 't-wemo-plug_merged_us_IMC19.csv', 'wansview-cam-wired_merged_us_IMC19.csv', 'WeMoInsightSwitch_merged_sentinel.csv', 'WeMoSwitch2_merged_sentinel.csv', 'WeMoLink_merged_sentinel.csv', 'Wink2Hub_yourthings.csv', 'Withings_Smart_scale_Wireless_sivanathan.csv', 'Withings_Smart_Baby_Monitor_Wired_sivanathan.csv', 'Withings_Aura_smart_sleep_sensor_Wireless_sivanathan.csv', 'Wyze_IPCam_our_lab.csv', 'xiaomi-cleaner_merged_uk_IMC19.csv', 'xiaomi-hub_merged_us_IMC19.csv', 'Xiaomi_imilab_camera_our_lab.csv', 'Xiaomi_light_bulb_our_lab.csv', 'Xiaoyi_YI_camera_our_lab.csv', 'Triby_Speaker_Wireless_sivanathan.csv', 'SmarterCoffee_merged_sentinel.csv', 'Insteon_Camera_Wired_sivanathan.csv', 'NestCam_our_lab.csv', 'Withings_merged_sentinel.csv', 'KoogeekLightbulb_yourthings.csv', 'Insteon_Camera_Wireless_sivanathan.csv', 'LogitechHarmonyHub_yourthings.csv', 'Blipcare_Blood_Pressure_meter_Wireless_sivanathan.csv', 'WithingsHome_yourthings.csv', 'MAXGateway_merged_sentinel.csv', 'ChamberlainmyQGarageOpener_yourthings.csv']
acquired_by_gpt_vendors_list = ['*.iot.us-west-2.amazonaws', '*akamaiedge.net', '.net', '0.asia.pool.ntp.org', '0.cn.pool.ntp.org', '0.europe.pool.ntp.org', '0.openwrt.pool.ntp.org', '0.rhel.pool.ntp.org', '1.asia.pool.ntp.org', '1.cn.pool.ntp.org', '1.europe.pool.ntp.org', '100-us-scproxy.alibaba.com.gds', '192.168.137.8', '1932', '1e100.net', '1stream', '2', '2-spyware.com', '2.2.404000.0', '2.android.pool.ntp.org', '2.asia.pool.ntp.org', '2.cn.pool.ntp.org', '2.europe.pool.ntp.org', '2.nettime.pool.ntp.org', '25-courier.push.apple.com', '3.asia.pool.ntp.org', '3.cn.pool.ntp.org', '360se', '3bays', '43-courier.push.apple.com', '4thcorporation.com', '51cto博客', '51degrees', '791840461.r.cdn77.net', '79423.analytics.edgesuite.net', '798-luk-731.mktoresp.com', '7986417.log.optimizely.com', '7layerstudio.typepad.com', '7news.com-ii.co', '8', '@amazon', '_companion-link._tcp.local', 'a', 'a+', 'a+e', 'a/b', 'a104-91-77-229.deploy.static.akamaitechnologies.com', 'a1108.da1.akamai.net', 'a184-26-132-27.deploy.static.akamaitechnologies.com', 'a24rq1e5m4mtei.iot.us-west-2.amazonaws.com', 'a2r1ssw73fw43.iot.us-east-1.amazonaws.com', 'a2uowfjvhio0fa.iot.us.', 'aapks', 'aapl', 'aapl.o', 'aaplimg.com', 'aaron', 'aax-eu.amazon-adsystem.com', 'aax.amazon-adsystem.com', 'abbott', 'abbyy', 'abc', 'aboriginal', 'abuseipdb', 'acadianaracing.com', 'acapella', 'accenture', 'accessify', 'account.xiaomi.com', 'accountauthenticator', 'accounts-api-xbcs-net-1266788097.us-east-1', 'accuweather', 'acm', 'acme', 'act', 'activeperl', 'activestate', 'activite::client', 'actualite', 'acura', 'ad-free', 'ad.doubleclick.net', 'ad2pi', 'ada', 'adam', 'adaway', 'adblock', 'adcp-vpc.ap-southeast-6.aliyuncs.com', 'addigy', 'addthis.com', 'adguard', 'adguardfilters', 'adguardteam', 'admincp', 'adnxs.com', 'adobe', 'adobe.com', 'adobe.io', 'ads.yahoo.com', 'adsense', 'adserver-us.adtech.advertising.com', 'adsrvr.org', 'adt', 'advertising.com', 'aeon', 'afp', 'africa.pool.ntp.org', 'agi.amazon.com', 'agoodm.wapa.taobao.com', 'ags', 'ags-ext.amazon.com', 'ai-asm-api.wyzecam.com', 'ai-thinker', 'aidc', 'aidc.apple.com', 'aiedge.net', 'aircheq.netatmo.com', 'airplay', 'airport', 'airprint', 'ais', 'aiv-cdn.net', 'aiv-delivery.net', 'ajb413/pubnub-functions-mock', 'akadns', 'akadns.net', 'akamai', 'akamai-as', 'akamaiedge.net', 'akamaighost', 'akbooer', 'akrithi', 'aladdin', 'alamy', 'alb-openapi-share.eu-central-1.aliyuncs.com.gds', 'albert', 'alex', 'alexa', 'alexa,', 'alibaba', 'alibaba-cn-net', 'alibaba.com', 'alienvault', 'aliyun', 'all', 'allegro-software-webclient', 'allegro-software-webclient/5.40b1', 'almond', 'alphabet', 'altui/veraloginaction.php', 'amal', 'amarillo.logs.roku.com', 'amazon', 'amazon-02', 'amazon-adsystem.com', 'amazon-aes', 'amazon.co.uk', 'amazon.com', 'amazon.com,', 'amazon.com:', 'amazon.in', 'amazon.jobs', 'amazon.sg', 'amazonfiretv', 'amazonfiretv.txt', 'amazonian', 'amazonsmile', 'amazonvideo.com', 'ambrussum', 'amcs-tachyon.com', 'amdc.alipay.com', 'amdc.m.taobao.com', 'american', 'amiga', 'amir', 'amir2016@vera.com.uy', 'amoeba-plus.web.roku.com', 'ampak', 'ampproject.org', 'ams', 'amzdigital-a.akamaihd.net', 'amzdigitaldownloads.edgesuite.net', 'amzn/selling-partner-api-docs', 'amzn1.as-ct.v1.', 'amzn1.comms.id.person.amzn1~amzn1.account.my_amazon_id', 'analytics.kraken.com', 'analytics.localytics.com', 'analyticsmarket', 'ancestral', 'ancestrydna', 'anderson', 'andr-785f3ec7eb-cbc62794911ff31b-6275da66d5254dff56-2400322.eu.api.amazonvideo.com', 'andrew', 'android', 'android.googleapis.com', 'angeloc/htpdate', 'annapurna', 'antarctica.pool.ntp.org', 'antix-forum', 'antonio', 'any.run', 'anycast', 'anycodings', 'anytime', 'aol', 'ap', 'apache', 'apache-httpclient', 'apertis', 'apevec', 'apex', 'api', 'api-eu.netflix.com', 'api-global.netflix.com', 'api-latam.netflix.com', 'api-oauth-us.xiaoyi.com', 'api-push-us.xiaoyi.com', 'api-secure.solvemedia.com', 'api-user.netflix.com', 'api.ad.xiaomi.com', 'api.amazonvideo.com', 'api.io.mi.com', 'api.netatmo.net', 'api.netflix.com', 'api.radioparadise.com', 'api.roku.com', 'api.segment.io', 'api.solvvy.com', 'api.sr.roku.com', 'api.us-east-1.aiv-delivery.net', 'api.us.xiaoyi.com', 'api.wyzecam.com', 'api.xbcs.net', 'api.xiaoyi.com.tw', 'api.xwemo.com', 'api2.iheart', 'api2.iheart.com', 'api2.sr.roku.com', 'apicache.vudu.com', 'apis::firebaseremoteconfigv1::firebaseremoteconfigservice', 'apkmirror', 'apns', 'apollon77/iobroker.alexa2', 'app', 'app-measurement.com', 'app.segment.io', 'app4cdn.moovitapp.com', 'app5.moovitapp.com', 'appboot.netflix.com', 'appbrain', 'appcensus.io', 'appcenter.ms', 'appldnld.apple.com.akadns.net', 'apple', "apple's", 'apple-austin', 'apple-dns', 'apple-dns.net', 'apple-engineering', 'apple-tv.local', 'apple.com', 'apple_team_id', 'applecache', 'applecoremedia', 'appleid.apple.com', 'appletv', 'appletv3', 'appletv3,1', 'appletv5', 'appletv5,3', 'appletv5.3', 'applewatch', 'applewebkit', 'application', 'applovin', 'appnexus', 'apps.mios.com', 'apps.mzstatic.com', 'apps.mzstatic.com.mwcname.com', 'appsamurai', 'appsflyer', 'appstore', 'aptoide', 'aquaforte', 'aquatone', 'arch.pool.ntp.org', 'archer', 'arcol', 'arcus-uswest.amazon.com', 'arduino', 'arena', 'arlo', 'arnold', 'around', 'arpi', 'arrayent.com', 'arrow', 'arthouse', 'arthur', 'aruba', 'arxiv', 'ashburn', 'ashburn,', 'asia.pool.ntp.org', 'ask', 'aspect', 'aspen', 'astra', 'astracleaners', 'astralagos', 'astrodienst', 'asus', 'at', 'atheros', 'atid', 'atlanta', 'atlas', 'atomtime', 'atsec', 'atv', 'atv-ext.amazon.com', 'atvproxy', 'audid-api.taobao.com', 'audio_mpeg', 'august', 'ausweisapp2', 'authenticode', 'authorized', 'automated', 'automotive', 'autoparts', 'autovera', 'av-centerd', 'avahi', 'avahi-daemon', 'avahi-resolve', 'available', 'avast', 'avg', 'aviary', 'avira', 'avs', 'avs-alexa-14-na.amazon.compi', 'aws', 'aws-iot.wyzecam.com', 'ax.init.itunes.apple.com.edgesuite.net', 'ayde', 'azure', 'azureware', 'azurewave', 'b', 'b.scorecardresearch.com', 'babygearlab', 'bad', 'baddie', 'bag.itunes.apple.com', 'bagae', 'baidu', 'baltimorebanner-the-baltimore-banner-sandbox.web.arc-cdn.net', 'banco', 'bandwidth', 'bank', 'bard', 'barr', 'barracuda', 'bash.ws', 'basketapi', 'basketball-reference.com', 'bat.bing.com', 'baylor', 'bbb', 'bbc.com', 'beacons.gvt2.com', 'beijing', 'belkin', 'berkeley', 'berkshire', 'berto', 'bertoni', 'best', 'better', 'bhavana', 'bidswitch', 'big', 'bigtreetech', 'binance', 'bind', 'bing', 'bing.com', 'bird', 'bitdefender', 'black', 'blackstone', 'blb', 'bleacher', 'blink', 'blocking', 'blocklist', 'blocklists/appstore.txt', 'blood', 'bluelithium', 'bob-dispatch-prod-eu.amazon.com', 'bol.com', 'bombora', 'bonjour', 'boo', 'book', 'bookdown', 'booking.com', 'bookkeeper.itunes.apple.com', 'bootstrap', 'bootstrapcdn', 'bose', 'bose.io', 'bose.vtuner.com', 'boss', 'boston', 'boto3', 'boulder', 'boulder,', 'bountysource', 'brainworks', 'branch.io', 'branditechture', 'braze', 'brettm', 'brillano', 'brilliant', 'broadbandnow', 'broadcast', 'broadcom', 'brooklyn', 'brother', 'bso', 'bugcrowd', 'bugsense', 'bugsfighter', 'buildbase', 'busybox', 'busybox-ntp', 'buy.itunes.apple.com', 'c-al2o3', 'c.amazon-adsystem.com', 'c4', 'c714', 'c99.nl', 'cab', 'cablelabs', 'cacá', 'cafesperl.at', 'cai', 'caldav.icloud.com', 'calebcall/netatmo-influxdb', 'calendarsync-pa.clients6.google.com', 'california', 'callpod', 'calm', 'cambridge', 'cameron', 'canadian', 'canary', 'canary.dc-na02-useast1.connect.smartthings.com', 'candemir', 'candid.technology', 'candor', 'candum', 'canon', 'canopus', 'canvas', 'canwatch', 'capabilities', 'capi-lux', 'capital', 'capterra', 'captive', 'captive.apple.com', 'caratteristiche', 'carbonfootprint.com', 'carematix', 'carpenters', 'casetext', 'castr', 'caséta', 'catchpoint', 'cbs', 'cbsi.live.ott.irdeto.com', 'cbsig', 'cbsplaylistserver.aws.syncbak.com', 'cbsservice.aws.syncbak.com', 'cc3100', 'cc3200', 'cc4all', 'cc4skype', 'cdc', 'cde-g7g.amazon.com', 'cde-ta-g7g.amazon.com', 'cdm', 'cdn', 'cdn-apple.com', 'cdn-profiles.tunein.com', 'cdn-profiles.tunein.com.cdn.cloudflare.net', 'cdn-settings-segment.com', 'cdn.ampproject.org', 'cdn.shopify.com', 'cdnjs', 'cdns-content.dzcdn.net', 'cdnt-proxy-e.dzcdn.net', 'cdtech', 'cdws.eu-west-1.amazonaws.com', 'cdws.us-east-1.amazonaws.com', 'cedexis', 'celtic', 'census.gov', 'centos', 'centurylink', 'ceramic', 'certbot-dns-google', 'certificate', 'certification', 'certplus', 'certs', 'cetecom', 'cfnetwork', 'cfnetworkagent', 'chamberlain', 'chand', 'changelog', 'channel', 'charles', 'charlotte', 'chase.com', 'cheat.exe', 'check', 'checkip.dyndns.org', 'checkphish', 'checktls.com', 'chenega', 'chicago', 'chico', "chili's", 'china', 'choco', 'chocolatey', 'choice', 'chris', 'christofle', 'chrome', 'chrome.mdns', 'chromebook', 'chromecast', 'chromecasts', 'chunghwa', 'cialde', 'cibc', 'cisco', 'citrix', 'city', 'cl1-cdn.origin-apple.com.akadns.net', 'cl2-cdn.origin-apple.com.akadns.net', 'cl2-cn.apple.com', 'cl2.apple.com', 'cl2.apple.com.edgekey.net.globalredir.akadns.net', 'cl3.apple.com', 'cl4-cn.apple.com', 'cl5.apple.com', 'cl5.apple.com.edgekey.net', 'clare', 'clark', 'clasp', 'classifying', 'clc', 'clean.io', 'clearesult', 'clearstream', 'clepsydra.dec.com/clepsydra.labs.hp.com/clepsydra.hpl.hp.com', 'cleveland', 'clickthink', 'client', 'client-api.itunes.apple.com', 'clientflow', 'clientservices.googleapis.com', 'cling', 'clock.fmt.he.net', 'cloud', 'cloud-to-cloud', 'cloudbasic', 'cloudberry', 'cloudflare', 'cloudflare,', 'cloudflare.cloudflare-dns.com', 'cloudflare.net', 'cloudflarenet', 'cloudfront', 'cloudfront.net', 'cloudservices.roku.com', 'cloudshark.org', 'club', 'clustering', 'cmdts.ksmobile.com', 'cmybabee', 'cn-dc1.uber.com', 'cn.ntp.org.cn', 'cn.pool.ntp.org', 'cnbc', 'cnet', 'cnn', 'cnnmoney.com', 'cnpbagwell', 'co', 'cochenilles', 'codeberg', 'cognito-identity.us-east-1.amazonaws.com', 'cognito-idp.us-east-1.amazonaws.com', 'colasoft', 'collins', 'coloressence', 'com.amazon.dee.app', 'com.ants360.yicamera.international', 'com.apple', 'com.apple.trustd/2.0', 'com.appsflyer.sender', 'com.sap.prd.mobile.ios.mios', 'com.squareup.okhttp', 'com.squareup.okhttp3', 'comcast', "comcast's", 'comed', 'comedy', 'comfort', 'comics', 'commencement', 'commercial', 'commission', 'communications', 'comodo', 'comodoca', 'companies', 'companion', 'companionlink', 'compro', 'comptia', 'computer', 'comscore', 'confection.io', 'configsvc.cs.roku.com', 'configuration', 'configuration.apple.com.akadns.net', 'configuration.apple.com.edgekey.net', 'configuring', 'connect-sessionvoc', 'connect.facebook.net', 'connectivitycheck.android.com', 'connectivitycheck.gstatic.com', 'connectivitymanager', 'connman', 'connor', 'conrad', 'consiglio', 'consumer', 'consumeraffairs.com', 'content', 'content-na.drive.amazonaws.com', 'content.api.bose.io', 'continuum', 'contrast', 'control', 'control-m', 'control.kochava.com', 'control4', 'controller', 'conviva', 'cookiex.ngd.yahoo.com', 'cooper', 'coredns-mdns', 'cornell', 'cornerstone', 'corpus', 'cortana', 'counter', 'courier', 'courier.push.apple.com', 'cox', 'cpanel', 'cracked.io', 'crashlytics', 'crc', 'creamsource', 'create', 'creativeanvil.com', 'cree', 'crestron', 'criminalz.org', 'criteo', 'crl.apple.com', 'crossfit', 'crunchbase', 'crunchyroll', 'cs', 'csc', 'csdn', 'csdn博客', 'csi.gstatic.com', 'cspserver.net', 'ctg', 'cti', 'ctv', 'cult', 'cups', 'curl', 'curl.haxx.se', 'custom', 'customer-feedbacks.web.roku.com', 'cuteftp', 'cve', 'cyber', 'cybergarage', 'cybergarage-http', 'cybergarage-upnp', 'cyberlink', 'cybernews', 'cybersecurity', 'cyware', 'cz15y20kg2.execute-api.us-east-1.amazonaws.com', 'd-link', "d-link's", 'd-trust', 'd.turn.com', 'd1fk93tz4plczb.cloudfront.net', 'd1n00d49gkbray.cloudfront.net', 'd1s31zyz7dcc2d.cloudfront.net', 'd21m0ezw6fosyw.cloudfront.net', 'd3h5bk8iotgjvw.cloudfront.net', 'd3p8zr0ffa9t17.cloudfront.net', 'd63r8xi1zu867.cloudfront.net.piholeblocklist', 'dacor.net', 'dain', 'dallas', 'dalvik', 'daniel', 'dart', 'darwin', 'dashboardadvisoryd', 'dashkiosk', 'dast', 'data', 'data.api.bose.io', 'databricks', 'dataflair', 'datex', 'daventry', 'david', 'dc-eu01-euwest1', 'dc-na02-useast1.connect.-', 'dc-na02-useast1.connect.smartthings.com', 'dcape-na.amazon', 'dcp.cpp.philips.com', 'dcp.dc1.philips.com', 'dcs-5030l', 'dcs-930l', 'dd-wrt', 'de.pool.ntp.org', 'debian', 'debian.pool.ntp.org', 'dec', 'dec/compaq/hp', 'deets', 'deezen', 'deezer', 'deezer.com', 'deeztek', 'defy', 'deledao', 'deliverect', 'dell', 'demdex.net', 'demio', 'denver', 'department', 'dericam', 'design&innovation', 'destinypedia', 'destinypedia,', 'det-g7g.amazon.com', 'det-ta-g7g.amazon.com', 'detroit', 'dev', 'dev.java', 'devcentral', 'developer.amazon.com', 'device', 'device-login.lutron.com', 'device-messaging-na.amazon.com', 'device-metrics-us', 'device-metrics-us-2.amazon.com', 'device-metrics-us.amazon.com', 'device-metrics-us.amazon.com.device-metrics-us.amazon.com', 'device-metrics-us.amazon.comamazon', 'device...amazon.com', 'devicemessaging.us-east-1.amazon.com', 'devimages.apple.com.edgekey.net', 'devs.tplinkcloud.com', 'dga', 'dictionary.com', 'didier', 'digicert', 'digicert,', 'digicert.com', 'digieffects', 'digitalinx', 'dir-850l', 'directv', 'disc-prod.iot.irobotapi.com', 'disconnected.io', 'discovery', 'disney', 'disney+', 'disqus', 'diss', 'diversion/skynet', 'diversion/skynet:', 'django', 'dlink', 'dlink.com', 'dlna', 'dlnadoc', 'dls-udc.dqa.samsung.com', 'dls.di.atlas.samsung.com', 'dlx.addthis.com', 'dna', 'dngupload.turner.com', 'dns', 'dns-sd', 'dnsbl', 'dnsmasq', 'dnssec', "doc's", 'docker', 'docslib', 'doj.me', 'dollarshaveclub.com', 'domain', 'domain.com', 'domain.glass', 'domain.org', 'domain:', 'domainwatch', 'dommer', 'domoticz', 'don', 'dong', 'doordash', 'doppio+', 'dorita980', 'dot', 'dot.net', 'doubleclick', 'doubleclick.net', 'doubleverify', 'doubleverify.com', 'doulci', 'dow', 'downdetector', 'download', 'dp-discovery-na-ext', 'dp-discovery-na-ext.amazon.com', 'dp-gw-na.amazon.com', 'dp-gw.amazon.com', 'dp-rsm-prod.amazon.com', 'dpd', 'dpm.demdex.net', 'dpws', 'dr', 'dragon', 'dribbble', 'drogueria', 'dropbox', 'dropbox.com', 'dropcam', 'dropcam™', 'dsa596', 'dslreports', 'dualstack.iotmoonraker-u-elb-1w8qnw1336zq-1186348092.us-west-2.elb.amazonaws.com', 'dudek', 'dyn', 'dynaimage.cdn.turner.com', 'dzcdn.net', 'e&s', 'e-cdn-content.deezer.com', 'e-cdn-images.deezer.com', 'e11290.dspg.akamaiedge.net', 'e14.ultipro.com', 'e1410.x.akamaiedge.net', 'e1412.usps.gov', 'e14128.a.akamaiedge.net', 'e14868.dsce9.akamaiedge.net', 'e14k.com', 'e15.cz', 'e15.ultipro.com', 'e28622', 'e4478.a.akamaiedge.net', 'e5153.e9.akamaiedge.net', 'e5977.dsce9.akamaiedge.net', 'e673.dsce9.akamaiedge.net', 'e673.e9.akamaiedge.net', 'e673.g.akamaiedge.net', 'e6858.dsce9.akamaiedge.net', 'e6987.a.akamaiedge.net', 'e6987.e9.akamaiedge.net', 'e8218.dscb1.akamaiedge.net', 'e88p2lpstz[.]com', 'eai', 'earl', 'easy', 'easypost-files.s3-us-west-2.amazonaws.com', 'eatbrain', 'ebay', 'ebestpurchase', 'ecampus:', 'echo', 'echofon', 'eclipse', 'ecobee', 'ecp', 'ecx.images-amazon.com', 'edgecast', 'edgekey.net', 'edgesuite.net', 'edgewood', 'edimax', 'ediview', 'edmodoandroid', 'eds', 'educative.io', 'educba', 'ee', 'eehelp.com', 'eero', 'eggplant', 'egloo', 'elb-status-us.statuspage.io', 'eleanor', 'electric', 'electricbrain', 'electronic', 'eliminate', 'elizabeth', 'elv', 'elv-/', 'elv/eq-3', 'em.lf1925.com', 'email', 'emailsentry', 'emailveritas', 'emby', 'eml-al00', 'enabling', 'encrypted-tbn0.gstatic.com', 'end.scorecardresearch.com', 'endpoint', 'engadget', 'enjoyshop777', 'enterprise', 'entries', 'entrust', 'entrust,', 'epa', 'epubread.com', 'epubreader', 'eq-3', 'equifax', 'eric', 'ericsson', 'erie', 'erwbgy/pdns', 'eset', 'esp32-cam', 'espn', 'espn+', 'espressif', 'essalud', 'essentials!', 'esszimmer', 'esxi', 'et', 'eternal', 'ethereum', 'eu-', 'eu.api.amazonvideo.com', 'euapi.getpiper.com', 'eudc-sep02.notes.mckinsey.com', 'eulen.com', 'europe.pool.ntp.org', 'eurosport.com', 'ev3dev/connman', 'events.api.bosecm.com', 'events.gfe.nvidia.com', 'everesttech.net', 'evomaster', 'evrythng', 'evrythng.js', 'ex6120', 'exploit-db', 'express', 'external', 'external-mrs2-1.xx.fbcdn.net', 'ey', 'eyeota', 'ezlo', 'f-02h', 'f-droid', 'f.w.', 'f2.netatmo.net', 'f5', 'fabric', 'fabrik', 'facebook', 'facebook,', 'facetime', 'factory', 'factory-reset.com', 'fadell', 'falcon', 'faleemi', 'fandango', 'fandangonow', 'fanduel', 'fantasy', 'faq', 'fashiontv', 'fast', 'fastclick.com.edgesuite.net', 'fastclick.net', 'fastgetsoftware.com', 'fastly', 'fastly,', 'fastream', 'faststream', 'fbbruteforce', 'fbcdn.net', 'fca', 'fcc', 'federal', 'fedora', 'ffmpeg', 'ffmpeg/apichanges', 'fiddler', 'fing.io', 'fingerbank', 'fints-g7g.amazon.com', 'fire', 'fire-', 'firebase', 'firebaseinstallations.googleapis.com', 'firebaselogging-pa.googleapis.com', 'firebaselogging.googleapis.com', 'firebaseremoteconfig.googleapis.com', 'firefox', 'fireoscaptiveportal.com', 'fireserve', 'firesticks', 'firetv', 'firewalla', 'firs-g7g.amazon.com', 'firs-ta-g7g.amazon.com', 'first', 'fishersci.com', 'fivethirtyeight', 'flashtalking', 'flex', 'flic', 'flipboard', 'fls-eu.amazon.com', 'fluxtream.atlassian.net', 'fmstream', 'fmstream.org', 'font', 'fonts.gstatic.com', 'food', 'forcetlssm', 'ford', 'forest', 'forgotten', 'fortinet', 'fortiproxy', 'fortisiem', 'fortnite', 'fourier.taobao.com', 'fox', "fox's", 'foxtel.com.au', 'fr-register.xmpush.global.xiaomi.com', 'fr.api.xmpush.global.xiaomi.com', 'fr.pool.ntp.org', 'fractioncalculatorplusfree', 'frank', 'frank-comp-na.amazon.com', 'fraternitas', 'free', 'freecodecamp', 'freelance', 'freertos', 'frnog', 'ftp', 'ftvr-na.amazon.com', 'fuboplayer', 'fujifilm', 'fulfillment', 'funimation', 'futomi/node-dns-sd', 'fw.xbcs.net', 'g', 'g-ecx.images-amazon.com', 'g.aaplimg.com', 'g2', 'gac', 'gainspan', 'gaithersburg', 'galleon', 'gameflycdn.com', 'games', 'gamestream', 'gateway.fe.apple-dns.net', 'gbd', 'gdmf.apple.com', 'gearlab', 'gecko', 'geddy', 'geeksforgeeks', 'geekzone', 'geforce', 'geller-pa.googleapis.com', 'gembur', 'gen', 'general', 'genesys', 'genie', 'gentoo', 'geocerts', 'geomobileservices-pa.googleapis.com', 'george', 'geotrust', 'germany', 'getapp', 'getgo', 'getgo,', 'gethpinfo.com', 'getorder', 'getpiper.com', 'getting', 'getty', 'gfnapi.developer.nvidiagrid.net', 'ggpht.com', 'gicert', 'gimme', 'gists', 'gitbook', 'gitee.com', 'github', 'github.com', 'github.com/qkzsky/galaxy-fds', 'github.com/unknownfallen/xi', 'github.com/xiaomi', 'github.com/xiaomi/galaxy-fds-sdk-golang', 'gitlab', 'global', 'globalbrandeshop.com', 'globalsign', 'glpals.com', 'gmail', 'gmedia', 'gnu', 'go', 'go-cbor', 'go.solvvy.com', 'goal.com', 'gobtron', 'gocart-web-prod-*.elb.amazonaws.com', 'godaddy.com', 'godaddy.com,', 'godiva', 'golan', 'goldborough', 'golden', 'good', 'goodheart-willcox', 'googe', 'google', 'google-cloud-platform', 'google.apis.admin.directory.directory_v1', 'google/files/b0f3e76e.0', 'google::apis::androidmanagementv1', 'googleads.g.doubleclick.net', 'googleadservices.com', 'googleapis.com', 'googlecast', 'googlesyndication', 'googleusercontent.com', 'gov.uk', 'gpcapt', 'gpm.samsungqbe.com', 'gpstracker', 'gradle', 'graduate', 'graham', 'graph-na02-useast.api.smartthings.com', 'graph-na02-useast1.api.smartthings.com', 'grclark', 'greater', 'greatfire', 'greg', 'gregmillerphoto.com', 'grifco', 'gruenwald', 'grupo', 'gs-loc.apple.com', 'gs908e', 'gsa', 'gsa.apple.com', 'gsas', 'gsas.apple.com', 'gsas.apple.com.akadns.net', 'gsas.apple.com/grandslam/gsservice2/postdata', 'gspe1-ssl.ls.apple.com', 'gspe19-ssl.ls.apple.com', 'gspe21-ssl.ls.apple.com', 'gspe35-ssl.ls.apple.com', 'gstatic', 'gstatic.com', 'gts', 'guam', 'guardicore', 'guides', 'guru3d', 'guru99', 'gustavo', 'guzzoni-apple-com.v.aaplimg.com', 'guzzoni.apple.com', 'gv-dispatch', 'gvt2.com', 'gw', 'gwu', 'gyeonggi', 'h.', 'h.,', 'h.d.', 'h9utzk4f.execute-api.us-east-1.amazonaws.com', 'ha', 'hack', 'hacker', 'hackerattackvector', 'hackertoolkit', 'hackmag', 'haier', 'hardreset.info', 'harley', 'harman', 'harmony', 'harpyeagle', 'hashicorp', 'haveibeenexpired', 'havis', 'hawaii', 'hawking', 'haxx', 'hbo.com', 'hd', 'hd>library>application', 'hdiotcamera', 'head-fi', 'headers', 'health', 'heartbeat.xwemo.com', 'heat', 'heejoung', 'hehe', 'heidi', 'hello', 'help', 'help.origin', 'hewlett', 'hewlett-packard', 'hipaa', 'hl7.fhir.us.udap-security', 'hmrc', 'hobbes', 'hoh999', 'hollyman', 'holy', 'home', 'home.getvera.com', 'homeassistant', 'homebridge', 'homekit', 'homematic', 'homepage', 'homepod', 'homeseer', 'host-tools', 'hostname', 'hostname:', 'hot', 'hotel', 'hotnewhiphop', 'houston', 'how', 'how-to', 'howtoremove.guide', 'hp', 'hplip', 'hs100', 'hs110', 'hs200', 'htmltvui.netflix.com', 'htpdate_1.2.0-1_arm64.deb', 'http', 'http://device-login.lutron.com', 'http://pd.itunes.apple.com.mwcname.com', 'http://whatismyip.akamai.com/', 'http://www.pool.ntp.org/zone/north-america', 'httpbrowser', 'httpclient', 'httpd_request_handler', 'https', 'https://alexa.amazon.es', 'https://buy.itunes.apple.com/verifyreceipt', 'https://buy.itunes.apple.com/webobjects/mzfinance.woa/wa/associatevppuserwithitsaccount?cc=au&amp%3b', 'https://itunes.apple.com/us/app/ansonia-pd/id1299883731?ls=1&mt=8', 'https://play.google.com/store/apps/details?id=com.citizenobserver.ansoniapd', 'https://static.gc.apple.com/public-key/gc-prod-4.cer', 'https://static.gc.apple.com/public-key/gc-prod-6.cer', 'https://tinytts-eu-west-1.amazon.com/3/89203f66-58ec-xxxx-xxxx-xxxxxxx-95b8e6/14/', 'https://tinytts.amazon.com/', 'https://tinytts.amazon.com/amazon', 'https://tinytts.amazon.com/path', "hu's", 'huade-teacher.oss-cn-beijing.aliyuncs.com', 'huawei', 'hubitat', 'hudong-baike-dev.oss-cn-hangzhou.aliyuncs.com.gds', 'hue', 'hulu', 'humanitariannet', 'huntsman', 'hurricane', 'hwaseong', 'hybrid', 'hyjoy', 'hyperice', 'hypestat', 'i-config', 'i-stream.pl', 'i.iheart.com', 'i.instagram.com', 'i.scdn.co', 'i.ytimg.com', 'i2-iiowhrcheqobnprwqvrrmnltkxdkjz.init.cedexis-radar.net', 'iadsdk.apple.com', 'iata', 'ibm', 'icanhazip.com', 'iciot', 'icloud', 'icloud.com', 'iconfig', 'iconnectivity', 'icontrol', 'icontrol.com', 'idaasciam-inner-share.us-west-1.aliyuncs.com.gds', 'identification', 'identity.apple.com.akadns.net', 'identrust', 'ideone.com', 'ieee', 'ienvault', 'if', 'ifconfig.dk', 'ifft.com', 'ifttt', 'iftttapis.xwemo.com', 'iheart', 'iheart.com', 'iheartradio', 'ihome', 'ijofdplsm7[.]com', 'ikea', 'ima', 'imagekit.io', 'images-amazon.com', 'images-na.ssl-images-amazon.com', 'imatechinnovations.com', 'imessage', 'img-egc.xvideos.com', 'immedia', 'immedia-semi', 'immuniweb', 'imogen', 'imperial', 'imrworldwide.com', 'in.api.io.mi.com', 'in5', 'inappcheck.itunes.apple.com', 'incident', 'independent', 'index', 'indiana', 'indicative', 'indigo', 'indo', 'infinitedata-pa.googleapis.com', 'infopath', 'information', 'informer', 'init-p01md-lb.push-apple[.]com.akadns[.]net', 'init-p01md.apple.com', 'init.ess.apple.com', 'init.ess.apple.com.edgesuite.net', 'init.gc.apple.com', 'init.itunes.apple.com', 'inputapi.theneura.com', 'inquenta', 'insight', 'insignia', 'instacart', 'instagram', 'installgentoo', 'insteon', 'integra', 'integra.vtuner.com', 'integrated', 'integrating', 'intel', 'intercom', 'intergovernmental', 'international', 'internet', 'internetat', 'internetat.tv', 'internode', 'interventionen', 'invoke', 'invoxia', 'ioactive', 'ioe', 'ioeclient', 'iomi', 'ion', 'ios', 'iot', 'iot.api.bose.io', 'iot.eu-west-1.amazonaws.com', 'iotathena', 'iotmoonraker.us-west-2.prod.iot.us-west-2.amazonaws.com', 'iowa', 'ip', 'ip-lookup.org', 'ip-neighbors', 'ipad', 'ipaddress.com', 'ipados', 'ipads', 'ipc', 'ipcdn.apple.com', 'ipfire', 'iphone', 'ipod', 'iptime', 'irobot', 'irobot.axeda.com', 'irobotapi.com', 'is-ssl.mzstatic.com.itunes-apple.com.akadns.net', 'is1-ssl.mzstatic[.]com', 'is2-ssl.mzstatic[.]', 'is4-ssl.mzstatic.com', 'is5-ssl.mzstatic.com', 'isc', 'isp', 'isp,', 'isp:', 'ispy', 'isrg', 'issac', 'it', 'it.pool.ntp.org', 'ita', 'itperfection', 'its', 'itunes', 'itunes-apple', 'itunes-apple.com', 'itunes-apple.com.akadns.net', 'itunes.apple.com', 'itunes.apple.com.is5-ssl.mzstatic.com', 'itunes.com.xml', 'itv', 'ivanov', 'ivy', 'iwork.com.xml', 'j.t.baker', 'jack', 'jamf', 'janbar/noson-app', 'jandedobbeleer', 'jason', 'jatt', 'java', 'javascript', 'javatpoint', 'jawsdb', 'jboss', 'jeff', 'jeffrey', "jesse's", 'jigga', 'jimi', 'jk.ry', 'joao', 'joe', 'john', 'jonathan', 'jpmorgan', 'jpp-net', 'js-cdn', 'js2coffee', 'jsdelivr', 'json', 'jsp', 'juan', 'justinstolpe.com', 'juzi', 'jwmedia-oss.oss-cn-shenzhen.aliyuncs.com', 'kaber', 'kaggle', 'kain', 'kakaotv', 'kalpataru', 'kasa', 'kasacare', 'kaspersky', 'katastros', 'kaufmann', 'kb974488', 'kbase', 'kc-s301ae', 'kde', 'kdk', 'kendrick', 'kerargaouyat', 'kerika', 'keychain', 'kfix/sleepproxyserver', 'khanmammadov@vera.com.uy', 'khms1.googleapis.com', 'kikkoman', 'kindle', 'kindle-time.amazon.com', 'kindleforpc-installer', 'kinesis', 'king', 'kinsa', 'kinvolk.io', 'knight', 'knowledge', 'knox', 'koalazak', 'kochava', 'kodak', 'kodi', 'koogeek', 'kotlin', 'kpn', 'kqms', 'kraftwerk', 'krsh', 'ksmobile.com', 'ksoap2-android', 'ktpx-eu.amazon.com', 'ktpx-tv', 'ktpx-uk.amazon.com', 'ktpx.amazon.com', 'kuliner', 'kuow', 'kwynn.com', 'kyong', 'kyunggido-guangdongsheng', 'la', 'lace', 'ladyscent.com', 'lake', 'lakeviewlodgedl.com', 'lambdatek', 'lamssettings-pa.googleapis.com', 'laramie', 'laserfiche', 'lastline', 'lavf', 'lavf_check', 'lb-link', 'ldp', 'le', 'leeo.pool.ntp.org', 'legacy', 'legato', 'legitimate', 'lehi', 'leica', 'lenovo', 'lenox', "let's", 'lg', 'libav.git', 'libavcodec', 'libavformat', 'liberty', 'liberty.logs.roku.com', 'libre', 'libreddit', 'libreelec', 'lifi', 'liftmaster', 'lifx', 'lifx.com', 'lightify', 'lightspeed', 'lil', 'lincolnville', 'linda.radio', 'lineageos', 'link', 'link.theplatform.com', 'linkedin', 'linkezofmidwest', 'linksys', 'linux', 'lisa', 'lisco', 'list', 'listen', 'lite-on', 'liteon', 'litestream', 'little', 'live', 'livecommunity', 'liveramp', 'living', 'living-accs.us-east-1', 'lmt822/comp116-mli', 'local', 'local.com', 'localcam', 'localcam-belfair', 'localhost', 'locally', 'localtel', 'localytics', 'location-inner.cn-nanjing.aliyuncs.com.gds', 'location-services-measurements.s3-external-1.amazonaws.com', 'log', 'log-ingestion.samsungacr.com', 'log.us.xiaoyi.com', 'loggly', 'login.gov', 'login.live.com', 'logitech', 'logmein', 'logo', 'long', 'longman', 'longview', 'lorraine', 'los', 'louisville', 'loupedeck', 'lowes.com', 'ls.cti.roku.com', 'lua-luasocket', 'lua-socket_2.0.2-make.patch', 'luasocket', 'luasocket:', 'luci', 'luges', 'lutron', 'lutron.broker.xively.com', 'lutronpro', 'm-go', 'm.media-amazon.com', 'm/i', 'ma67-c.analytics.edgesuite.net', 'ma67-r.analytics.edgekey.net', 'mac', 'mac-forums', 'macbook', 'macintosh', 'macos', 'macosx', 'macpaw', 'macrumors', 'madboa.com', 'maddox/harmony-api', 'mads', 'mads-eu.amazon.com', 'mads.amazon-adsystem.com', 'maemo.org', 'magic', 'magnepan', 'magyar', 'mail-attachment.googleusercontent.com', 'mail.ru', 'majestic', 'major', 'makeuseof', 'maltiverse', 'malwarebytes', 'malwaretips', 'manchester', 'manifest.localytics.com', 'mans.io', 'maradns', 'marcus', 'maria', 'marine', 'mark', 'marketplace', 'markmonitor', 'markmonitor,', 'marshallofsound', 'marshallofsound/google-play-music-desktop-player-unofficial', 'martha', 'martin', 'marvel', 'marvell', 'marzbar', 'mashable', 'masterpiece', 'match.adsrvr.org', 'mathsci.solano.edu', 'matiu', 'mattermost', 'maurhinck1@gmail.com', 'maven', 'maxdome', 'mbmccormick/dropcam', 'mboxtrace', 'mbp', 'mdm', 'mdns', 'mdns/dns-sd', 'mdnsresponder', 'mdpi', 'me.com.xml', 'meb', 'mecool', 'media-exp1.licdn.com', 'media.bose.io', 'mediamath', 'medianavico', 'mediaservices.cdn-apple.com.edgesuite.net', 'mediate.com', 'mediawiki', 'medium', 'meethue.com', 'melchor9000.net', 'memios', 'memoryframe', 'mendeleybot', 'mercator-', 'mercedes-benz', 'mercuryboard', 'mercuryboard_user_agent_sql_injection.nasl', 'meridian', 'merlin', 'merriam-webster', 'messaging-director-us-east-1.amazon.com', 'mesu.apple.com', 'meta', 'metadriver/discoveryprocess.md', 'metrics.nba.com', 'mgo', 'mi', 'mi-img.com', 'miami', 'mibox', 'michael', 'michigan', 'micloudconnector', 'microlab', 'microsoft', 'microsoft-cryptoapi', 'microsoft-windows', 'microsoft-windows-nt', 'microsoft-windowsmediaplayer', 'microsoftedge', 'microsoft®', 'mifit', 'mikrotic', 'mikrotik', 'miktex', 'milenio', 'millennia', 'millennia®', 'mimecast', 'minecraft', 'minissdpd', 'miniupnp', 'miniupnpc', 'miniupnpd', 'mio', "mio's", 'mio-app', 'mios', 'mios.com', 'miospizza', 'mirror', 'mischosts/apple-telemetry', 'mistore.pk:', 'miui', 'mixcloud', 'mixpanel', 'mixpanel.com', 'mmg.whatsapp.net', 'mms', 'mnum1hy0-android', 'mo/cofeb/mgo', 'mobile', 'mobile.webview.gmail.com', 'mobileanalytics.us-east-1.amazonaws.com', 'mobilemaps-pa', 'mobilemaps-pa-gz.googleapis.com', 'mobilemaps-pa.googleapis.com', 'mobilenetworkscoring-pa.googleapis.com', 'mobileread.mobi', 'mobilesafari', 'mockoon', 'mode', 'modo', 'modulator', 'mol', 'monero', 'mongo', 'monitortlssm', 'monorail', 'moonpoint', 'moovit', 'moovitapp.com', 'morioh', 'motoe2', 'motortrend', 'mountain', 'movilixa', 'mozilla', 'mozilla/4.0', 'mozilla/5.0', 'mozillawiki', 'mp3stream', 'mrsputnik', 'mscrashes.self', 'msdn', 'msh', 'msh-london', 'msh.amazon.co.uk', 'msh.amazon.com', 'msie', 'msmetrics.ws.sonos.com', 'msnbc', 'msys2', 'mui', 'multicast', 'multimedios', 'musicbee', 'mv', 'my', 'mybb', 'mycroft', 'mydlink', 'mydns', 'mydns-ng', 'myharmony', 'myheritage', 'myhomescreen.tv', 'myip.es', 'myq', 'mysimplelink', 'mysql®', 'mywot', 'mzstatic', 'mzstatic.com', 'mzstatic.com.xml', 'métron', 'n-devs.tplinkcloud.com', 'n-euw1-devs.tplinkcloud.com', 'n1npjk0x7l[.]com', 'n5812.vh.eunethosting.com', 'na.api.amazonvideo.com', 'nancy', 'nanoleaf', 'naruto', 'nas', 'nasdaq', 'national', 'nature', 'naver', 'nba', 'nbasense', 'nbc', 'ncaa.org', 'nccp.netflix.com', 'ncert', 'nds', 'neatcomponents', 'neosec', 'nessus', 'nessus-professional-fullcolor-rgb.svg', 'nessus.org', 'nessusreportpraser', 'nest', "nest's", 'neston', 'net', 'netapp', 'netatmo', 'netatmo-exporter', 'netatmowelcomewizard_windo', 'netcraft', 'netflix', 'netgate', 'netgear', 'netgear,', 'netgear_ext', 'netify', 'netmux', 'netop', 'netscape', 'netscout', 'netspi', 'netsupport', 'nettime.pool.ntp.org', 'network', 'neura', 'neverssl', 'neverssl.com', 'new', 'newegg', 'news', 'nextdns', 'nextdns.io', 'nexus', 'nexus-websocket-a.intercom.io', 'nexusapi.dropcam.com', 'nfl', 'nfl.com', 'nflenterprises.tt.omtrdc.net', 'nflsavant.com', 'nflxvideo.net', 'nginx', 'niagahoster', 'nickeil', 'nielsen', 'nieman', 'nighthawk', 'ninjacentral', 'ninox', 'nintendo', 'nipr', 'nirsoft', 'nist', 'nist,', 'nist.gov', 'nixplay', 'nl.pool.ntp.org', 'nmap', 'node', 'node-dns-sd', 'node-red-contrib-openhab2', 'nodemcu', 'nokia', 'nom-iq', 'nord', 'north-america.pool.ntp.org', 'northeastern', 'northern', 'norton™', 'npm', 'npr', 'nrdp51-appboot.netflix.com', 'ns1', 'ns1.netatmo.net', 'ns2.dlink.com', 'nse', 'nsf', 'nslookup.io', 'nsw.gov.au', 'ntp', 'ntp-0.cso.uiuc.edu', 'ntp-1.ece.cmu.edu', 'ntp-2.ece.cmu.edu', 'ntp-g7g.amazon.com', 'ntp-g7g.ntp-g7g.amazon.com', 'ntp.ntsc.ac.cn', 'ntp.org', 'ntp.pool.org', 'ntp.se', 'ntp.xs4all.nl', 'ntp0.cornell.edu', 'ntp2a.mcc.ac.uk', 'ntpclient', 'ntpserver', 'nubati.net', 'nuget', 'nullsoft', 'numberfire', 'nutanix', 'nuvola', 'nvidia', 'nvidiagrid.net', 'nwbrowser', 'nwsvr', 'nxp', "nxp's", 'nys', 'oakbits', 'oauth.xwemo.com:8443', 'oauth2.googleapis.com', 'observatory', 'ocn', 'ocsp-ds.ws.symantec.com.edgekey.net', 'ocsp.apple.com', 'ocsp.digicert.com', 'odm', 'odm.platform.cnn.com', 'oe-core', 'oem', 'oem/odm', 'office', 'official', 'ofilipatrick', 'oisd', 'okhttp', 'okhttpclient', 'okta', 'oldham', 'olin', 'olivier', 'omada', 'omniture', 'one', 'one-way', 'oneplus', 'online', 'onsong', 'open', 'open-meteo.com', 'openbsd', 'opencve', 'opendns', 'openhab', 'openhome', 'openluup/userdata.lua', 'openssl', 'openstack', 'opensuse', 'opentrons', 'openvpn', 'openweathermap', 'openwrt', 'openwrtai', 'opera', 'opnsense', 'oppo', 'opr', 'optimizationguide-pa.googleapis.com', 'oracle', 'orange', 'orangebot', 'orbi', 'organization', 'origin.guzzoni-apple.com.akadns.net', 'ormp3', 'os', 'os.fandango.com', 'osb-krsvc.samsungqbe.com', 'osi', 'ospfpacket', 'ospscraper', 'osram', 'ot.io.mi.com', 'otosection', 'ott', 'outlook', 'overdrive', 'ovh', 'owasp', 'owasp.org', 'owntone', 'p06-ckdatabase.icloud.com', 'p07-geows.icloud.com', 'p08-ckdatabase.icloud.com', 'p09-ckdatabase.icloud.com', 'p13.zdassets.com', 'p15-ckdatabase.icloud.com', 'p18-bookmarks.icloud.com', 'p212-mccgateway-china.icloud.com', 'p27-ubiquity.icloud.com', 'p2p3.nwsvr2.com', 'p302-ckdatabase.icloud.com', 'p306-mcc.icloud.com', 'p31-contacts.icloud.com', 'p310-ckcoderouter.icloud.com', 'p33-ckdatabase.icloud.com', 'p37-ckdatabase.icloud.com', 'p42-iwmb9.icloud.com', 'p43-buy.itunes-apple.com.akadns.net', 'p50-ckcoderouter.icloud.com', 'p52-calendars.icloud.com', 'p54-buy.itunes.apple.com', 'p54-caldav.icloud.com', 'p54-calendars.icloud.com', 'p54-ckcoderouter.icloud.com', 'p54-ckdatabase.icloud.com', 'p54-contacts.icloud.com', 'p56-buy.itunes.apple.com.akadns.net', 'p58-fmfmobile.icloud.com', 'p679', 'pa-500', 'packet', 'packettotal', 'packt', 'palabasento', 'palavra', 'palo', 'paltap', 'pancake', 'pancake.apple.com', 'pancake.g.aaplimg.com', 'pandora', 'paramount', 'parrot', 'parse_table', 'partiality.itunes.apple.com', 'partner', 'partnerapis.xwemo.com', "paspo's", 'passive', 'passport', 'pastebin.com', 'pau', 'paul', 'paulina', 'payment.deezer.com', 'paymentology', 'pcmag', 'pd.itunes.apple.com', 'pdns', 'pdns_server', 'pe', 'peach', 'pecsusa.com', 'pegasus', 'pen', 'people-pa.googleapis.com', 'pepperland', 'perfect', 'perflyst', 'perflyst.github.io', 'perflyst/piholeblocklist', 'peri', 'perl', 'pet', 'petenetlive', 'pfizer', 'philips', 'phillips', 'phishcheck', 'php', 'phpfox', 'phpmyadmin', 'phyks', 'physical', 'physikalisch-technische', 'pi', 'pi-hole', 'pi-hole/ftl', 'pi-hole/pi-hole', "pia's", 'pihole', 'piholeblocklist', 'piholeblocklist/smarttv.txt', 'pimatic-echo', 'ping', 'pink', 'pinterest', 'pipe.prd.skypedata.akadns.net', 'pipedream', 'piper', 'piper-heidsieck', 'pitchbook', 'pix-star', 'pixabay', 'pixel', 'pixel.everesttech.net', 'pixel.solvemedia.com', 'pixels', 'pixstar', 'pki', 'pl.scdn.co', 'play', 'play.googleapis.com', 'playmods', 'playstation', 'plesk', 'plex', 'plos', 'pngkey', 'pocket', 'policy-based', 'pool.ntp.org', 'portable', 'portal', 'portsmouth', 'post', 'postlund/dlink_hnap', 'postman', 'poweramp', 'powerdns', 'precision', 'premier', 'premiumproxy.net', "president's", 'press', 'press@getpiper.com', 'prezi', 'prime', 'privacyscore', 'private', 'private.googleapis.com', 'pro-football-reference.com', 'proactive', 'processing.org', 'processone', 'processor.smartcamera.api.io.mi.com', 'proclivity', 'prod.amazoncrl.com', 'prod.amcs-tachyon.com', 'prod.insteon.pubnub.com', 'prod1-fs-xbcs-net-1101221371', 'product', 'professionally', 'professor', 'profile', 'project', 'property', 'property,', 'prospect', 'proton', 'proxy', 'proxypremium.top', 'ps', 'ps-pdf.cxx', 'ps-tree', 'ps3', 'psf', 'pswec.com', 'pt', 'pt/co/ru/mgo', 'ptb.de', 'pubads.g.doubleclick.net', 'pubchem', 'pubgw.ads.yahoo.com', 'publicwww.com', 'pubmatic', 'pubmatic.com', 'pubnub', 'pubnub-client', 'pubnub-curses', 'pubnub.com', 'pubnub/c-core', 'pubsub.pubnub.com', 'pulsedive', 'pulumi', 'puppet', 'purnima', 'pypi', 'pyrebase', 'pyrok', 'pyromation', 'python', 'python-miio', 'qiku', 'qnap', 'qps', 'qq', 'qualcomm', 'qualys', "quarkslab's", 'quartz', 'queen', 'quickstart', 'quora', 'qwest', 'r', 'r.dlx.addthis.com', 'r.turn', 'r/applehelp', 'r/appletv', 'r/controld', 'r/homenetworking', 'r/macsysadmin', 'r/pihole', 'r/protonmail', 'r/tplink_omada', 'r/wifi', 'r01-noah.dch.dlink.com', 'race', 'rachio', 'rackspace', 'radar.cedexis.com', 'radio', 'radio-online-json-url/australia.json', 'radware', 'rajiv', 'rakuten', 'rakuten.tv', 'randal', 'random', 'ransomware', 'rapidapi', 'rapiddns', 'raptor', 'raspberry', 'ravelin', 'raven.js', 'raymii.org', 'rbc', 'rbs', 'rbs-sticky.august.com', 'rbsgimmel@groups.io', 'rclone', 'react-native-firebase', 'read', 'read&write', 'readaloud.googleapis.com', 'readynas', 'red', 'reddit', 'redirector.gvt1.com', 'redmi', 'redteam.pl', 'refinedet:', 'reflectoring', 'register.xmpush.global.xiaomi.com', 'registry.api.cnn.io', 'regular', 'remke', 'remoteassistance@vera-us-oem-ts11.mios.com', 'reolink', 'report', 'republic', 'researchgate', 'resource', 'respeaker', 'response', 'restricted.googleapis.com', 'reuters', 'reverseaccess', 'reversing', 'revir', 'revision', 'rexing', 'rezolt', 'rezolute', 'rezolute,', 'rfc', 'rfc6763', 'rh-mongodb36', 'rhel.pool.ntp.org', 'rhinoceros', 'richard', 'ring', 'ringtone', 'riskified', 'riskiq', "rixstep's", 'rmc', 'robbshop', 'robert', 'robo', 'roborock', 'roborock-vacuum-s6', 'robtex', 'rock', 'rockville', 'rogers', 'roi4cio', 'roku', 'roku,', 'roku.com', 'rolex.peachnet.edu', 'rollingwood.logs.roku.com', 'roma', 'romwebclient', 'roomba', 'rpm', 'rpmforge', 'rpsv4', 'rslonline', 'rss.rpg', 'rssing.com', 'rtl', 'rtsak', 'rturn', 'ru', 'ru.pool.ntp.org', 'rue', 'ruelala', 'runkeeper', 'runzero', 'russell', 'russian', 'rustez', 'rx&d', 'rxr.ravm.tv', 'rytilahti/python-miio', 's-trust', 's.amazon-adsystem.com', 's.cdn.turner.com', 's.mzstatic.com', 's.mzstatic.itunes-apple.com.akadns.net', 's1.symcb.com', 's2.symcb.com', 's3', 's3-eu-west-1.amazonaws.com', 's3-us-west-2.amazonaws.com', 's3.amazonaws.com', 's3.ap-southeast-2.amazonaws.com', 's3.eu.west.1.amazonaws.com', 's3client', 'saaspass', 'sabnzbd', 'safari', 'safebrowsing.google.com', 'safety', 'sagemcom', 'sales', 'salesforce', 'salesforce.com', 'salmonsec', 'sam', 'samboy/maradns', 'samples.voice.cti.roku.com', 'samsumg', 'samsung', "samsung's", 'samsung-eden.wuaki.tv', 'samsung.com', 'samsung_query_fqdn.txt', 'samsungacr.com', 'samsungcloud.tv', 'samsungcloudsolution.com', 'samsungcloudsolution.net', 'samsungelectronics.com', 'samsungnyc.com', 'samsungosp.com', 'samsungqbe.com', 'san', 'sandeepsidhu.com', 'sanjana4011/ksoap2-android', 'sans', 'sap', 'sapito', 'sast', 'satellite', 'sb.scorecardresearch.com', 'sba', 'scam', 'scamadviser', 'scamvoid', 'scanalert', 'scaninfo', 'scanurl', 'schmidt', 'sciencedirect.com', 'sciences', 'scip', 'scontent-atl3-1.xx.fbcdn.net', 'scontent-cdg2-1.cdninstagram.com', 'scontent-cdt1-1.cdninstagram.com', 'scontent-frt3-2.cdninstagram.com', 'scontent-hkg4-2.xx.fbcdn.net', 'scontent-mrs2-1.cdninstagram.com', 'scontent-mrs2-2.cdninstagram.com', 'scontent-mrs2-2.xx.fbcdn.net', 'scontent.cdninstagram.com', 'scontent.fdac6-1.fna.fbcdn.net', 'scontent.ftlv5-1.fna.fbcdn.net', 'scorecard', 'screen', 'scribd', 'scribe.logs.roku.com', 'script', 'sdkconfig.ad.intl.xiaomi.com', 'sdkconfig.ad.xiaomi.com', 'sea', 'searchcode', 'sears', 'seattle', 'seattle,', 'sebastian', 'sec.gov', 'sec_hhp_[tv]', 'secret', 'sectigo', 'secure', 'secure-leadback.stubhub.db.advertising.com', 'secure.leadback.advertising.com', 'securepubads', 'securepubads.g.doubleclick.net', 'secureswitches.com', 'securifi', 'security', 'securitytrails', 'securiwiki', 'seedonk', 'segment', 'segment.io', 'selling', 'semanticlocation-pa.googleapis.com', 'seminar', 'seminarsonly', 'sengled', 'sense', 'sensible', 'sensus12', 'sentry', 'seo', 'sequans', 'sercomm', 'serious', 'sermat', 'servedby.flashtalking.com', 'server', 'server-54-230-9-198.man50.r.cloudfront.net', 'server:windows', 'servercentral', 'serveroperations.net', 'serviio', 'settings', 'sevro', 'sf', 'sfr', 'shadowwhisperer', 'shb', 'shenzhen', 'sheraton', 'shhh-cretly', 'shi', 'shift4shop', 'shodan', 'shopback', 'shopify', 'showtime', 'shutterfly', 'shutterstock', 'si(100)', 'sigma-aldrich', 'signal', 'signal-desktop', 'signaler-pa.googleapis.com', 'signify', 'silk', 'simage2.pubmatic.com', 'similarweb', 'simon', 'simple', 'siri', 'sister', 'sister2sister', 'sitejabber', 'sk', 'skill', 'skylight', 'skype', 'skyzoo', 'sleep', 'sleepproxyclient', 'sleepproxyserver', 'slideshare', 'sm-g610f', 'sm-g930p', 'sm-g950u', 'sm-j260az', 'sm-t380', 'sm-t580', 'small', 'smart', 'smart-tv', 'smart-tv-ads-tracking.txt', 'smartclip.net', 'smartfrog&canary', 'smartlabs', 'smartrecruiters', 'smartscout', 'smartthings', 'smartthings-dev', 'smartthingsmud.json', 'smartthingsoh3', 'smugmug', 'smy.iheart.com', 'snap', 'sniffinet', 'snyk', 'soap', 'social', 'socialgrep', 'softscheck', 'softwareupdates.amazon.com', 'sohdms', 'solarwinds', 'solve', 'solved:', 'solvemedia.com', 'solvvy', 'sonicwall', 'sonnar', 'sonos', 'sonos,', 'sonoszp', 'sony', 'sophos', 'soul', 'sound', 'soundcloud', 'soundhound', 'soundtouch', 'soundtouch®', 'soundtouch™', 'sourcefire.pool.ntp.org', 'sourceforge.net', 'sovereign', 'spark', 'spectrum', 'spectrum.s3.amazonaws.com', 'speech', 'speechify', 'speedtouch', 'sphere', 'spiceworks', 'spiritfarer', 'sponsorships', 'sportsmax', 'spotify', 'spotify.com', 'spotify_eliminate_advertisements', 'spotnet', 'sprint', 'spyhunter', 'spying', 'spywarecure', 'sql', 'sqlshack', 'squid', 'sr.symcd.com', 'ss64.com', 'ssl', 'ssl-images-amazon.com', 'ssl-tools', 'ssl/tls', 'ssltrust', 'ssofootball', 'ssp.samsungosp.com', 'stack', 'stackblitz', 'stackshare', 'standardizing', 'star', 'starck', 'startcom', 'starzsoft', 'state', 'static-cdn.jtvnw.net', 'static.doubleclick.net', 'static.ips.apple.com', 'static.moovitapp.com', 'static.verify.sec.xiaomi.com', 'static.zdassets.com', 'statistics', 'stats.g.doubleclick.net', 'statuspage.io', 'steam', 'stefan', 'stendhal', 'stephan', 'steve', 'stevenblack/hosts', 'stitcher', 'stmicroelectronics', 'stores', 'storj', 'streamdns', 'streamguys.com', 'streaming.bose.com', 'streamingoauth.bose.com', 'streamtheworld', 'streamyard', 'stuart', 'stun.wtfismyip.com', 'su', 'su18.wagbridge.gaode.com.gds', 'sub', 'subdomain', 'sugarland.sb.roku.com', 'sundial.columbia.edu', 'super', 'supercharged', 'supercowpowers', 'superior', 'support-fr.deezer.com', 'support.com', 'supported', 'supporting', 'suppress', 'surgemail', 'surgeweb', 'suse', 'sustainable', 'suwon', 'suzuki', 'swann', 'swar', 'swisscom', 'switcher', 'symantec', 'symc', 'symcd.com', 'sync', 'sync-tm.everesttech.net', 'syneos', 'synergetic', 'synology', 'sysntpd', 'system', 'systemd-timesyncd', 'systemtek', 't-mobile', 't.mookie1.com', 'tab-tv', 'tailscale', 'tailwind', 'taiwan', 'taiyo', 'tanning', 'tarlab', 'tatyana', 'tax', 'tbd', 'tbs', 'tbs-certificates.co.uk', 'tcl', 'tcpdump/libpcap', 'tcplighting', 'tealium', 'tech', 'techdocs', 'techflow', 'techmeme', 'technical', 'techpenny.com', 'techpro', 'techradar', 'techrepublic', 'techshift.net', 'techstars', 'techtarget', 'techy', 'teejlab', 'telefonica', 'telerik', 'tenable', 'tenable,', 'tenable.ot', 'tenable®', 'terminal', 'terraform', 'test', 'texas', 'texstudio', 'textdoc', 'tf-login-button', 'thai-nk.com', 'thaweatherman', 'thawte', 'thawte,', 'the', 'thebackyardnaturalist.com', 'theblocklist', 'themeforest', 'themoon.linksys.router', 'thesaurus.com', 'theterninn.com', 'theworld', 'thinksystem', 'thinkware', 'thomas', 'thorax-web', 'threat', 'threatcrowd', 'threatcrowd.org', 'threatminer.org', 'thunderbird', 'ti', 'ticalc.org', 'tick.chi1.ntfo.org', 'tiktok', 'time', 'time-a-g.nist.gov', 'time-a-wwv.nist.gov', 'time-a.nist.gov', 'time-b-g.nist.gov', 'time-b.nist.gov', 'time-c-g.nist.gov', 'time-d-g.nist.gov', 'time-ios.g.aaplimg.com', 'time-macos.apple.com', 'time-osx.g.aaplimg.com', 'time.apple.com', 'time.gov', 'time.nist.gov', 'time.nrc.ca', 'time.samsungcloudsolution.com', 'time.windows.com', 'timetools', 'timezoneconverter.com', 'tinman', 'tinnitus', 'tinytts.amazon.com', 'tizen', 'tizen30.wuaki.tv', 'tlc', 'todo-g7g.amazon.com', 'todo-ta-g7g-preprod.amazon.com', 'todo-ta-g7g.amazon.com', 'togo', "tom's", 'tomtom', 'tony', 'top', 'touchdown', 'tourism', 'towerdata', 'town', 'tp', 'tp-link', 'tp-link.com', 'tpc.googlesyndication.com', 'tpcamera', 'tplink_omada', 'tplinkcloud.com', 'tplinkra.com', 'tracker', 'trackr', 'traefik', 'transmission', 'transport-trust.ru', 'trellis.law', 'trend', 'trendnet', 'triage', 'trident', 'trident/4.0', 'tripadvisor', 'troubleshooting', 'trunktr2.oss-cn-shanghai.aliyuncs.com.gds', 'trustasia', 'trustpilot', 'trustscam', 'tt.omtrdc.net', 'ttrh-700', 'tubes', 'tumblr', 'tunein', 'tunein.com', 'tunesremote', 'turicas', 'turn', 'turner', 'turner.com', 'turnip', 'turnip.cdn.turner.com', 'tuxfamily', 'tuya', 'tv', 'tv-static.scdn.co', 'tvos', 'tvsbook', 'twilio', 'twitch.tv', 'twitch.tv/wailerswl', 'twitter', 'twonky', 'txn', 'tycho.ws', 'type', 'typeform', "typeform's", 'typekit', 'typekit.net', 'u.', 'u.s.', 'u1.amazonaws.co', 'uav', 'uber', 'ubiquiti', 'ubiquiti-discovery', 'ubuntu', 'ucb', 'udap', 'udm', 'uelfbaby', 'ufrn', 'ui', 'uiboot.netflix.com', 'uidivstats', 'uio', 'uk.pool.ntp.org', 'umbrella', 'ums', 'unagi', 'unagi-na.amazon.com', 'unblock', 'unbound', 'uncle', 'unearthing', 'unifi', 'unistarchemicalcom.ipage.com', 'united', 'universal', 'universalmediaserver', 'university', 'unizeto', 'unms', 'unofficial', 'unsplash', 'update.googleapis.com', 'updates.amazon.com', 'upland', 'upload', 'uploader', 'upnp', 'upp.itunes.apple.com', 'urban', 'url', 'urlscan.io', 'urlvoid', 'urrea', 'us', 'us-east-1.elb.amazonaws.com', 'us-environmental', 'us.api.iheart.com', 'us.pool.ntp.org', 'usa', 'usatf', 'uscis', 'use-stls.adobe.com.edgesuite.net', 'usenix', 'user-agents.net', 'useragentstring.com', 'userlocation.googleapis.com', 'utm', 'uw', 'v', 'v2.broker.lifx.co', 'v6.netatmo.net', 'vaguefield', 'vaillant-netatmo-api', 'valkyrie', 'valle', 'valve', 'vas.samsungapps.com', 'vbulletin', 'vedbex', 'veille', 'velux', 'venturi', 'vera', 'vera-us-oem-authd.mios.com', 'vera-us-oem-authd11.mios.com', 'vera-us-oem-authd12.mios.com', 'vera-us-oem-device11.mios.com', 'vera-us-oem-relay11.mios.com', 'vera-us-oem-relay31.mios.com', 'vera-us-oem.relay11.mios.com', 'veraalerts', 'vera™', 'verein', 'verisign', 'verizon', 'version', 'vico', "victoria's", 'videojs', 'village', 'vimeo', 'vincentkenny01', 'vindicosuite', 'virusresearch.org', 'virustotal', 'visa', 'vision', 'visual', 'visuino', 'vizbee', 'vk', 'vlc', 'vmb3010', 'vmware', 'vnutz', 'vocabulary.com', 'vocolinc', 'vodafone', 'voices.rss', 'voledevice-pa.googleapis.com', 'volt', 'volumio', 'vorstand', 'voting', 'voxox', 'vtuner', 'vtuner.com', 'vudu', 'vym', 'vysshaya', 'w3', 'w3docs', 'w3schools', 'wac564', 'wailers', 'wailerswl', 'wakelet', 'wall-smart', 'walmart', 'walmart.com', 'walnuts', 'walt', 'wansview', 'wapi.theneura.com', 'warner', 'warnermedia', 'warnermedia,', 'washington', 'watanabes', 'watchguard', 'wayfair', 'wb', 'wc9500', 'wcasd', 'wd', 'wdde', 'weather.com', 'weatherapi.com', 'web', 'webflow', 'webinspect', 'webkit', 'webpage', 'webroot', 'websec', 'website', 'webtechsurvey.com', 'weintour-cattolica', 'wells', 'wemo', 'wener', 'west', 'western', 'wexted', 'wextractor', 'wget', 'what', 'whatsapp', 'whatsapp.net', 'whatsmydns.net', 'who.is', 'whois', 'whois.com', 'whud-fm', 'wi-fi', 'wifi.solvemedia.com', 'wikihow', 'wikileaks', 'wikimedia', 'wikipedia', 'wiktionary', 'wild', 'williams', 'win32', 'winamp', 'windows', 'windowsspyblocker', 'wink', 'wink-connected', 'winkapp', 'winscp', 'winters', 'wipliance', 'wired', 'wireless', 'wireshark', 'wisenet', 'wisp.pl', 'wispr', 'withings', 'wix', 'wl', 'wl.amazon-dss.com', 'wlc', 'wms.assoc-amazon.com', 'wmsdk', 'wn802t-200', 'wodsee', 'wolt', 'wolttigroup', 'wordpress.com', 'wordpress.org', 'works', 'workspace', 'world', 'world-gen.g.aaplimg.com', 'worldwide.bose.com', 'wosign', 'wot', 'wowza', 'wp', 'wrpd.dlink.com', 'wrvo', 'wsa', 'wsdapi', 'wsp', 'wtf', 'wtfismyip.com', 'wu', 'wuaki.tv', 'wwp.greenwichmeantime.com', 'wwv', 'wwv,', 'www-domain-com.cdn.ampproject.org', 'www.aboutamazon.com', 'www.belkin.com', 'www.ecdinterface.philips.com', 'www.etsy.com', 'www.example.com', 'www.example.org', 'www.google.com', 'www.gstatic.com', 'www.mgo.com', 'www.myedimax.com', 'www.ntp.pool.org', 'www.opensyllabusproject.org', 'www.otaru-uc.ac.jp', 'www.owasp.org', 'www.tenable.com', 'wyndham', 'wytwornia', 'wyze', 'wyze-general-api.wyzecam.com', 'wyzecam', 'x-force', 'x.bidswitch.net', 'x.dlx.addthis.com', 'xbcs.net', 'xbox', 'xcode-maven-plugin', 'xda', 'xen', 'xenforo', 'xenserver', 'xfinity', 'xiaomi', 'xiaomi.net', 'xiaomi_account_quick_login', 'xiaoyi', 'xiarch', 'xieng', 'xively', 'xl', 'xp.apple.com', 'xs4all', 'xtm', 'xwemo.com', 'xx-fbcdn-shv-01-cdt1.fbcdn.net', 'yabrowser', 'yahoo', 'yahoo!', 'yale', 'yamaha', 'yandex', 'yeelights', 'yhs.5020', 'yi', 'yiv.com', 'yoast', 'yonkers', 'youtube', 'youtubei.googleapis.com', 'yowser', 'yr', 'yt3.ggpht.com', 'ytimg.com', 'yue', 'yumpu', 'z.cdn.turner.com', 'zalando', 'zanies', 'zathura-ps', 'zayo', 'zebra', 'zell', 'zembra', 'zendesk', 'zenodo', 'zero', 'zeroconf', 'zhejiang', 'zheqogzh', 'zhimi-airmonitor-v1', 'zmoviles', 'zoho', 'zoom', 'zoominfo', 'zoominfo.com', 'zowee', 'εxodus', 'вмешательства', '上海交通大学', '东南大学', '北京大学', '北京大学国内常用ntp服务器地址及ip', '北京邮电大学', '北京锐讯灵通科技有限公司', '小米开放平台', '清华大学']
acquired_by_gpt_vendors_list_filtered_domains = ['1932', '1stream', '2', '360se', '3bays', '51cto博客', '51degrees', '8', '@amazon', 'a', 'a+', 'a+e', 'a/b', 'aapks', 'aapl', 'aaron', 'abbott', 'abbyy', 'abc', 'aboriginal', 'abuseipdb', 'acapella', 'accenture', 'accessify', 'accountauthenticator', 'accuweather', 'acm', 'acme', 'act', 'activeperl', 'activestate', 'activite::client', 'actualite', 'acura', 'ad-free', 'ad2pi', 'ada', 'adam', 'adaway', 'adblock', 'addigy', 'adguard', 'adguardfilters', 'adguardteam', 'admincp', 'adobe', 'adsense', 'adt', 'aeon', 'afp', 'ags', 'ai-thinker', 'aidc', 'airplay', 'airport', 'airprint', 'ais', 'ajb413/pubnub-functions-mock', 'akadns', 'akamai', 'akamai-as', 'akamaighost', 'akbooer', 'akrithi', 'aladdin', 'alamy', 'albert', 'alex', 'alexa', 'alexa,', 'alibaba', 'alienvault', 'aliyun', 'all', 'allegro-software-webclient', 'almond', 'alphabet', 'amal', 'amazon', 'amazon-02', 'amazon-aes', 'amazonfiretv', 'amazonian', 'amazonsmile', 'ambrussum', 'american', 'amiga', 'amir', 'ampak', 'ams', 'amzn/selling-partner-api-docs', 'analyticsmarket', 'ancestral', 'ancestrydna', 'anderson', 'andrew', 'android', 'angeloc/htpdate', 'annapurna', 'antix-forum', 'antonio', 'anycast', 'anycodings', 'anytime', 'aol', 'ap', 'apache', 'apache-httpclient', 'apertis', 'apevec', 'apex', 'api', 'apis::firebaseremoteconfigv1::firebaseremoteconfigservice', 'apkmirror', 'apns', 'app', 'appbrain', 'apple', "apple's", 'apple-austin', 'apple-dns', 'apple-engineering', 'apple_team_id', 'applecache', 'applecoremedia', 'appletv', 'appletv3', 'appletv3,1', 'appletv5', 'appletv5,3', 'applewatch', 'applewebkit', 'application', 'applovin', 'appnexus', 'appsamurai', 'appsflyer', 'appstore', 'aptoide', 'aquaforte', 'aquatone', 'archer', 'arcol', 'arduino', 'arena', 'arlo', 'arnold', 'around', 'arpi', 'arrow', 'arthouse', 'arthur', 'aruba', 'arxiv', 'ashburn', 'ashburn,', 'ask', 'aspect', 'aspen', 'astra', 'astracleaners', 'astralagos', 'astrodienst', 'asus', 'at', 'atheros', 'atid', 'atlanta', 'atlas', 'atomtime', 'atsec', 'atv', 'atvproxy', 'audio_mpeg', 'august', 'ausweisapp2', 'authenticode', 'authorized', 'automated', 'automotive', 'autoparts', 'autovera', 'av-centerd', 'avahi', 'avahi-daemon', 'avahi-resolve', 'available', 'avast', 'avg', 'aviary', 'avira', 'avs', 'aws', 'ayde', 'azure', 'azureware', 'azurewave', 'b', 'babygearlab', 'bad', 'baddie', 'bagae', 'baidu', 'banco', 'bandwidth', 'bank', 'bard', 'barr', 'barracuda', 'basketapi', 'baylor', 'bbb', 'beijing', 'belkin', 'berkeley', 'berkshire', 'berto', 'bertoni', 'best', 'better', 'bhavana', 'bidswitch', 'big', 'bigtreetech', 'binance', 'bind', 'bing', 'bird', 'bitdefender', 'black', 'blackstone', 'blb', 'bleacher', 'blink', 'blocking', 'blocklist', 'blood', 'bluelithium', 'bombora', 'bonjour', 'boo', 'book', 'bookdown', 'bootstrap', 'bootstrapcdn', 'bose', 'boss', 'boston', 'boto3', 'boulder', 'boulder,', 'bountysource', 'brainworks', 'branditechture', 'braze', 'brettm', 'brillano', 'brilliant', 'broadbandnow', 'broadcast', 'brooklyn', 'brother', 'bso', 'bugcrowd', 'bugsense', 'bugsfighter', 'buildbase', 'busybox', 'busybox-ntp', 'c-al2o3', 'c4', 'c714', 'cab', 'cablelabs', 'cacá', 'cai', 'california', 'callpod', 'calm', 'cambridge', 'cameron', 'canadian', 'canary', 'candemir', 'candor', 'candum', 'canon', 'canopus', 'canvas', 'canwatch', 'capabilities', 'capi-lux', 'capital', 'capterra', 'captive', 'caratteristiche', 'carematix', 'carpenters', 'casetext', 'castr', 'caséta', 'catchpoint', 'cbs', 'cbsig', 'cc3100', 'cc3200', 'cc4all', 'cc4skype', 'cdc', 'cdm', 'cdn', 'cdnjs', 'cdtech', 'cedexis', 'celtic', 'centos', 'centurylink', 'ceramic', 'certbot-dns-google', 'certificate', 'certification', 'certplus', 'certs', 'chamberlain', 'chand', 'changelog', 'channel', 'charles', 'charlotte', 'check', 'checkphish', 'chenega', 'chicago', 'chico', "chili's", 'china', 'choco', 'chocolatey', 'choice', 'chris', 'christofle', 'chrome', 'chromebook', 'chromecast', 'chromecasts', 'chunghwa', 'cialde', 'cibc', 'cisco', 'citrix', 'city', 'clare', 'clark', 'clasp', 'classifying', 'clc', 'clearesult', 'clearstream', 'cleveland', 'clickthink', 'client', 'clientflow', 'cling', 'cloud', 'cloud-to-cloud', 'cloudbasic', 'cloudberry', 'cloudflare', 'cloudflare,', 'cloudfront', 'club', 'clustering', 'cmybabee', 'cnbc', 'cnn', 'cnpbagwell', 'co', 'cochenilles', 'codeberg', 'colasoft', 'collins', 'coloressence', 'configuration', 'configuring', 'connect-sessionvoc', 'connectivitymanager', 'connman', 'connor', 'conrad', 'consiglio', 'consumer', 'content', 'continuum', 'contrast', 'control', 'control-m', 'control4', 'controller', 'conviva', 'cooper', 'coredns-mdns', 'cornell', 'cornerstone', 'corpus', 'cortana', 'counter', 'courier', 'cox', 'cpanel', 'crashlytics', 'crc', 'creamsource', 'create', 'cree', 'crestron', 'criteo', 'crossfit', 'crunchbase', 'crunchyroll', 'cs', 'csc', 'csdn', 'csdn博客', 'ctg', 'cti', 'ctv', 'cult', 'cups', 'curl', 'custom', 'cuteftp', 'cve', 'cyber', 'cybergarage', 'cybergarage-http', 'cybergarage-upnp', 'cyberlink', 'cybernews', 'cybersecurity', 'cyware', 'd-link', "d-link's", 'd-trust', 'dain', 'dallas', 'dalvik', 'daniel', 'dart', 'darwin', 'dashboardadvisoryd', 'dashkiosk', 'dast', 'data', 'databricks', 'dataflair', 'datex', 'daventry', 'david', 'dc-eu01-euwest1', 'dcs-5030l', 'dcs-930l', 'dd-wrt', 'debian', 'dec', 'deets', 'deezen', 'deezer', 'deeztek', 'defy', 'deledao', 'deliverect', 'dell', 'demio', 'denver', 'department', 'dericam', 'design&innovation', 'destinypedia', 'destinypedia,', 'detroit', 'dev', 'devcentral', 'device', 'device-metrics-us', 'dga', 'didier', 'digicert', 'digicert,', 'digieffects', 'digitalinx', 'dir-850l', 'directv', 'discovery', 'disney', 'disney+', 'disqus', 'diss', 'django', 'dlink', 'dlna', 'dlnadoc', 'dna', 'dns', 'dns-sd', 'dnsbl', 'dnsmasq', 'dnssec', "doc's", 'docker', 'docslib', 'domain', 'domain:', 'domainwatch', 'dommer', 'domoticz', 'don', 'dong', 'doordash', 'doppio+', 'dorita980', 'dot', 'doubleclick', 'doubleverify', 'doulci', 'dow', 'downdetector', 'download', 'dp-discovery-na-ext', 'dpd', 'dpws', 'dr', 'dragon', 'dribbble', 'drogueria', 'dropbox', 'dropcam', 'dropcam™', 'dsa596', 'dslreports', 'dudek', 'dyn', 'e&s', 'e28622', 'eai', 'earl', 'easy', 'eatbrain', 'ebay', 'ebestpurchase', 'ecampus:', 'echo', 'echofon', 'eclipse', 'ecobee', 'ecp', 'edgecast', 'edgewood', 'edimax', 'ediview', 'edmodoandroid', 'eds', 'educba', 'ee', 'eero', 'eggplant', 'egloo', 'eleanor', 'electric', 'electricbrain', 'electronic', 'eliminate', 'elizabeth', 'elv', 'elv-/', 'elv/eq-3', 'email', 'emailsentry', 'emailveritas', 'emby', 'eml-al00', 'enabling', 'endpoint', 'engadget', 'enjoyshop777', 'enterprise', 'entries', 'entrust', 'entrust,', 'epa', 'epubreader', 'eq-3', 'equifax', 'eric', 'ericsson', 'erie', 'erwbgy/pdns', 'eset', 'esp32-cam', 'espn', 'espn+', 'espressif', 'essalud', 'essentials!', 'esszimmer', 'esxi', 'et', 'eternal', 'ethereum', 'eu-', 'ev3dev/connman', 'evomaster', 'evrythng', 'ex6120', 'exploit-db', 'express', 'external', 'ey', 'eyeota', 'ezlo', 'f-02h', 'f-droid', 'f5', 'fabric', 'fabrik', 'facebook', 'facebook,', 'facetime', 'factory', 'fadell', 'falcon', 'faleemi', 'fandango', 'fandangonow', 'fanduel', 'fantasy', 'faq', 'fashiontv', 'fast', 'fastly', 'fastly,', 'fastream', 'faststream', 'fbbruteforce', 'fca', 'fcc', 'federal', 'fedora', 'ffmpeg', 'ffmpeg/apichanges', 'fiddler', 'fingerbank', 'fire', 'fire-', 'firebase', 'firefox', 'fireserve', 'firesticks', 'firetv', 'firewalla', 'first', 'fivethirtyeight', 'flashtalking', 'flex', 'flic', 'flipboard', 'fmstream', 'font', 'food', 'forcetlssm', 'ford', 'forest', 'fortiproxy', 'fortisiem', 'fortnite', 'fox', "fox's", 'fractioncalculatorplusfree', 'frank', 'fraternitas', 'free', 'freecodecamp', 'freelance', 'freertos', 'frnog', 'ftp', 'fuboplayer', 'fujifilm', 'fulfillment', 'funimation', 'futomi/node-dns-sd', 'g', 'g2', 'gac', 'gainspan', 'gaithersburg', 'galleon', 'games', 'gamestream', 'gbd', 'gearlab', 'gecko', 'geddy', 'geekzone', 'geforce', 'gembur', 'gen', 'general', 'genesys', 'genie', 'gentoo', 'geocerts', 'geotrust', 'germany', 'getapp', 'getgo', 'getgo,', 'getorder', 'getting', 'getty', 'gicert', 'gimme', 'gists', 'gitbook', 'github', 'gitlab', 'global', 'globalsign', 'gmail', 'gmedia', 'gnu', 'go', 'go-cbor', 'gobtron', 'godiva', 'golan', 'goldborough', 'golden', 'good', 'goodheart-willcox', 'googe', 'google', 'google-cloud-platform', 'google::apis::androidmanagementv1', 'googlecast', 'googlesyndication', 'gpcapt', 'gpstracker', 'gradle', 'graduate', 'graham', 'grclark', 'greater', 'greatfire', 'greg', 'grifco', 'gruenwald', 'grupo', 'gs908e', 'gsa', 'gsas', 'gstatic', 'gts', 'guam', 'guardicore', 'guides', 'guru3d', 'guru99', 'gustavo', 'gv-dispatch', 'gw', 'gwu', 'gyeonggi', 'ha', 'hack', 'hacker', 'hackerattackvector', 'hackertoolkit', 'hackmag', 'haier', 'harley', 'harman', 'harmony', 'harpyeagle', 'hashicorp', 'haveibeenexpired', 'havis', 'hawaii', 'hawking', 'haxx', 'hd', 'hd>library>application', 'hdiotcamera', 'head-fi', 'headers', 'health', 'heat', 'heejoung', 'hehe', 'heidi', 'hello', 'help', 'hewlett', 'hewlett-packard', 'hipaa', 'hmrc', 'hobbes', 'hoh999', 'hollyman', 'holy', 'home', 'homeassistant', 'homebridge', 'homekit', 'homematic', 'homepage', 'homepod', 'homeseer', 'host-tools', 'hostname', 'hostname:', 'hot', 'hotel', 'hotnewhiphop', 'houston', 'how', 'how-to', 'hp', 'hplip', 'hs100', 'hs110', 'hs200', 'http', 'httpbrowser', 'httpclient', 'httpd_request_handler', 'https', "hu's", 'huawei', 'hubitat', 'hue', 'hulu', 'huntsman', 'hurricane', 'hwaseong', 'hybrid', 'hyjoy', 'hyperice', 'hypestat', 'i-config', 'iata', 'ibm', 'iciot', 'icloud', 'iconfig', 'iconnectivity', 'icontrol', 'identification', 'identrust', 'ieee', 'ienvault', 'if', 'ifttt', 'iheart', 'iheartradio', 'ihome', 'ikea', 'ima', 'imessage', 'immedia', 'immedia-semi', 'immuniweb', 'imogen', 'imperial', 'in5', 'incident', 'independent', 'index', 'indiana', 'indicative', 'indigo', 'indo', 'infopath', 'information', 'informer', 'inquenta', 'insight', 'insignia', 'instacart', 'instagram', 'installgentoo', 'insteon', 'integra', 'integrated', 'integrating', 'intel', 'intergovernmental', 'international', 'internode', 'interventionen', 'invoke', 'invoxia', 'ioactive', 'ioe', 'ioeclient', 'iomi', 'ion', 'ios', 'iot', 'iotathena', 'iowa', 'ip', 'ip-neighbors', 'ipad', 'ipados', 'ipads', 'ipc', 'ipfire', 'iphone', 'ipod', 'iptime', 'irobot', 'isc', 'isp', 'isp,', 'isp:', 'ispy', 'isrg', 'issac', 'it', 'ita', 'itperfection', 'its', 'itunes', 'itunes-apple', 'itv', 'ivanov', 'ivy', 'jack', 'jamf', 'janbar/noson-app', 'jandedobbeleer', 'jason', 'jatt', 'java', 'javascript', 'javatpoint', 'jawsdb', 'jboss', 'jeff', 'jeffrey', "jesse's", 'jigga', 'jimi', 'joao', 'joe', 'john', 'jonathan', 'js-cdn', 'js2coffee', 'jsdelivr', 'json', 'jsp', 'juan', 'juzi', 'kaber', 'kaggle', 'kain', 'kakaotv', 'kalpataru', 'kasa', 'kasacare', 'kaspersky', 'katastros', 'kaufmann', 'kb974488', 'kbase', 'kc-s301ae', 'kde', 'kdk', 'kendrick', 'kerargaouyat', 'kerika', 'keychain', 'kfix/sleepproxyserver', 'kikkoman', 'kindle', 'kindleforpc-installer', 'kinesis', 'king', 'kinsa', 'knight', 'knowledge', 'knox', 'koalazak', 'kochava', 'kodak', 'kodi', 'koogeek', 'kotlin', 'kpn', 'kqms', 'kraftwerk', 'krsh', 'ksoap2-android', 'ktpx-tv', 'kuliner', 'kuow', 'kyong', 'kyunggido-guangdongsheng', 'la', 'lace', 'lake', 'lambdatek', 'laramie', 'laserfiche', 'lastline', 'lavf', 'lavf_check', 'lb-link', 'ldp', 'le', 'legacy', 'legato', 'legitimate', 'lehi', 'leica', 'lenovo', 'lenox', "let's", 'lg', 'libavcodec', 'libavformat', 'liberty', 'libre', 'libreddit', 'libreelec', 'lifi', 'liftmaster', 'lifx', 'lightify', 'lightspeed', 'lil', 'lincolnville', 'lineageos', 'link', 'linkedin', 'linkezofmidwest', 'linksys', 'linux', 'lisa', 'lisco', 'list', 'listen', 'lite-on', 'liteon', 'litestream', 'little', 'live', 'liveramp', 'living', 'local', 'localcam', 'localcam-belfair', 'localhost', 'locally', 'localtel', 'localytics', 'log', 'loggly', 'logitech', 'logmein', 'logo', 'long', 'longman', 'longview', 'lorraine', 'los', 'louisville', 'loupedeck', 'lua-luasocket', 'luasocket', 'luasocket:', 'luci', 'luges', 'lutron', 'lutronpro', 'm-go', 'm/i', 'mac', 'mac-forums', 'macbook', 'macintosh', 'macos', 'macosx', 'macpaw', 'macrumors', 'maddox/harmony-api', 'mads', 'magic', 'magnepan', 'magyar', 'majestic', 'major', 'makeuseof', 'maltiverse', 'malwarebytes', 'malwaretips', 'manchester', 'maradns', 'marcus', 'maria', 'marine', 'mark', 'marketplace', 'markmonitor', 'markmonitor,', 'marshallofsound', 'marshallofsound/google-play-music-desktop-player-unofficial', 'martha', 'martin', 'marvel', 'marvell', 'marzbar', 'mashable', 'masterpiece', 'matiu', 'mattermost', 'maven', 'maxdome', 'mbmccormick/dropcam', 'mboxtrace', 'mbp', 'mdm', 'mdns', 'mdns/dns-sd', 'mdnsresponder', 'mdpi', 'meb', 'mecool', 'mediamath', 'medianavico', 'mediawiki', 'medium', 'memios', 'memoryframe', 'mendeleybot', 'mercator-', 'mercedes-benz', 'mercuryboard', 'meridian', 'merlin', 'merriam-webster', 'meta', 'mgo', 'mi', 'miami', 'mibox', 'michael', 'michigan', 'micloudconnector', 'microlab', 'microsoft', 'microsoft-cryptoapi', 'microsoft-windows', 'microsoft-windows-nt', 'microsoft-windowsmediaplayer', 'microsoftedge', 'microsoft®', 'mifit', 'mikrotic', 'mikrotik', 'miktex', 'milenio', 'millennia', 'millennia®', 'mimecast', 'minecraft', 'minissdpd', 'miniupnp', 'miniupnpc', 'miniupnpd', 'mio', "mio's", 'mio-app', 'mios', 'miospizza', 'mirror', 'mischosts/apple-telemetry', 'miui', 'mixcloud', 'mixpanel', 'mms', 'mnum1hy0-android', 'mo/cofeb/mgo', 'mobile', 'mobilemaps-pa', 'mobilesafari', 'mockoon', 'mode', 'modo', 'modulator', 'mol', 'monero', 'mongo', 'monitortlssm', 'monorail', 'moonpoint', 'moovit', 'morioh', 'motoe2', 'motortrend', 'mountain', 'movilixa', 'mozilla', 'mozillawiki', 'mp3stream', 'mrsputnik', 'msdn', 'msh', 'msh-london', 'msie', 'msnbc', 'msys2', 'mui', 'multicast', 'multimedios', 'musicbee', 'mv', 'my', 'mybb', 'mycroft', 'mydlink', 'mydns', 'mydns-ng', 'myharmony', 'myheritage', 'myq', 'mysimplelink', 'mysql®', 'mywot', 'mzstatic', 'métron', 'nancy', 'nanoleaf', 'naruto', 'nas', 'nasdaq', 'national', 'nature', 'naver', 'nba', 'nbasense', 'nbc', 'ncert', 'nds', 'neosec', 'nessus', 'nessusreportpraser', 'nest', "nest's", 'neston', 'neura', 'neverssl', 'new', 'newegg', 'news', 'nextdns', 'nexus', 'nfl', 'nginx', 'niagahoster', 'nickeil', 'nielsen', 'nieman', 'nighthawk', 'ninjacentral', 'ninox', 'nintendo', 'nipr', 'nirsoft', 'nist', 'nist,', 'nixplay', 'nmap', 'node', 'node-dns-sd', 'node-red-contrib-openhab2', 'nodemcu', 'nokia', 'nom-iq', 'nord', 'northeastern', 'northern', 'norton™', 'npm', 'npr', 'ns1', 'nse', 'nsf', 'ntp', 'ntpclient', 'ntpserver', 'nuget', 'nullsoft', 'numberfire', 'nutanix', 'nuvola', 'nvidia', 'nwbrowser', 'nwsvr', 'nxp', "nxp's", 'nys', 'oakbits', 'observatory', 'ocn', 'odm', 'oe-core', 'oem', 'oem/odm', 'office', 'official', 'ofilipatrick', 'oisd', 'okhttp', 'okhttpclient', 'okta', 'oldham', 'olin', 'olivier', 'omada', 'omniture', 'one', 'one-way', 'oneplus', 'online', 'onsong', 'open', 'openbsd', 'opencve', 'opendns', 'openhab', 'openhome', 'openssl', 'openstack', 'opensuse', 'opentrons', 'openvpn', 'openweathermap', 'openwrt', 'openwrtai', 'opera', 'opnsense', 'oppo', 'opr', 'oracle', 'orange', 'orangebot', 'orbi', 'ormp3', 'os', 'osi', 'ospfpacket', 'ospscraper', 'osram', 'otosection', 'ott', 'outlook', 'overdrive', 'ovh', 'owasp', 'owntone', 'p679', 'pa-500', 'packet', 'packettotal', 'packt', 'palabasento', 'palavra', 'palo', 'paltap', 'pancake', 'pandora', 'paramount', 'parrot', 'parse_table', 'partner', "paspo's", 'passive', 'passport', 'pau', 'paul', 'paulina', 'paymentology', 'pcmag', 'pdns', 'pdns_server', 'pe', 'peach', 'pegasus', 'pen', 'pepperland', 'perfect', 'perflyst', 'perflyst/piholeblocklist', 'peri', 'perl', 'pet', 'pfizer', 'philips', 'phillips', 'phishcheck', 'php', 'phpfox', 'phpmyadmin', 'phyks', 'physical', 'physikalisch-technische', 'pi', 'pi-hole', 'pi-hole/ftl', 'pi-hole/pi-hole', "pia's", 'pihole', 'piholeblocklist', 'pimatic-echo', 'ping', 'pink', 'pinterest', 'pipedream', 'piper', 'piper-heidsieck', 'pitchbook', 'pix-star', 'pixabay', 'pixel', 'pixels', 'pixstar', 'pki', 'play', 'playmods', 'playstation', 'plesk', 'plex', 'plos', 'pngkey', 'pocket', 'policy-based', 'portable', 'portal', 'portsmouth', 'post', 'postlund/dlink_hnap', 'postman', 'poweramp', 'powerdns', 'precision', 'premier', "president's", 'press', 'prezi', 'prime', 'privacyscore', 'private', 'proactive', 'processone', 'proclivity', 'product', 'professionally', 'professor', 'profile', 'project', 'property', 'property,', 'prospect', 'proton', 'proxy', 'ps', 'ps-tree', 'ps3', 'psf', 'pt', 'pt/co/ru/mgo', 'pubchem', 'pubmatic', 'pubnub', 'pubnub-client', 'pubnub-curses', 'pubnub/c-core', 'pulsedive', 'pulumi', 'puppet', 'purnima', 'pypi', 'pyrebase', 'pyrok', 'pyromation', 'python', 'python-miio', 'qiku', 'qnap', 'qps', 'qq', 'qualys', "quarkslab's", 'quartz', 'queen', 'quickstart', 'quora', 'qwest', 'r', 'r/applehelp', 'r/appletv', 'r/controld', 'r/macsysadmin', 'r/pihole', 'r/protonmail', 'r/tplink_omada', 'r/wifi', 'race', 'rachio', 'rackspace', 'radio', 'radware', 'rajiv', 'rakuten', 'randal', 'random', 'ransomware', 'rapidapi', 'rapiddns', 'raptor', 'raspberry', 'ravelin', 'rbc', 'rbs', 'rclone', 'react-native-firebase', 'read', 'read&write', 'readynas', 'red', 'reddit', 'redmi', 'refinedet:', 'reflectoring', 'regular', 'remke', 'reolink', 'report', 'republic', 'researchgate', 'resource', 'respeaker', 'response', 'reuters', 'reverseaccess', 'reversing', 'revir', 'revision', 'rexing', 'rezolt', 'rezolute', 'rezolute,', 'rfc', 'rfc6763', 'rh-mongodb36', 'rhinoceros', 'richard', 'ring', 'ringtone', 'riskified', 'riskiq', "rixstep's", 'rmc', 'robbshop', 'robert', 'robo', 'roborock', 'roborock-vacuum-s6', 'robtex', 'rock', 'rockville', 'rogers', 'roi4cio', 'roku', 'roku,', 'roma', 'romwebclient', 'roomba', 'rpm', 'rpsv4', 'rslonline', 'rtl', 'rtsak', 'rturn', 'ru', 'rue', 'ruelala', 'runkeeper', 'runzero', 'russell', 'russian', 'rustez', 'rx&d', 'rytilahti/python-miio', 's-trust', 's3', 's3client', 'saaspass', 'sabnzbd', 'safari', 'safety', 'sales', 'salesforce', 'salmonsec', 'sam', 'samboy/maradns', 'samsumg', 'samsung', "samsung's", 'san', 'sanjana4011/ksoap2-android', 'sans', 'sap', 'sapito', 'sast', 'satellite', 'sba', 'scam', 'scamadviser', 'scamvoid', 'scanalert', 'scaninfo', 'scanurl', 'schmidt', 'sciences', 'scip', 'scorecard', 'screen', 'scribd', 'script', 'sea', 'searchcode', 'sears', 'seattle', 'seattle,', 'sebastian', 'sec_hhp_[tv]', 'secret', 'sectigo', 'secure', 'securepubads', 'securifi', 'security', 'securitytrails', 'securiwiki', 'seedonk', 'segment', 'selling', 'seminar', 'seminarsonly', 'sengled', 'sense', 'sensible', 'sensus12', 'sentry', 'seo', 'sequans', 'serious', 'sermat', 'server', 'server:windows', 'servercentral', 'serviio', 'settings', 'sevro', 'sf', 'sfr', 'shadowwhisperer', 'shb', 'shenzhen', 'sheraton', 'shhh-cretly', 'shi', 'shift4shop', 'shodan', 'shopback', 'shopify', 'showtime', 'shutterfly', 'shutterstock', 'si(100)', 'sigma-aldrich', 'signal', 'signal-desktop', 'signify', 'silk', 'similarweb', 'simon', 'simple', 'siri', 'sister', 'sister2sister', 'sitejabber', 'sk', 'skill', 'skylight', 'skype', 'skyzoo', 'sleep', 'sleepproxyclient', 'sleepproxyserver', 'slideshare', 'sm-g610f', 'sm-g930p', 'sm-g950u', 'sm-j260az', 'sm-t380', 'sm-t580', 'small', 'smart', 'smart-tv', 'smartfrog&canary', 'smartlabs', 'smartrecruiters', 'smartscout', 'smartthings', 'smartthings-dev', 'smartthingsoh3', 'smugmug', 'snap', 'snyk', 'soap', 'social', 'socialgrep', 'softscheck', 'sohdms', 'solarwinds', 'solve', 'solved:', 'solvvy', 'sonicwall', 'sonnar', 'sonos', 'sonos,', 'sonoszp', 'sony', 'sophos', 'soul', 'sound', 'soundcloud', 'soundhound', 'soundtouch', 'soundtouch®', 'soundtouch™', 'sovereign', 'spark', 'spectrum', 'speech', 'speechify', 'speedtouch', 'sphere', 'spiceworks', 'spiritfarer', 'sponsorships', 'sportsmax', 'spotify', 'spotify_eliminate_advertisements', 'sprint', 'spyhunter', 'spying', 'spywarecure', 'sql', 'sqlshack', 'squid', 'ssl', 'ssl-tools', 'ssl/tls', 'ssltrust', 'ssofootball', 'stack', 'stackblitz', 'stackshare', 'standardizing', 'star', 'starck', 'starzsoft', 'state', 'statistics', 'steam', 'stefan', 'stendhal', 'stephan', 'steve', 'stevenblack/hosts', 'stitcher', 'stmicroelectronics', 'stores', 'storj', 'streamdns', 'streamtheworld', 'streamyard', 'stuart', 'su', 'sub', 'subdomain', 'super', 'supercharged', 'supercowpowers', 'superior', 'supported', 'supporting', 'suppress', 'surgemail', 'surgeweb', 'suse', 'sustainable', 'suwon', 'suzuki', 'swann', 'swar', 'switcher', 'symantec', 'symc', 'sync', 'syneos', 'synergetic', 'synology', 'sysntpd', 'system', 'systemd-timesyncd', 'systemtek', 't-mobile', 'tab-tv', 'tailscale', 'tailwind', 'taiwan', 'taiyo', 'tanning', 'tarlab', 'tatyana', 'tax', 'tbd', 'tbs', 'tcl', 'tcpdump/libpcap', 'tcplighting', 'tealium', 'tech', 'techdocs', 'techflow', 'techmeme', 'technical', 'techpro', 'techradar', 'techrepublic', 'techstars', 'techtarget', 'techy', 'teejlab', 'telefonica', 'telerik', 'tenable', 'tenable,', 'tenable®', 'terminal', 'terraform', 'test', 'texas', 'texstudio', 'textdoc', 'tf-login-button', 'thaweatherman', 'thawte', 'thawte,', 'the', 'theblocklist', 'themeforest', 'theworld', 'thinksystem', 'thinkware', 'thomas', 'thorax-web', 'threat', 'threatcrowd', 'thunderbird', 'ti', 'tiktok', 'time', 'timetools', 'tinman', 'tinnitus', 'tizen', 'tlc', 'togo', "tom's", 'tomtom', 'tony', 'top', 'touchdown', 'tourism', 'towerdata', 'town', 'tp', 'tp-link', 'tpcamera', 'tplink_omada', 'tracker', 'trackr', 'traefik', 'transmission', 'trend', 'triage', 'trident', 'tripadvisor', 'troubleshooting', 'trustasia', 'trustpilot', 'trustscam', 'ttrh-700', 'tubes', 'tumblr', 'tunein', 'tunesremote', 'turicas', 'turn', 'turner', 'turnip', 'tuxfamily', 'tuya', 'tv', 'tvos', 'tvsbook', 'twilio', 'twitter', 'twonky', 'txn', 'type', 'typeform', "typeform's", 'typekit', 'uav', 'uber', 'ubiquiti', 'ubiquiti-discovery', 'ubuntu', 'ucb', 'udap', 'udm', 'uelfbaby', 'ufrn', 'ui', 'uidivstats', 'uio', 'umbrella', 'ums', 'unagi', 'unblock', 'unbound', 'uncle', 'unearthing', 'unifi', 'united', 'universal', 'universalmediaserver', 'university', 'unizeto', 'unms', 'unofficial', 'unsplash', 'upland', 'upload', 'uploader', 'upnp', 'urban', 'url', 'urlvoid', 'urrea', 'us', 'us-environmental', 'usa', 'usatf', 'uscis', 'usenix', 'utm', 'uw', 'v', 'vaguefield', 'valkyrie', 'valle', 'valve', 'vbulletin', 'vedbex', 'veille', 'velux', 'venturi', 'vera', 'veraalerts', 'vera™', 'verein', 'verisign', 'verizon', 'version', 'vico', "victoria's", 'videojs', 'village', 'vimeo', 'vincentkenny01', 'vindicosuite', 'virustotal', 'visa', 'vision', 'visual', 'visuino', 'vizbee', 'vk', 'vlc', 'vmb3010', 'vmware', 'vnutz', 'vocolinc', 'vodafone', 'volt', 'volumio', 'vorstand', 'voting', 'voxox', 'vtuner', 'vudu', 'vym', 'vysshaya', 'w3', 'w3docs', 'w3schools', 'wac564', 'wailers', 'wailerswl', 'wakelet', 'wall-smart', 'walmart', 'walnuts', 'walt', 'wansview', 'warner', 'warnermedia', 'warnermedia,', 'washington', 'watanabes', 'watchguard', 'wayfair', 'wb', 'wc9500', 'wcasd', 'wd', 'wdde', 'web', 'webflow', 'webinspect', 'webkit', 'webpage', 'webroot', 'websec', 'website', 'weintour-cattolica', 'wells', 'wemo', 'wener', 'west', 'western', 'wexted', 'wextractor', 'wget', 'what', 'whatsapp', 'whois', 'whud-fm', 'wi-fi', 'wikihow', 'wikileaks', 'wikimedia', 'wikipedia', 'wiktionary', 'wild', 'williams', 'win32', 'winamp', 'windows', 'windowsspyblocker', 'wink', 'wink-connected', 'winkapp', 'winscp', 'winters', 'wipliance', 'wired', 'wireless', 'wireshark', 'wispr', 'withings', 'wix', 'wl', 'wlc', 'wmsdk', 'wn802t-200', 'wodsee', 'wolt', 'wolttigroup', 'works', 'workspace', 'world', 'wosign', 'wot', 'wowza', 'wp', 'wrvo', 'wsa', 'wsdapi', 'wsp', 'wtf', 'wu', 'wwv', 'wwv,', 'wyndham', 'wytwornia', 'wyze', 'wyzecam', 'x-force', 'xbox', 'xcode-maven-plugin', 'xda', 'xen', 'xenforo', 'xenserver', 'xfinity', 'xiaomi', 'xiaomi_account_quick_login', 'xiaoyi', 'xiarch', 'xieng', 'xively', 'xl', 'xs4all', 'xtm', 'yabrowser', 'yahoo', 'yahoo!', 'yale', 'yamaha', 'yandex', 'yeelights', 'yi', 'yoast', 'yonkers', 'youtube', 'yowser', 'yr', 'yue', 'yumpu', 'zalando', 'zanies', 'zathura-ps', 'zayo', 'zebra', 'zell', 'zembra', 'zendesk', 'zenodo', 'zero', 'zeroconf', 'zhejiang', 'zheqogzh', 'zhimi-airmonitor-v1', 'zmoviles', 'zoho', 'zoom', 'zoominfo', 'zowee', 'εxodus', 'вмешательства', '上海交通大学', '东南大学', '北京大学', '北京大学国内常用ntp服务器地址及ip', '北京邮电大学', '北京锐讯灵通科技有限公司', '小米开放平台', '清华大学']
acquired_by_gpt_vendors_list_gpt_filtered = [
    '360se', '51degrees', 'abbyy', 'accenture', 'acme', 'activeperl', 'activestate', 'acura', 'adobe', 'adt', 'afp', 'ags', 'ai-thinker', 'airplay', 'airport', 'airprint', 'akamai', 'alibaba', 'alienvault', 'aliyun', 'almond', 'amazon', 'amazonfiretv', 'amzn/selling-partner-api-docs', 'android', 'annapurna', 'apache', 'apple', 'apple-austin', 'apple-dns', 'apple-engineering', 'applecache', 'applecoremedia', 'appletv', 'appletv3', 'appletv3,1', 'appletv5', 'appletv5,3', 'applewatch', 'applewebkit', 'aruba', 'asus', 'atheros', 'atid', 'avahi', 'avahi-daemon', 'avahi-resolve', 'avast', 'avg', 'azure', 'azurewave', 'baidu', 'barracuda', 'belkin', 'bing', 'bitdefender', 'blink', 'blocklist', 'bose', 'brother', 'c4', 'cablelabs', 'canon', 'cisco', 'citrix', 'cloudflare', 'cloudfront', 'cmybabee', 'cnpbagwell', 'comodo', 'control4', 'crestron', 'criteo', 'd-link', 'dell', 'denon', 'devolo', 'digi', 'digicert', 'dlink', 'dnsmasq', 'docker', 'dropcam', 'dudek', 'dyn', 'eero', 'elv/eq-3', 'espressif', 'f5', 'facebook', 'faleemi', 'fandango', 'fandangonow', 'firefox', 'fitbit', 'fortiproxy', 'fortisiem', 'fox', 'fujifilm', 'gac', 'gainspan', 'gecko', 'gmedia', 'gnu', 'go', 'google', 'google-cloud-platform', 'googlecast', 'googlesyndication', 'gw', 'hawking', 'hdiotcamera', 'hikvision', 'hmrc', 'hp', 'hplip', 'hs100', 'hs110', 'hs200', 'huawei', 'hubitat', 'hue', 'hyjoy', 'ibm', 'icloud', 'intel', 'integra', 'intelbras', 'iot', 'ipfire', 'ipod', 'irobot', 'isc', 'isp', 'itunes', 'jbl', 'jenkins', 'joomla', 'juniper', 'kasa', 'kaspersky', 'kerio', 'kindle', 'kodi', 'koogeek', 'kubernetes', 'kyocera', 'lambdatek', 'latch', 'lenovo', 'lg', 'libavcodec', 'libavformat', 'lifx', 'link', 'linkedin', 'linksys', 'linux', 'lite-on', 'liteon', 'localcam', 'logitech', 'logmein', 'lutron', 'mac', 'macos', 'magento', 'magicjack', 'magnepan', 'mailchimp', 'malwarebytes', 'maradns', 'marvell', 'mastodon', 'mcafee', 'mediawiki', 'meraki', 'meross', 'microsoft', 'mikrotik', 'miro', 'microsoftedge', 'microsoft-windows', 'microsoft-windowsmediaplayer', 'mikrotic', 'minecraft', 'miui', 'mjpg-streamer', 'mongodb', 'mozilla', 'mqtt', 'msdn', 'mysql', 'mydlink', 'myq', 'nanoleaf', 'nas', 'nasa', 'nba', 'nbc', 'netatmo', 'netflix', 'netgear', 'nginx', 'nikon', 'nintendo', 'nmap', 'node', 'node-red', 'nokia', 'nordvpn', 'norton', 'npr', 'nvidia', 'office', 'okhttp', 'okta', 'olivier', 'onkyo', 'openbsd', 'opendns', 'openhab', 'openhome', 'openssl', 'openvpn', 'openwrt', 'opera', 'oracle', 'orbi', 'osram', 'owncloud', 'paloaltonetworks', 'panasonic', 'pandora', 'parrot', 'pfsense', 'philips', 'pioneer', 'plex', 'plume', 'polycom', 'postfix', 'postman', 'powershell', 'privoxy', 'proxmox', 'python', 'qnap', 'qualcomm', 'quantum', 'quasar', 'quicktime', 'rachio', 'raspberry', 'raspbian', 'realtek', 'redhat', 'ring', 'roku', 'roon', 'rosetta', 'rosetta@home', 'ruckus', 'safari', 'samsung', 'sangoma', 'sanyo', 'sensibo', 'sentry', 'shairport', 'shairport-sync', 'sharp', 'shodan', 'shure', 'siemens', 'simplisafe', 'siri', 'slack', 'sling', 'slingbox', 'slingmedia', 'slingtv', 'smartthings', 'snap', 'snapchat', 'snom', 'sonicwall', 'sonos', 'sony', 'sophos', 'soundcloud', 'spotify', 'sprint', 'sricam', 'ssl', 'starbucks', 'startpage', 'steam', 'synology', 't-mobile', 'tado', 'tasmota', 'teamviewer', 'telstra', 'tenda', 'tencent', 'thomson', 'tivo', 'tplink', 'trendnet', 'tripleplay', 'tunein', 'twilio', 'twitter', 'ubiquiti', 'ubnt', 'ubuntu', 'ue', 'ultravnc', 'unifi', 'univision', 'upnp', 'ups', 'usps', 'vagrant', 'vbulletin', 'verisign', 'verizon', 'viber', 'vimeo', 'virtualbox', 'vizio', 'vmware', 'vodafone', 'vpn', 'vudu', 'vultr', 'w3', 'walmart', 'wd', 'webex', 'webos', 'webrtc', 'whatsapp', 'wikipedia', 'windows', 'windows-media-player', 'windowsphone', 'winpcap', 'wireshark', 'wistia', 'wlan', 'wmp', 'wordpress', 'wpad', 'wrt', 'wunderground', 'wyze', 'x10', 'xbox', 'xiaomi', 'xilinx', 'xiongmai', 'xirrus', 'yahoo', 'yamaha', 'yandex', 'yeelight', 'yelp', 'youtube', 'zabbix', 'zattoo', 'zebra', 'zendesk', 'zmodo', 'zoho', 'zoom', 'zwave', 'zyxel'
]
acquired_by_gpt_vendors_list_gpt_filtered_one_by_one = ['@amazon', 'abbott', 'acme', 'adt', 'ai-thinker', 'alexa', 'alibaba', 'aliyun', 'amazon', 'amazonfiretv', 'ampak', 'annapurna', 'apple', "apple's", 'apple-austin', 'apple-dns', 'apple-engineering', 'apple_team_id', 'applecache', 'applecoremedia', 'appletv', 'appletv3', 'appletv3,1', 'appletv5', 'appletv5,3', 'applewatch', 'applewebkit', 'appstore', 'arduino', 'arlo', 'aruba', 'asus', 'atheros', 'avast', 'avg', 'avira', 'aws', 'azure', 'azurewave', 'barracuda', 'belkin', 'bigtreetech', 'bitdefender', 'blink', 'bose', 'brother', 'cablelabs', 'callpod', 'canary', 'canon', 'cc3100', 'cc3200', 'centurylink', 'chamberlain', 'chromecast', 'chromecasts', 'cisco', 'citrix', 'cloudflare', 'cloudflare,', 'colasoft', 'control4', 'crestron', 'cyberlink', 'd-link', "d-link's", 'dcs-5030l', 'dell', 'dericam', 'dlink', 'dropcam', 'dropcam™', 'echo', 'ecobee', 'edimax', 'eero', 'eq-3', 'ericsson', 'esp32-cam', 'espressif', 'evrythng', 'ezlo', 'f5', 'faleemi', 'firewalla', 'flex', 'flic', 'ford', 'fortiproxy', 'fortisiem', 'freertos', 'gainspan', 'genesys', 'google', 'googlecast', 'haier', 'harman', 'hawking', 'hewlett-packard', 'homeassistant', 'homebridge', 'homematic', 'homeseer', 'hp', 'huawei', 'hubitat', 'hue', 'ibm', 'ifttt', 'ihome', 'ikea', 'immedia', 'immedia-semi', 'insignia', 'insteon', 'intel', 'invoxia', 'iptime', 'irobot', 'kasa', 'kindle', 'kodak', 'koogeek', 'kpn', 'laserfiche', 'lastline', 'lb-link', 'leica', 'lenovo', 'lg', 'liftmaster', 'lifx', 'lightify', 'linksys', 'lite-on', 'liteon', 'logitech', 'lutron', 'lutronpro', 'marvell', 'mecool', 'mibox', 'micloudconnector', 'microsoft', 'mikrotik', 'mio', 'mios', 'mydlink', 'myq', 'mysimplelink', 'nanoleaf', 'nest', "nest's", 'newegg', 'nighthawk', 'nintendo', 'nixplay', 'nodemcu', 'nokia', 'norton™', 'nutanix', 'nvidia', 'nxp', "nxp's", 'odm', 'oem', 'oem/odm', 'okta', 'omada', 'oneplus', 'openhab', 'opentrons', 'openwrt', 'opnsense', 'oppo', 'orange', 'orbi', 'osram', 'parrot', 'philips', 'phillips', 'piper', 'pix-star', 'pixstar', 'postlund/dlink_hnap', 'qnap', 'rachio', 'radware', 'raspberry', 'redmi', 'reolink', 'respeaker', 'rexing', 'ring', 'roborock', 'roborock-vacuum-s6', 'roku', 'roku,', 'roomba', 'samsung', "samsung's", 'sap', 'securifi', 'sengled', 'sequans', 'sfr', 'shenzhen', 'signify', 'smartfrog&canary', 'smartlabs', 'smartthings', 'solarwinds', 'sonicwall', 'sonos', 'sony', 'sophos', 'speedtouch', 'sprint', 'stmicroelectronics', 'suse', 'swann', 'symantec', 'synology', 'tcl', 'tcplighting', 'telefonica', 'thinkware', 'tizen', 'tomtom', 'tp-link', 'tplink_omada', 'trackr', 'tuya', 'ubiquiti', 'unifi', 'velux', 'vera', 'vera™', 'verizon', 'vmware', 'vocolinc', 'vodafone', 'volumio', 'wansview', 'watchguard', 'wemo', 'wink', 'wink-connected', 'winkapp', 'withings', 'wodsee', 'wyze', 'wyzecam', 'xiaomi', 'xiaoyi', 'xively', 'yale', 'yamaha', 'yeelights', 'yi', 'zebra', 'zhejiang', '北京锐讯灵通科技有限公司', '小米开放平台']


#filtered_gpt_acquired_list = ['amazon', 'apple', 'asus', 'belkin', 'cisco', 'd-link', 'dell', 'google', 'hp', 'huawei', 'intel', 'lenovo', 'lg', 'linksys', 'logitech', 'microsoft', 'netgear', 'nokia', 'nvidia', 'philips', 'samsung', 'sony', 'tp-link', 'xiaomi', 'acer', 'adobe', 'alibaba', 'amd', 'arduino', 'canon', 'dropbox', 'epson', 'fitbit', 'garmin', 'gopro', 'gigabyte', 'ibm', 'kodak', 'lexmark', 'mcafee', 'motorola', 'nest', 'nikon', 'nintendo', 'norton', 'panasonic', 'qualcomm', 'raspberry pi', 'roku', 'seagate', 'sharp', 'siemens', 'symantec', 'toshiba', 'vizio', 'western digital', 'xerox', 'yahoo', 'zte', 'zyxel', 'brother', 'fujifilm', 'kaspersky', 'kingston', 'kaspersky', 'netflix', 'oppo', 'oracle', 'pandora', 'realtek', 'redhat', 'spotify', 'square', 't-mobile', 'tenda', 'ubisoft', 'ubiquiti', 'verizon', 'vmware', 'whatsapp', 'yamaha', 'zoom']

dict_vendor_stringmatching_results, file_path = read_dict_results_file(
    "/Users/barmey/Dropbox/IoT-Meyuhas/IoT_lab/classification_results/9dc6a30d9289c3f7553aa21660f2b93c.pkl")
def decode_and_decompress(filename):

    return None