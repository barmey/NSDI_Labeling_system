from collections import Counter

import numpy
import re
from difflib import SequenceMatcher
import utils

def count_word_in_text_from_list(device, fields, words_list,threshold):
    dict_results = utils.read_pkl_file(utils.pkl_file_string_matching_counter)
    key = str(device['mac_address']) + '_' + utils.generate_filename_from_hash(str(words_list)) + '_' + str(threshold)
    vendor_field_count = {}
    has_changes = False
    if key not in dict_results:
        dict_results[key] = {}
        has_changes = True
    for field in fields:
        if field in dict_results[key]:
            vendor_field_count[field]=dict_results[key][field]
        else:
            if len(device[field]) > 0:
                vendor_field_count[field] = count_words_per_word(device[field], words_list, threshold)
                dict_results[key][field] = vendor_field_count[field]
                has_changes = True
    if has_changes:
        utils.save_dict_to_file_pkl(dict_results,utils.pkl_file_string_matching_counter)
    return vendor_field_count


def remove_words(list):
    to_remove = ['net','com','org','il','cn','co','gov','']
    for remove in to_remove:
        list = [i for i in list if i != remove]

    return list


def count_words_per_word(text, list_words,threshold):
    splited_text = re.split(r"[-;,.\s]\s*",' '.join(text))
    splited_text = remove_words(splited_text)
    words_count = {}
    for word in list_words:
        count = 0
        if ' ' in word or '-' in word:
            for sentence in text:
                if word.lower() in sentence.lower():
                    count = count + sentence.lower().count(word.lower())
        else:
            for wordi in splited_text:
                if match_strings(word, wordi, threshold):
                    count = count + 1
        words_count[word] = count
    return words_count


def calculate_vector_counter_word_in_text_equal_fields(device, fields, words,threshold_string_matching=0.95):
    dict_field_vendor_count = count_word_in_text_from_list(device,fields,words,threshold_string_matching)
    counter = Counter()
    for dictionary in dict_field_vendor_count.values():
        counter = counter + Counter(dictionary)
    return 'unknown' if len(counter.most_common()) == 0 else counter.most_common()

def calculate_vector_counter_word_in_text_fields_weighting(device, fields, words,data_fields,threshold_string_matching=0.95):
    dict_field_vendor_count = count_word_in_text_from_list(device,fields,words,threshold_string_matching)
    counter = Counter()
    for field, vendors in dict_field_vendor_count.items():
        field_name = field
        if 'filtered_' in field:
            field_name = "_".join(field.split("_")[:-2])
        if field_name in data_fields:
            weight = data_fields[field_name]["weight"]
        else:
            weight = data_fields[utils.field_to_enriched_field_dict[field_name]]["weight"]
        for vendor, count in vendors.items():
            counter[vendor] += count * weight
    filtered_counter = Counter({k: v for k, v in counter.items() if v != 0})
    return 'unknown' if len(filtered_counter.most_common()) == 0 else filtered_counter.most_common()


def find_vendor_in_text_hostname_prefered_fields(device, fields, vendors):
    return

def match_strings(vendor,word,threshold):
    if threshold == 1:
        return vendor.lower() == word.lower() or vendor.lower() + 's' == word.lower()
    return SequenceMatcher(None, vendor.lower(), word.lower()).ratio() >= threshold
