from collections import Counter
from statistics import mean

import numpy as np

import utils


def classify_text_to_word(text, vector_words, classifier):
    candidate_labels = vector_words
    classified_word_vec = [('unknown', 0)]
    if text != '':
        classified_word_vec = classifier(text, candidate_labels)
    return classified_word_vec
def classify_text_to_word_efficently(text, vector_words, classifier):
    candidate_labels = vector_words
    already_classified = utils.read_pkl_file(utils.pkl_file_classifications_sentences)
    change = False
    if text != '':
        classified_vector = []
        for sentence in text:
            key = (sentence, str(candidate_labels),classifier.model.base_model_prefix)
            if key not in already_classified:
                classified_word_vec = classifier(sentence, candidate_labels)
                already_classified[key]=classified_word_vec
                classified_vector.append(classified_word_vec)
                change = True
            else:
                classified_vector.append(already_classified[key])
    else:
        classified_vector = [('unknown', 0)]
    if change:
        utils.save_dict_to_file_pkl(already_classified,utils.pkl_file_classifications_sentences)
    return classified_vector


def calculate_zeroshot_labels_in_text_equal_fields(device, fields, vector_words, classifier):
    text = []
    for field in fields:
        text = text + device[field]
    return 'unknown' if len(text) == 0 else classify_text_to_word_efficently(text,vector_words,classifier)

def zeroshot_classification_equal_fields(device, fields, labels, classifier,data_fields):
    classification_list = calculate_zeroshot_labels_in_text_equal_fields(device, fields, labels, classifier)
    list_labels = []
    conf_list = {}
    if classification_list == 'unknown':
        return []
    for desc in classification_list:
        for score,label in zip(desc['scores'],desc['labels']):
            if score <= 0.5:
                continue
            list_labels.append(label)
            conf_list.setdefault(label, list())
            conf_list[label].append(score)
    result_dict = []
    for label in Counter(list_labels).most_common():
        result_dict.append((label[0], mean(conf_list[label[0]]), label[1]))
    result_dict.sort(key=lambda ele: ele[1],reverse=True)
    return result_dict

def zeroshot_classification_field_query_balance(device, fields, labels, classifier,fields_data):

    classification_list = calculate_zeroshot_labels_in_text_equal_fields(device, fields, labels, classifier)
    list_labels = []
    conf_list = {}
    if classification_list == 'unknown':
        return [],[]
    for desc in classification_list:
        query, field = get_query_field(device, desc['sequence'])
        if field in fields_data:
            threshold = fields_data[field]["threshold"]
        else:
            threshold = fields_data[utils.field_to_enriched_field_dict[field]]["threshold"]
        for score,label in zip(desc['scores'],desc['labels']):
            if score <= threshold:
                continue
            list_labels.append(label)
            amount_of_results = next((int(s.split('_filtered_')[-1]) for s in fields if s.startswith(field) and s.split('_filtered_')[-1].isdigit()), 10)
            amount_of_result_in_query = len(list(filter(lambda x: x[0] == query, device[field])))
            if amount_of_result_in_query != 0:
                amount_of_results = min([amount_of_results,amount_of_result_in_query])
            conf_list.setdefault((label,query,field,amount_of_results), list())
            conf_list[(label,query,field,amount_of_results)].append(score)
    #result_dict = []
    result_dict = calculate_result_dict_divide_and_equals_fields(conf_list,fields_data)

    # Sort the list by frequency (primary key) and average score (secondary key)
    result_dict = sorted(result_dict, key=lambda x: (x[2], x[1]), reverse=True)
    output_table = create_table_for_decision_tree(conf_list)

    return result_dict,output_table

def create_table_for_decision_tree(conf_list):
    return []
    features_dict = utils.enriched_field_to_field_dict.keys()
    features = [key for key in features_dict]
    labels = utils.read_csv_single_column_to_list(utils.types_path)
    labels = [label.lower() for label in labels]
    output_vector = [0] * (len(labels) * len(features))

    for key, value in conf_list.items():
        label,_, feature, _ = key
        label_index = labels.index(label.lower())
        feature_index = features.index(feature)
        value_index = label_index * len(features) + feature_index
        if len(value) > 0:
            average_value = sum(value) / len(value)
            output_vector[value_index] = average_value
        else:
            output_vector[value_index] = 0

    return output_vector

def calculate_result_dict_divide_by_amount_of_results(conf_dict):
    label_data = {}

    # Iterate over dictionary items
    for (label, _, _, amount_of_results), scores in conf_dict.items():
        quotient = len(scores) / amount_of_results
        avg_score = sum(scores) / len(scores) / amount_of_results

        # Accumulate quotient and average score
        if label in label_data:
            label_data[label]['frequency'] += quotient
            label_data[label]['total_score'] += avg_score
            label_data[label]['count'] += 1
        else:
            label_data[label] = {'frequency': quotient, 'total_score': avg_score, 'count': 1}

    # Produce list of tuples from accumulated values
    result = [(label, data['total_score'] / data['count'], data['frequency']) for label, data in label_data.items()]

    return result


def calculate_result_dict_divide_and_equals_fields(conf_dict,fields_data):
    label_field_data = {}

    # Accumulate data per (label, field) combination
    for (label, query, field, amount_of_results), scores in conf_dict.items():
        if (label, field) not in label_field_data:
            label_field_data[(label, field)] = {"total_score": 0, "count": 0, "queries_processed": 0}

        avg_score_for_query = sum(scores) / len(scores)
        label_field_data[(label, field)]["total_score"] += avg_score_for_query
        label_field_data[(label, field)]["count"] += len(scores) / amount_of_results
        label_field_data[(label, field)]["queries_processed"] += 1

    # Aggregate data per label
    label_data = {}
    for (label, field), data in label_field_data.items():
        weight = fields_data.get(field, {}).get("weight", 1)  # default to 1 if no weight specified for the field
        if label not in label_data:
            label_data[label] = {"total_score": 0, "fields_processed": 0, "count": 0}

        # Compute average for the field
        avg_score_for_field = data["total_score"] / data["queries_processed"] * weight
        avg_count_for_field = data["count"] / data["queries_processed"] * weight

        label_data[label]["total_score"] += avg_score_for_field
        label_data[label]["count"] += avg_count_for_field
        label_data[label]["fields_processed"] += 1

    result = []
    for label, data in label_data.items():
        avg_score = data["total_score"]
        avg_count = data["count"]
        result.append((label, avg_score, avg_count))

    # Sort by frequency and then by score
    result.sort(key=lambda x: (x[2], x[1]), reverse=True)

    return result
def get_query_field(device,search_result):
    for field in utils.enriched_field_to_field_dict.keys():
        for result in device[field]:
            if search_result == result[2]:
                return result[0],field
    for field in utils.enriched_field_to_field_dict.values():
        for result in device[field]:
            if search_result == result:
                return search_result, field

def zeroshot_classification_max_confidence(device, fields, labels, classifier,threshold):
    classification_list = calculate_zeroshot_labels_in_text_equal_fields(device, fields, labels, classifier)
    list_labels = []
    conf_list = {}
    if classification_list == 'unknown':
        return []
    for desc in classification_list:
        for score,label in zip(desc['scores'],desc['labels']):
            if score <= threshold:
                continue
            list_labels.append(label)
            conf_list.setdefault(label, list())
            conf_list[label].append(score)
    result_dict = []
    for label in Counter(list_labels).most_common():
        result_dict.append((label[0], max(conf_list[label[0]]), label[1]))
    result_dict.sort(key=lambda ele: ele[1],reverse=True)
    return result_dict


def build_fields_list_for_top_results(input_fields):
    # Create a mapping of fields to their Google equivalents
    field_map = {
        utils.hostname_field: utils.hostnames_google_field,
        utils.dns_field: utils.domains_google_field,
        utils.user_agents_field: utils.user_agents_google_field,
        utils.tls_issuers_field: utils.tls_issuers_google_field
    }

    # Create a list to store the updated fields
    updated_fields = []

    # Iterate over the input fields
    for field in input_fields:
        # If the field has a Google equivalent, add it to the updated fields list
        if field in field_map:
            updated_fields.append(field)
            updated_fields.append(field_map[field])

    # Return the updated fields
    return updated_fields


def get_top10_results_device(device, fields, labels, classifier,threshold):
    fields = build_fields_list_for_top_results(fields)
    classification_list = calculate_zeroshot_labels_in_text_equal_fields(device, fields, labels, classifier)
    list_labels = []
    conf_list = {}
    if classification_list == 'unknown':
        return []
    for desc in classification_list:
        if desc['scores'][0] <= threshold:
            continue
        list_labels.append(desc['labels'][0])
        conf_list.setdefault(desc['labels'][0], list())
        conf_list[desc['labels'][0]].append((desc['scores'][0],desc['sequence']))
    return select_top_sequences(conf_list,n=10)
    result_dict = []
    for label in Counter(list_labels).most_common():
        result_dict.append((label[0], mean([ele[0] for ele in conf_list[label[0]]]), label[1]))
    result_dict.sort(key=lambda ele: ele[2],reverse=True)
    sorted_list_sentences = sorted(list(conf_list[result_dict[0][0]]),key=lambda ele: ele[0], reverse=True)
    top10_sequences = [x[1] for x in sorted_list_sentences[:10]]
    return top10_sequences


def select_top_sequences(data, n):
    # Calculate total number of sequences
    total_sequences = sum(len(values) for values in data.values())

    # Initialize a list to store selected sequences
    selected_sequences = []

    # Keep track of the total sequences added
    sequences_added = 0

    # Iterate over each device type
    for device_type, sequences in data.items():
        # Sort the sequences by confidence level in descending order
        sequences.sort(key=lambda x: x[0], reverse=True)

        # Calculate the number of sequences to select from this list
        num_to_select = round(len(sequences) / total_sequences * n)

        # If adding these sequences would result in more than n total, reduce the number to select
        if sequences_added + num_to_select > n:
            num_to_select = n - sequences_added

        # Select the highest-confidence sequences and add only the sequence part to the list
        selected_sequences.extend([seq for conf, seq in sequences[:num_to_select]])

        # Update the total sequences added
        sequences_added += num_to_select

        # If we've already added n sequences, break the loop
        if sequences_added >= n:
            break

    # Return the selected sequences
    return selected_sequences


def split(list_a, chunk_size):
  for i in range(0, len(list_a), chunk_size):
    yield list_a[i:i + chunk_size]

def calculate_zeroshot_labels_in_text_concat_fields(device,fields, vector_words, classifier):
    text = []
    for field in fields:
        device[field] = list(filter(lambda item: 'ntp' not in item.lower(), device[field]))
        text = text + list(split(device[field], 10))
        #text = text + device[field]
    text = [''.join(x) for x in text]
    return 'unknown' if len(text) == 0 else classify_text_to_word(text,vector_words,classifier)


def zeroshot_classification_concat_fields(device, fields, labels, classifier):
    classification_list = calculate_zeroshot_labels_in_text_concat_fields(device, fields, labels, classifier)
    list_labels = []
    conf_list = {}
    if classification_list == 'unknown':
        return []
    for desc in classification_list:
        if desc['scores'][0] <= 0.14:
            continue
        list_labels.append(desc['labels'][0])
        conf_list.setdefault(desc['labels'][0], list())
        conf_list[desc['labels'][0]].append(desc['scores'][0])
    result_dict = []
    for label in Counter(list_labels).most_common():
        result_dict.append((label[0], mean(conf_list[label[0]]), label[1]))
    result_dict.sort(key=lambda ele: ele[1],reverse=True)
    return result_dict
