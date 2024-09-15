import openai
import pandas as pd

import data_augmentaion
import secrets
import json

import utils

openai.organization = secrets.org_id_openai
openai.api_key = secrets.api_key_openai
default_model = 'gpt-3.5-turbo'
default_prompt = "Given the information provided by the user and extracted info from the internet, please classify the following IoT devices into device name, Vendor and Type,\
         Please provide a confidence score for every type/vendor. If there are several options provide only first two with highest confidence level .\
        The device name is official name of the device such as Amazon Alexa.\
        A vendor could be the name of the related company such as TP-Link, Amazon, Google, Samsung.\
        The type could be  Television, Plug,Hub, Router, etc.\
        The confidence score is a number between 0% - 100%\
        Please add Explainability (max. 50 words) for you choice.\
        Your answer should be in a valid JSON format,up to 3 results, each result is your classification for device name, vendor,type, confidence and explainability \
        Dont add any further information before or after your answer"
def make_request_openai(dict_fields,model,prompt):
    response = openai.ChatCompletion.create(
    model=model,
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": str(dict_fields)},
    ],
      temperature=0.2,
      max_tokens=1024,
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0)
    return response


def retrieve_gpt_classification_device_by_fields(device_dict,fields_to_search,list,label,model,prompt,uselist=False):
    dict_results = utils.read_pkl_file(utils.pkl_file_gpt_results)
    dict_fields_to_classify_by = {x:device_dict[x] for x in fields_to_search}
    if uselist:
        with open(utils.vendor_type_list_chat_gpt4_path, 'r') as f:
            dict_vendor_type = json.load(f)
            dict_fields_to_classify_by['dictionary_vendor_device_types'] = dict_vendor_type
    dict_fields_to_classify_by[f'list_of_{label}'] = list
    if len(str(dict_fields_to_classify_by)) > 10000:
        if utils.full_uri_field in dict_fields_to_classify_by:
            shortend_urls = []
            for url in device_dict[utils.full_uri_field]:
                shortend_urls.append(utils.smart_shorten_url(url,25))
            shortend_urls = pd.array(shortend_urls).unique()
            dict_fields_to_classify_by[utils.full_uri_field] = shortend_urls
            if len(str(dict_fields_to_classify_by)) > 10000:
                dict_fields_to_classify_by[utils.full_uri_field] = dict_fields_to_classify_by[utils.full_uri_field][:10]
    key = str(dict_fields_to_classify_by) + model + prompt
    if key not in dict_results:
        response = make_request_openai(dict_fields_to_classify_by,model,prompt)
        dict_results[key] = response
        utils.save_dict_to_file_pkl(dict_results, utils.pkl_file_gpt_results)
    return dict_results[key]

def split_string_to_json(message):
    # Split the input string into lines
    lines = message.strip().split('\n')

    # Extract the JSON strings from each line
    json_strings = [line[line.index('{'):] for line in lines]

    # Convert JSON strings to dictionaries and store them in a list
    devices = [json.loads(json_string) for json_string in json_strings]

    # Convert the list of dictionaries to a JSON string
    devices_json = json.dumps(devices, indent=2)

    return devices_json

def classify_device_by_fields_gpt(device_dict,fields_to_search,model=default_model,prompt=default_prompt,uselist=False):
    if uselist:
        resp = retrieve_gpt_classification_device_by_fields(device_dict,fields_to_search,model,prompt,uselist)
    else:
        resp = retrieve_gpt_classification_device_by_fields(device_dict,fields_to_search,model,prompt)
    answer = resp['choices'][0]['message']['content']
    if answer[0] == '1':
        answer = split_string_to_json(answer)
    if answer[0]=='{':
        answer = answer[:answer.rfind('}')+1]
    elif answer[0] == '[':
        answer = answer[:answer.rfind(']')+1]
    dict_classification = json.loads(answer)
    if type(dict_classification) == type(list()):
        dict_classification = dict(zip(range(len(dict_classification)),dict_classification))
    dict_results = {'vendor':device_dict['vendor'],'type':device_dict['type'],'classified_gpt_vector':dict_classification}
    print(dict_results)
    return dict_results

def classify_device_by_fields_and_lists_gpt(device_dict,fields_to_search,list_labels,label,model=default_model,prompt=default_prompt):
    resp = retrieve_gpt_classification_device_by_fields(device_dict,fields_to_search,list_labels,label,model,prompt)
    answer = resp['choices'][0]['message']['content']
    if answer[0] == '1':
        answer = split_string_to_json(answer)
    if answer[0]=='{':
        answer = answer[:answer.rfind('}')+1]
    elif answer[0] == '[':
        answer = answer[:answer.rfind(']')+1]
    answer = validate_and_fix_json(answer)
    dict_classification = json.loads(answer)
    if type(dict_classification) == type(list()):
        dict_classification = dict(zip(range(len(dict_classification)),dict_classification))
    if label == 'types and vendor':
        dict_results = {'vendor':device_dict['vendor'],'type':device_dict['type'],'classified_gpt_vector':dict_classification}
    else:
        dict_results = {label:device_dict[label],'classified_gpt_vector':dict_classification}

    print(dict_classification)
    return dict_results

#sometimes gpt return json in a format with 'results' this remove it.
if True:
    dict_res = utils.read_pkl_file(utils.pkl_file_gpt_results)
    for resp in dict_res:
        answer = dict_res[resp]['choices'][0]['message']['content']
        if answer[0] == '1':
            answer = split_string_to_json(answer)
        if answer[0] == '{':
            answer = answer[:answer.rfind('}') + 1]
        elif answer[0] == '[':
            answer = answer[:answer.rfind(']') + 1]
        try:
            dict_classification = json.loads(answer)
        except:
            print(resp)
        if type(dict_classification) == type(list()):
            dict_classification = dict(zip(range(len(dict_classification)), dict_classification))
        if 'result' in dict_classification:
            print(1)
            for index in range(len(dict_classification['result'])):
                dict_classification[str(index)]=dict_classification['result'][index]
            del dict_classification['result']
            dict_res[resp]['choices'][0]['message']['content'] = json.dumps(dict_classification)
            utils.save_dict_to_file_pkl(dict_res,utils.pkl_file_gpt_results)
        if 'results' in dict_classification:
            print(1)
            for index in range(len(dict_classification['results'])):
                dict_classification[str(index)]=dict_classification['results'][index]
            del dict_classification['results']
            dict_res[resp]['choices'][0]['message']['content'] = json.dumps(dict_classification)
            utils.save_dict_to_file_pkl(dict_res,utils.pkl_file_gpt_results)

def validate_and_fix_json(json_string):
    try:
        # Try to load the JSON string
        data = json.loads(json_string)
        #print("The JSON is valid.")
        return json_string
    except json.JSONDecodeError as e:
        print("The JSON is not valid. Trying to fix...")
        # If it fails, try to add the missing closing brace
        try:
            fixed_json_string = json_string + "}"
            data = json.loads(fixed_json_string)
            print("The JSON is now valid.")
            return fixed_json_string
        except json.JSONDecodeError:
            try:
                fixed_json_string = '['+json_string + "]"
                data = json.loads(fixed_json_string)
                print("The JSON is now valid.")
                return fixed_json_string
            except json.JSONDecodeError:
                print("Couldn't fix the JSON.")
            return None