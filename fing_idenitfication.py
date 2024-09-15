import requests
import json
import utils
def device_fing_request_query(mac_addr,dhcp_hostname,user_agents_array,dhcp_params=None):

    headers = {
        'Content-Type': 'application/json',
        'X-Api-Key': 'JHN0dWRlbi0tLXJlY2lobWFuLXVuaS1kZXZyZWNvZy10cmlhbBVTdHVkZW4gLSBSZWNpaG1hbiBVbmktc3R1ZGVuLS0tcmVjaWhtYW4tdW5pLWRldnJlY29nLXRyaWFsLWRldnJlY29nAAABjTeAqxgNkmOM',
    }
    user_agents_dict =  {"userAgent":user_agent for user_agent in user_agents_array}

    data = {
        "devices": [
            {
                "mac": mac_addr,
                "dhcp": {
                    "hostname": dhcp_hostname
                },
                "hua":[
                               str(user_agents_dict)
                           ],
            }
        ]
    }
    response = requests.post('https://service.fing.com/3/devrecog', headers=headers, data=json.dumps(data))
    json_resp = json.loads(response.text)
    if 'recognition' not in json_resp['devices'][0]:
        return '','',''
    try:
        vendor = json_resp['devices'][0]['recognition']['brand']
    except:
        if 'mac-vendor' not in json_resp['devices'][0]['recognition']:
            vendor = ''
        else:
            vendor = json_resp['devices'][0]['recognition']['mac-vendor']
    if 'type' in json_resp['devices'][0]['recognition']:
        function = json_resp['devices'][0]['recognition']['type']
    else:
        function = ''
    conf = json_resp['devices'][0]['recognition']['rank']
    return vendor,function,conf
#print(response)
def device_fing_query(device_dict,fields_to_search):
    dict_results = utils.read_pkl_file(utils.pkl_file_fing_results)
    key = device_dict['mac_address']+'_'+str(fields_to_search) + '_fing'
    if key not in dict_results:
        response = device_fing_request_query(mac_addr=device_dict['mac_address'],dhcp_hostname=device_dict[utils.hostname_field],user_agents_array=device_dict[utils.user_agents_field])
        dict_results[key] = response
        utils.save_dict_to_file_pkl(dict_results, utils.pkl_file_fing_results)
    return dict_results[key]


#device_fing_request_query(mac_addr='74:da:38:23:22:7b',dhcp_hostname='',user_agents_array=["CyberGarage-HTTP/1.0", "Wemo/491000 CFNetwork/811.5.4 Darwin/16.6.0"])