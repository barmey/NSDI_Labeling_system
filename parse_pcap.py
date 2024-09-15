import subprocess
from datetime import datetime

import pandas as pd
import os
import utils
import csv
import shutil
from tqdm import tqdm
from OuiLookup import OuiLookup
from dateutil import parser as date_parser
from dateutil import tz

dict_fields =  ["dhcp.option.hostname","dns.ptr.domain_name","http.user_agent","dns.qry.name","tls.handshake.extensions_server_name","x509ce.dNSName","http.request.uri","x509sat.printableString"]
def generate_csvs(inputdir,identifier_source):
    tshark = "/Applications/Wireshark.app/Contents/MacOS/tshark"
    for pcapFilename in tqdm(os.listdir(inputdir)):
        if not (pcapFilename.endswith(".pcapng") or pcapFilename.endswith(".pcap")) or pcapFilename.endswith('NestCamIQ.pcap'):
            continue
        with open(utils.csvs_path + os.path.splitext(pcapFilename)[0] + "_" + identifier_source + ".csv","w+") as outfile:
            subprocess.run([tshark, "-r",
                os.path.join(inputdir, pcapFilename),
                            "-T", "fields", "-e", "eth.src", "-e", "eth.dst",
                            "-e", "frame.time",
                            "-e", "ip.src", "-e", "ip.dst",
                            "-e", "dhcp.option.hostname", "-e", "dns.ptr.domain_name", "-e", "http.user_agent", "-e",
                            "dns.qry.name", "-e",
                            "tls.handshake.extensions_server_name", "-e", "x509ce.dNSName", "-e",
                            "x509sat.printableString","-e", "http.request.full_uri", "-e", "http.request.uri",
                            "-E", "header=y", "-E", "separator=,", "-E",
                            "quote=d", "-E", "occurrence=f"],
                stdout=outfile, check=True)

def get_mac_address(df):
    filtered_df =df[~df['dhcp.option.hostname'].isnull()]
    if len(filtered_df.index) == 0:
        filtered_df = df[~df['dns.ptr.domain_name'].isnull()]
    if len(filtered_df.index) == 0:
        filtered_df = df[~df['http.user_agent'].isnull()]
    if len(filtered_df.index) == 0:
        filtered_df = df[~df['dns.qry.name'].isnull()]
    return filtered_df.iloc[0]['eth.src'] if len(filtered_df.index) > 0 else 'ff:ff:ff:ff:ff:ff'

def csv_to_dict(gapminder_csv_url):
    fields = ["dhcp.option.hostname","dns.ptr.domain_name","http.user_agent","dns.qry.name","tls.handshake.extensions_server_name","x509ce.dNSName", "http.request.uri","http.request.full_uri","x509sat.printableString"]
    record = pd.read_csv(gapminder_csv_url, low_memory=False, on_bad_lines='skip')
    details = {}
    details['mac_address'] = get_mac_address(record)
    details[utils.oui_field] = list(OuiLookup().query(details['mac_address'])[0].values())
    if details[utils.oui_field] == [None]:
        details[utils.oui_field] = []
    details['origin_dataset'] = os.path.splitext(os.path.basename(gapminder_csv_url))[0].split('_')[-1]
    first_record_time = record['frame.time'][0]

    for field in fields:
        if field == 'http.request.full_uri':
            try:
                details[field] = list(record[field].unique())
            except:
                print(gapminder_csv_url)
                details[field] = list()
                continue
        else:
            details[field] = list(record[field].unique())
        details[field] = [x for x in details[field] if type(x) == type('')]
    return details

def parse_time(time_string):
    if time_string.endswith(' IST'):
        time_string = time_string[:-4]  # Remove the ' IST' suffix

    formats = [
        '%b %d, %Y %H:%M:%S.%f %Z',  # Format for most records with timezone
        '%b %d, %Y %H:%M:%S.%f000 %Z',  # Format for records with milliseconds and timezone
        '%b %d, %Y %H:%M:%S.%f %Z',  # Format for records with timezone
        '%b %d, %Y %H:%M:%S.%f000 %Z',  # Format for records with milliseconds and timezone
        '%b %d, %Y %H:%M:%S.%f %Z'  # Format for records with milliseconds and timezone
    ]
    for fmt in formats:
        try:
            parsed_time = datetime.strptime(time_string, fmt)
            return parsed_time
        except ValueError:
            pass
    print(f"Failed to parse {time_string}")
    return datetime.now()  # Use default time if none of the formats match

def csv_to_dict_with_time(gapminder_csv_url):
    fields = ["dhcp.option.hostname","dns.ptr.domain_name","http.user_agent","dns.qry.name","tls.handshake.extensions_server_name","x509ce.dNSName", "http.request.uri","http.request.full_uri","x509sat.printableString"]
    record = pd.read_csv(gapminder_csv_url, low_memory=False, on_bad_lines='skip')
    details = {}
    details['mac_address'] = get_mac_address(record)
    details[utils.oui_field] = list(OuiLookup().query(details['mac_address'])[0].values())
    if details[utils.oui_field] == [None]:
        details[utils.oui_field] = []
    details['origin_dataset'] = os.path.splitext(os.path.basename(gapminder_csv_url))[0].split('_')[-1]
    first_record_time = parse_time(record['frame.time'][0])

    for field in fields:
        if field == 'http.request.full_uri':
            try:
                unique_values = list(record[field].unique())
            except:
                print(gapminder_csv_url)
                unique_values = []
        else:
            unique_values = list(record[field].unique())
        unique_values = [x for x in unique_values if isinstance(x, str)]

        # Add time left from the first record to the finding of each value
        details[field] = []
        details[field+'_with_time'] = []

        for value in unique_values:
            time_left = parse_time(record.loc[record[field] == value, 'frame.time'].iloc[0]) - first_record_time
            details[field+'_with_time'].append({'value': value, 'discovery_time': time_left})
            details[field].append(value)
    return details
def list_devices_files_paths(dir_path):
    list = []
    for csvFilename in os.listdir(dir_path):
        if not csvFilename.endswith(".csv"):
            continue
        list.append(dir_path+csvFilename)
    return list


def read_devices_details():
    devices_dict = {}
    for device_file_path in tqdm(list_devices_files_paths(utils.csvs_path)):
        devices_dict[os.path.basename(device_file_path)] = csv_to_dict_with_time(device_file_path)
    return devices_dict

def add_filtered_domains_field(input_dict, target_field, filter_function):
    for item in input_dict:
        input_dict[item][target_field] = [value for value in input_dict[item][utils.dns_field] if filter_function(value)]
    return input_dict
def fill_with_vendors(dict_devices):
    list_vendors_devices = {'echo': ['amazon'],
                            'fire': ['amazon'],
                            'sengled': ['sengled'],
                            'philips': ['philips'],
                            'blink': ['blink','b link'],
                            'tp_link': ['tplink', 'tp link', 'tp-link'],
                            'tplink': ['tplink','tp link','tp-link'],
                            'ring': ['ring'],
                            'insteon': ['insteon','Smartlabs'],
                            'xiaomi-cleaner': ['roborock','xiaomi'],
                            'samsung': ['samsung'],
                            'apple': ['apple'],
                            'magichome': ['magic home','magichome'],
                            'smart_things': ['smartthings', 'smart things','Samsung'],
                            'smartthings': ['smartthings','smart things','Samsung'],
                            'sousvide': ['anova'],
                            'google-home': ['google'],
                            'wansview': ['wansview'],
                            'xiaomi-hub': ['xiaomi'],
                            'nest': ['nest','dropcam','google'],
                            'Nest': ['nest','dropcam','google'],
                            'lightify': ['lightify','osram'],
                            'roku':['roku'],
                            'yi-camera':['yi','xiaoyi'],
                            'xiaoyi': ['yi', 'xiaoyi'],
                            'iKettle':['Smarter','ikettle'],
                            'Edimax': ['edimax'],
                            'MAXGateway':['homematic','homematic-ip','homematic ip','home matic','EQ-3','EQ3','EQ 3'],
                            'D-Link':['dlink','d-link','d link'],
                            'WeMo':['WeMo','We Mo','Belkin'],
                            'Hue':['Philips'],
                            'Withings':['Withings'],
                            'TP-Link': ['tplink', 'tp link', 'tp-link'],
                            'HomeMatic':['homematic','homematic-ip','homematic ip','home matic','EQ-3','EQ3','EQ 3'],
                            'Ednet':['Ednet'],
                            'Aria':['Fitbit','Aria'],
                            'Smarter':['Smarter'],
                            'Lightify':['lightify','osram'],
                            'Piper':['Piper','icontrol'],
                            'google':['Google'],
                            'Lifx':['Lifx'],
                            'Renpho':['Renpho'],
                            'Belkin':['WeMo','We Mo','Belkin'],
                            'Bose':['Bose'],
                            'Nvidia':['Nvidia'],
                            'Netgear':['Netgear',"Arlo"],
                            'Logitech':['Logitech'],
                            'August':['August'],
                            'Canary':['Canary'],
                            'Caseta':['Caseta','Lutron'],
                            'Chamberlain':['Chamberlain','MYQ'],
                            'HarmonKardonInvoke':['HarmonKardon'],
                            'Koogeek':['Koogeek'],
                            'MiCasaVerdeVera':['MiCasaVerdeVera'],
                            'Sonos':['Sonos'],
                            'Roomba':['Roomba','iRobot'],
                            'Securifi':['Securifi-Almond','Securifi Almond','Securifi'],
                            'Wink':['Wink'],
                            'HP':['HP'],
                            'Blipcare':['Blipcare'],
                            'PIX-Star':['PIX-STAR','PIX STAR'],
                            'ihome':['ihome'],
                            'Wyze':['Wyze'],
                            'Netatmo':['Netatmo'],
                            'Xiaomi': ['xiaomi'],
                            'Switcher': ['Switcher'],
                            'Dropcam': ['nest','dropcam','google'],
                            'Triby': ['Triby', 'Invoxia Triby', 'Invoxia']
                            }
    print("fill with vendors loop")
    for device in tqdm(dict_devices.keys()):
        guessed_vendor = ''
        for vendor in list_vendors_devices.keys():
            if vendor.lower() in device.lower():
                guessed_vendor = list_vendors_devices[vendor]
                break
        if guessed_vendor != '':
            dict_devices[device]['vendor'] = guessed_vendor
        elif 'vendor' not in dict_devices[device].keys():
            print(device)
            dict_devices[device]['vendor'] = 'unknown vendor'

    return dict_devices
def fill_with_categories(dict_devices):
    print('fill with categories loop')
    for device in tqdm(dict_devices.keys()):
        guessed_category = ''
        for cat in utils.categories_dict.keys():
            for sub_cat in utils.categories_dict[cat]:
                for real_type in dict_devices[device]['type']:
                    if sub_cat.lower() == real_type.lower():
                        guessed_category = cat
                        break
                if guessed_category != '':
                    break
            if guessed_category != '':
                dict_devices[device][utils.real_category_field] = guessed_category
                break
        if guessed_category == '':
                print(device)
                dict_devices[device][utils.real_category_field] = 'unknown category'
    return dict_devices
def fill_with_types(dict_devices):
    list_types_devices = {'echo': ['echo','speaker','hub','gateway'],
                            'fire': ['Television','Streamer','Streaming'],
                            'hub': ['gateway','hub'],
                            'Hub': ['gateway','hub'],
                            'Smart_Things':['gateway','hub'],
                            'light': ['bulb','light'],
                            'Light': ['bulb', 'light'],
                            'camera': ['camera'],
                            'plug': ['plug','switch'],
                            'Plug': ['plug', 'switch'],
                            'Switch': ['switch','plug'],
                            'ring': ['doorbell','camera'],
                            'cam': ['camera'],
                            'Cam': ['camera'],
                          'AugustDoorbell': ['doorbell', 'camera'],
                          'roku': ['Television','Set-top box','Streaming','Streamer'],
                            'tv': ['Television','Set-top box','Streaming','Streamer'],
                            'SecurifiAlmond':['router','gateway'],
                          'Samsung_smart_camera':['camera'],
                            'SamsungSmartThingsHub': ['hub'],
                            'bulb':['bulb','light'],
                            'cleaner':['vacuum'],
                            'google-home-mini':['speaker','hub'],
                            'GoogleHomeMini': ['speaker', 'hub'],
                          'GoogleHome': ['speaker', 'hub'],
                          'Google_SmartSpeaker': ['speaker', 'hub'],
                            'PiperNV':['camera'],
                            'magichome-strip':['bulb','light'],
                            'appletv':['Television','Streamer','Streaming','Set-top box'],
                            'nest-tstat':['thermostat'],
                            'sousvide':['Kitchen Aid'],
                            'blink-security-hub':['camera','hub'],
                          'canary': ['camera','security'],
                          'PIX-STAR_Photo-frame':['Photo Display'],
                          't-wemo-plug':['switch','plug'],
                            'iKettle': ['kitchen aid'],
                            'MAXGateway': ['gateway'],
                            'Siren':['siren'],
                            'D-LinkSensor':['motion','sensor'],
                            'WeMoInsightSwitch':['plug','switch'],
                            'Withings_merged':['weight scale','Scale'],
                            'WeMoLink_merged':['gateway','hub'],
                            'Bridge':['gateway','hub'],
                            'Gateway':['gateway','hub'],
                            'Water':['water sensor','sensor'],
                            'DoorSensor':['door sensor','sensor'],
                            'Aria_merged':['weight scale','scale','Fitness Tracker'],
                            'Coffee':['Kitchen Aid'],
                          'Sound':['Speaker'],
                          'NestProtect':['Smoke detector','Sensor','Siren'],
                          'Nest_Protect': ['Smoke detector', 'Sensor', 'Siren'],
                          'iHome_Wireless':['switch','plug'],
                          'Google_Home_mini':['speaker', 'hub'],
                          'NestGuard':['Lock','security'],
                          'MiCasaVerdeVeraLite':['gateway','hub'],
                          'Triby_Speaker':['Speaker'],
                          'Printer':['Printer'],
                          'Humidifier':['Humidifier'],
                          'Roomba': ['vacuum'],
                          'Motion_Sensor':['Sensor'],
                          'Smart_scale':['Weight scale','scale'],
                          'AppleHomePod':['gateway','hub','Speaker'],
                          'MotionSensor':['Sensor','Motion'],
                          'sleep_sensor':['Sensor'],
                          'Blood_Pressure_meter':['Sensor'],
                          'Smart_Baby_Monitor':['camera','Baby Monitor'],
                          'BelkinWeMoCrockpot':['Kitchen Aid'],
                          'BelkinWeMoLink':['bulb','light'],
                          'Sonos':['speaker'],
                          'Netatmo_Welcome':['Camera'],
                          'nVidiaShield':['Television','Streamer','Streaming'],
                          'HarmonKardonInvoke':['Speaker','Hub'],
                          'WithingsHome':['Camera'],
                          'ChamberlainmyQGarageOpener':['Garage Door','Lock'],
                          'Netatmo_weather_station':['Weather Station']
                          }
    print('fill with types loop')
    for device in tqdm(dict_devices.keys()):
        guessed_type = ''
        for type in list_types_devices.keys():
            if type.lower() in device.lower():
                guessed_type = list_types_devices[type]
        if guessed_type != '':
            dict_devices[device]['type'] = guessed_type
        elif 'type' not in dict_devices[device].keys():
            print(device)
            dict_devices[device]['type'] = 'unknown type'

    return dict_devices

def run_fast_scandir(dir, ext):    # dir: str, ext: list

    subfolders, files = [], []

    for f in os.scandir(dir):
        if f.is_dir():
            subfolders.append(f.path)
        if f.is_file():
            if os.path.splitext(f.name)[1].lower() in ext:
                files.append(f.path)

    for dir in list(subfolders):
        sf, f = run_fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files

def merge_pcaps(folder,mergedfilename):
    os.chdir(folder)
    if os.name == 'nt':
        mergecap_path = "C:\Program Files\Wireshark\mergecap.exe"
    else:
        mergecap_path = '/Applications/Wireshark.app/Contents/MacOS/mergecap'
    arg0 = '-w'
    arg1 =  mergedfilename
    arg2 = '*.pcap'
    subprocess.run([mergecap_path,arg0 ,arg1,arg2], shell=True)

def merge_all_pcaps_in_folders(dataset_folder):
    subfolders, files = run_fast_scandir(dataset_folder,'pcap')
    for folder in subfolders:
        merge_pcaps(folder,folder.split('/')[-1]+'_merged.pcapng')

def pre_proccessing_yourthings(pcaps_dir, csv_mac_ip_name_path):
    # you should run the extract_tgz before. to extract all tgz files.
    data = []
    with open(csv_mac_ip_name_path, "r",encoding='utf-8-sig') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            data.append(row)

    # Search for files in the folder path that match the strings in the third column of the CSV file
    for row in data:
        search_string = row["mac"].replace(':','')
        if search_string == '':
            continue
        for filename in os.listdir(pcaps_dir):
            if search_string in filename.lower():
                # Construct the new filename by replacing the original filename with the value in the first column of the CSV file
                new_filename = row["device_name"] + ".pcap"
                # Rename the file with the new filename
                shutil.move(os.path.join(pcaps_dir, filename), os.path.join(pcaps_dir, new_filename))
                print(f"Renamed {filename} to {new_filename}")

def pre_proccessing_sivanthan(data_dir):
    # you should run the extract_tgz before. to extract all tgz files.
    # Define the directory where your pcap files are located
    pcap_dir = data_dir + '/pcaps'

    # Define the output directory for the split pcap files
    split_pcap_dir = pcap_dir

    # Define the path to the CSV file containing the list of IPs and device names
    csv_file = data_dir + 'List_Of_Devices.txt'
    merged_pcap_file = pcap_dir + '/merged.pcap'

    # Read the CSV file and create a dictionary mapping each IP to its device name
    mac_dict = {}
    with open(csv_file) as f:
        for line in f:
            if "MAC ADDRESS" not in line:
                device_info = line.strip().split()
                device_name = "_".join(device_info[:-2])
                device_mac = device_info[-2]
                connection = device_info[-1]
                mac_dict[device_name+'_'+connection] = device_mac

    # Split the merged pcap file into separate files for each IP address
    for device_name, mac in mac_dict.items():
        split_pcap_file = os.path.join(split_pcap_dir, f"{device_name}.pcap")
        tshark_cmd = ["/Applications/Wireshark.app/Contents/MacOS/tshark", "-r", merged_pcap_file, "-w",
                      split_pcap_file,
                      f"eth.addr=={mac}"]
        subprocess.call(tshark_cmd)

def pre_proccessing_ourlab(data_dir):
    csv_mac_path = data_dir + "/oracle.csv"
    files = []
    # Loop through all subdirectories in the current directory
    for dir in os.listdir(data_dir):
        path_dir = os.path.join(data_dir,dir)
        if os.path.isdir(path_dir):
            # Create a list of all files (excluding .txt files) in the current subdirectory
            files.extend([os.path.join(path_dir,f) for f in os.listdir(path_dir) if os.path.isfile(os.path.join(path_dir, f)) and not f.endswith('.txt') and not f.startswith('.')])
        elif (dir.endswith('.pcap')):
            files.append(os.path.join(path_dir))
    # Check if there are any files (excluding .txt files) in the current subdirectory
    if len(files) > 0:
        # Concatenate all files (excluding .txt files) in the current subdirectory into a single file
        mergefile = f"{data_dir.rstrip('_')}_merged.pcap"
        cmd = f"mergecap -w {mergefile} " + " ".join(files)
        subprocess.run(cmd, shell=True)

    # Read the CSV file and create a dictionary mapping each IP to its device name
    with open(csv_mac_path, "r") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            device_name = row['device_name']
            mac = row['mac']
            split_pcap_file = os.path.join(data_dir, f"{device_name}.pcap")
            tshark_cmd = ["/Applications/Wireshark.app/Contents/MacOS/tshark", "-r", mergefile, "-w",
                          split_pcap_file,
                          f"eth.addr=={mac}"]
            subprocess.call(tshark_cmd)

def pre_proccessing_checkpoint(data_file):
    fields = dict_fields
    records = pd.read_csv(data_file, low_memory=False, on_bad_lines='skip')
    device_ids = list(records['device_id'].unique())
    grouped = records.groupby(['device_id'])
    devices_dict = {}
    for device_id in device_ids:
        details = {}
        record = grouped.get_group(device_id)
        details['origin_dataset'] = os.path.dirname(data_file).split('/')[-1]
        details['type'] = [record['name'].unique()[0]]
        details['vendor'] = [record['brand'].unique()[0]]
        domains_list = list(record['dst_hostname'].unique())
        import validators
        domains_list = [x for x in domains_list if x != '(empty)' and validators.domain(x)]
        details['dns.qry.name'] = domains_list
        for field in fields:
            if field == 'dns.qry.name':
                continue
            else:
                details[field] = list()
        details['vendor'] = get_vendor_from_domain_checkpoint(details)
        devices_dict[details['vendor'][0] + '_'+details['type'][0]+'_'+details['origin_dataset']]=details
    filter_dict = dict(filter(lambda elem:elem[1]['vendor'] != ['-'], devices_dict.items()))
    return filter_dict
def get_vendor_from_domain_checkpoint(device):
    if device['vendor'][0] == '-':
        domain = device['dns.qry.name'][0]
        if 'nuki' in domain:
            device['vendor'] = ['Nuki']
        elif 'ring' in domain:
            device['vendor'] = ['Ring']
        elif 'hismarttv' in domain:
            device['vendor'] = ['Hisense']
        elif 'lgtvs' in domain:
            device['vendor'] = ['LG']
        elif 'hp' in domain:
            device['vendor'] = ['HP']
        elif len(device['dns.qry.name']) > 1:
            if 'wyze' in device['dns.qry.name'][1]:
                device['vendor'] = ['Wyze']
    return device['vendor']



