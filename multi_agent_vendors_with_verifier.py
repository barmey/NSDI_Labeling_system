import json
from collections import defaultdict
from tqdm import tqdm

import openai
import re
import random
import difflib
import os
import secrets
import utils

# Initialize OpenAI API key
openai.api_key = secrets.api_key_openai

# Known IoT manufacturers (initial list)
known_iot_manufacturers = [
    'Apple', 'Google', 'Samsung', 'Amazon', 'Microsoft', 'Huawei', 'Xiaomi',
    'Sony', 'LG', 'Intel', 'Cisco', 'HP', 'IBM', 'Dell', 'Nokia', 'Bosch',
    'Siemens', 'Philips', 'Panasonic', 'Lenovo'
]

# Define device categories
device_categories = [
     "Voice Control", "Power System", "Solar Panel", "Smart Meter", "HVAC", "Smart Appliance","Smart Hub","Smart Gateway",
    "Smart Washer", "Smart Fridge", "Smart Cleaner", "Sleep Tech", "Garage Door",
    "Sprinkler", "Touch Panel", "Scale", "Robot", "Weather Station", "Health Monitor", "Baby Monitor",
    "Pet Monitor", "Alarm", "Motion Detector", "Smoke Detector", "Water Sensor",
    "Sensor", "Media Player", "Television", "Game Console","Storage","Audio devices","Smart A/C",
    "Streamer", "Speaker", "Photo Camera", "Photo Display", "Projector","Health Sensor","Blood Pressure","Z-Wave hubs"
]
device_categories.extend(utils.read_csv_single_column_to_list(utils.types_path))
device_categories = list(set(device_categories))


def clean_json_string(json_string):
    # Replace single quotes with double quotes for keys and values
    # Ensure quotes within values are not affected
    def replace_quotes(match):
        return f'"{match.group(1)}"'

    # Match key-value pairs with single quotes and replace them with double quotes
    json_string = re.sub(r"\'([^\']+)\'", replace_quotes, json_string)

    # Fix any incorrect escaped quotes
    json_string = json_string.replace('\"', '"')
    json_string = json_string.replace('“', '"').replace('”', '"')
    json_string = json_string.replace('‘', "'").replace('’', "'")

    return json_string


# Initialize agents
agent1 = "gpt-4o"
agent2 = "gpt-4o"



class Coordinator:
    def __init__(self, agent1, agent2, existing_vendors=None):
        self.agent1 = agent1
        self.agent2 = agent2
        self.manufacturers_set = set()  # Unique manufacturers
        self.original_manufacturers = []  # Original names for prompts
        self.suggested_manufacturers = set()  # Track suggested manufacturers to avoid repetition
        self.categories_iter = iter(device_categories)  # Iterator for categories
        self.vendor_info = {}  # Dictionary to store vendor information
        self.wrong_names = []  # List to store wrong names

        # Load existing vendors if provided
        if existing_vendors:
            for vendor in existing_vendors:
                standardized_name = self.standardize_name(vendor)
                self.manufacturers_set.add(standardized_name)
                self.original_manufacturers.append(vendor)
                self.vendor_info[standardized_name] = existing_vendors[vendor]

    def standardize_name(self, name):
        return re.sub(r'[^a-zA-Z0-9\s-]', '', name).lower()

    def generate_manufacturers(self, agent, prompt):
        response = openai.ChatCompletion.create(
            model=agent,
            messages=[{"role": "system",
                       "content": "You are an AI that lists IoT manufacturers. Examples of IoT manufacturers include companies like Apple, Google, and Samsung. Examples of IoT devices include smart thermostats, smart speakers, smart lights, and smart security cameras."},
                      {"role": "user", "content": prompt}]
        )
        content = response.choices[0].message['content']
        # Extract the list from the content
        return re.findall(r'\d+\.\s*(.+)', content)  # Extract names without additional descriptions

    def clean_json_string(self, json_string):

        # Replace single quotes with double quotes for keys and values
        # Ensure quotes within values are not affected
        def replace_quotes(match):
            return f'"{match.group(1)}"'

        # Match key-value pairs with single quotes and replace them with double quotes
        json_string = re.sub(r"\'([^\']+?)\'", replace_quotes, json_string)

        # Fix any incorrect escaped quotes
        json_string = json_string.replace('\"', '"')
        json_string = json_string.replace('“', '"').replace('”', '"')
        json_string = json_string.replace('‘', "'").replace('’', "'")

        return json_string

    def verify_manufacturers(self, manufacturers):
        verified_manufacturers = []
        to_verify = []

        for manufacturer in manufacturers:
            standardized_name = self.standardize_name(manufacturer)
            if standardized_name not in self.wrong_names and standardized_name not in self.vendor_info:
                to_verify.append(manufacturer)

        # Split the list into chunks of 5
        chunks = [to_verify[i:i + 5] for i in range(0, len(to_verify), 5)]

        for chunk in chunks:
            prompt = f"Verify if the following are real IoT vendors. For each verified vendor, provide the information in the following JSON format:\n" \
                     f"{{'name': 'Vendor Name', 'description': 'Description of the vendor', 'url': 'URL to the vendor's website','types':['types that the vednor produces from the provided types list']}}. \n" \
                     f"Vendors: {', '.join(chunk)}."
            types_list = """Camera, Doorbell, Hub, Plug, Bulb, Light, Thermostat, Television, Speaker, Switch, Lock, Gateway, Vacuum, Echo, Siren, Motion, Weight Scale, Sensor, Photo Display, Smoke Detector, Water Sensor, Door Sensor, Printer, Humidifier, Kitchen Aid, Streamer, Garage Door, Weather Station, Router"""

            prompt += f"\n\nUse the following list of types: {types_list}\nDo not combine types into one. Use separate types as listed. don't add before or after the json output any characters. \n Ensure that the response is always in a valid JSON list format.\n"
            response = openai.ChatCompletion.create(
                model=self.agent2,
                messages=[
                    {"role": "system", "content": "You are an AI that verifies the authenticity of IoT manufacturers."},
                    {"role": "user", "content": prompt}],
                temperature=0  # Lower temperature for more deterministic responses
                )
            content = response.choices[0].message['content']
            content = self.clean_json_string(content)
            try:
                verified_list = json.loads(content)
                for item in verified_list:
                    name = item['name']
                    url = item['url']
                    types = item['types']
                    description = item['description']
                    standardized_name = self.standardize_name(name)
                    verified_manufacturers.append((standardized_name, description, url))
                    self.vendor_info[standardized_name] = {"name":name, "url": url,"types":types}
            except json.JSONDecodeError:
                print(f"Error JSON!!! need to reavlute the response: {content}")
                continue

            for manufacturer in chunk:
                standardized_name = self.standardize_name(manufacturer)
                if standardized_name not in [name for name, _, _ in verified_manufacturers]:
                    self.wrong_names.append(standardized_name)

        return verified_manufacturers

    def coordinate(self):
        round_num = 1
        consecutive_low_rounds = 0
        use_categories = False

        while consecutive_low_rounds < 5 or use_categories:
            print(f"Round {round_num}:")

            # Agent 1 generates manufacturers
            agent = self.agent1
            agent_name = "Agent 1"

            random.shuffle(self.original_manufacturers)  # Randomize the order of the original manufacturers

            if use_categories:
                try:
                    category = next(self.categories_iter)
                except StopIteration:
                    use_categories=False
                    break  # If no more categories, exit the loop
                prompt = f"Please list 50 unique IoT manufacturers that produce devices of the type: {category} and are not in this list: {', '.join(self.original_manufacturers)}. Do not repeat any previously mentioned manufacturers. Do not add any information or characters after the vendor name."
            else:
                prompt = f"Please list 50 unique IoT manufacturers that are not in this list: {', '.join(self.original_manufacturers)}. Do not repeat any previously mentioned manufacturers. Do not add any information or characters after the vendor name."

            new_manufacturers = self.generate_manufacturers(agent, prompt)

            standardized_new_manufacturers = []
            for m in new_manufacturers:
                standardized_name = self.standardize_name(m.strip())
                if standardized_name not in self.manufacturers_set and standardized_name not in self.wrong_names:
                    standardized_new_manufacturers.append(m.strip())

            # Agent 2 verifies manufacturers
            verified_manufacturers = self.verify_manufacturers(standardized_new_manufacturers)
            new_verified_count = len(verified_manufacturers)

            for name, description, url in verified_manufacturers:
                self.manufacturers_set.add(name)
                self.original_manufacturers.append(name)
                self.vendor_info[name] = {"description": description, "url": url}

            if new_verified_count > 10:
                consecutive_low_rounds = 0
            else:
                consecutive_low_rounds += 1
                if consecutive_low_rounds >= 5:
                    use_categories = True

            print(f"{agent_name} verified: {new_verified_count}")
            print(f"{agent_name} provided: {new_manufacturers}")
            print(f"Total manufacturers collected: {len(self.manufacturers_set)}\n")

            round_num += 1

        return self.vendor_info, self.wrong_names

if False:
    # Load and standardize the final list of IoT manufacturers from the JSON file
    with open('gpt_4o_vendor_functions.json', 'r') as file:
        verified_manufacturers = json.load(file)

    # Initialize coordinator
    coordinator = Coordinator(agent1, agent2,verified_manufacturers)

    # Coordinate interactions
    vendor_info, wrong_names = coordinator.coordinate()
    print("Final List of Verified IoT Manufacturers with Information:")
    print(vendor_info)

    print("List of Wrong Names:")
    print(wrong_names)

    # Save the final list to a JSON file
    with open('verified_iot_manufacturers_list_no_categories.json', 'w') as file:
        json.dump(vendor_info, file, indent=4)

    # Save the wrong names to a file
    with open('wrong_iot_manufacturers_list.txt', 'w') as file:
        for name in wrong_names:
            file.write(f"{name}\n")

# Load and standardize the test list
def read_and_standardize_list_from_file(filename):
    with open(filename, 'r') as file:
        return [re.sub(r'[^a-zA-Z0-9\s-]', '', line.strip()).lower() for line in file.readlines()]


def combine_vendors(vendors):
    # Step 1: Combine vendors with names that include each other
    combined_by_name = defaultdict(list)

    for name, info in vendors.items():
        base_name = name.split()[0]
        combined_by_name[base_name].append(name)

    combined_vendors_step1 = {}
    for base_name, names in combined_by_name.items():
        if len(names) == 1:
            combined_vendors_step1[names[0]] = vendors[names[0]]
        else:
            #combined_description = " ".join([vendors[name]['description'] for name in names])
            combined_types = []
            for name in names:
                if 'types' in vendors[name]:
                    combined_types.extend(vendors[name]['types'])
            url = vendors[names[0]]['url']  # Assuming all have the same URL
            combined_vendors_step1[base_name] = {
                "description": '',
                "url": url,
                "types":list(set(combined_types))
            }

    # Step 2: Check for repetitive URLs and combine them
    combined_by_url = defaultdict(list)

    for name, info in combined_vendors_step1.items():
        combined_by_url[info['url']].append(name)

    combined_vendors_final = {}
    for url, names in combined_by_url.items():
        if len(names) == 1:
            combined_vendors_final[names[0]] = combined_vendors_step1[names[0]]
        else:
            shortest_name = min(names, key=len)
            #combined_description = " ".join([combined_vendors_step1[name]['description'] for name in names])
            combined_types = []
            for name in names:
                if name in vendors:
                    if 'types' in vendors[name]:
                        combined_types.extend(vendors[name]['types'])
            combined_vendors_final[shortest_name] = {
                "description": '',
                "url": url,
                'types':combined_types
            }

    return combined_vendors_final

# File to save the results
results_file = 'gpt_4o_vendor_functions.json'

# Load existing results if the file exists
if os.path.exists(results_file):
    with open(results_file, 'r') as file:
        saved_results = json.load(file)
else:
    saved_results = {}


# Function to send request to OpenAI API and get vendor types
def get_vendor_types(vendors, batch_size=8):
    # Prepare vendor list
    vendors_list = [{"name": name, "url": info["url"]} for name, info in vendors.items()]

    # Filter out vendors that already have saved results
    new_vendors_list = [vendor for vendor in vendors_list if vendor['name'] not in saved_results]

    # Initialize results with saved results
    results = [saved_results[name] for name in saved_results]

    # Create batches of new vendors
    batches = [new_vendors_list[i:i + batch_size] for i in range(0, len(new_vendors_list), batch_size)]

    for batch in tqdm(batches):
        prompt = f"Check the types that the following vendors produce from the list provided:\n"
        types_list = """Camera, Doorbell, Hub, Plug, Bulb, Light, Thermostat, Television, Speaker, Switch, Lock, Gateway, Vacuum, Echo, Siren, Motion, Weight Scale, Sensor, Photo Display, Smoke Detector, Water Sensor, Door Sensor, Printer, Humidifier, Kitchen Aid, Streamer, Garage Door, Weather Station, Router"""

        for vendor in batch:
            name, url = vendor['name'], vendor['url']
            prompt += f"\nVendor: {name}, URL: {url}"

        prompt += f"\n\nUse the following list of types: {types_list}\nDo not combine types into one. Use separate types as listed. Answer with a structured JSON of name, url, and array of types without any additional text."

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {"role": "system", "content": "You are an assistant who knows all about vendors of IoT and smart home devices"},
                {"role": "user", "content": prompt}
            ]
        )

        # Extract JSON array from the response content
        response_content = response['choices'][0]['message']['content']
        start_idx = response_content.find('[')
        end_idx = response_content.rfind(']') + 1
        response_json = json.loads(response_content[start_idx:end_idx])
        print(response_json)

        for vendor_response in response_json:
            name = vendor_response['name']
            results.append(vendor_response)
            saved_results[name] = vendor_response

        # Save results to file after each batch
        with open(results_file, 'w') as file:
            json.dump(saved_results, file, indent=4)

    return results

# Read and standardize the test list from another file
test_list = list(set(read_and_standardize_list_from_file('vendors_list_updated.csv')))


# Load and standardize the final list of IoT manufacturers from the JSON file
with open('verified_iot_manufacturers_list_no_categories.json', 'r') as file:
    verified_manufacturers = json.load(file)

# Load and standardize the final list of IoT manufacturers from the JSON file
with open('gpt_4o_vendor_functions.json', 'r') as file:
    gpt_vendors_functions = json.load(file)


#with open('gpt_4o_vendor_functions.json', 'w') as file:
#    json.dump(gpt_vendors_functions, file, indent=4)

#combined_vendors = combine_vendors(verified_manufacturers)
# Save the final list to a JSON file
#with open('verified_iot_manufacturers_list_no_categories.json', 'w') as file:
#    json.dump(combined_vendors, file, indent=4)



#get_vendor_types(verified_manufacturers)

final_manufacturers_list = [re.sub(r'[^a-zA-Z0-9\s-]', '', vendor.lower()) for vendor in verified_manufacturers.keys()]
print(final_manufacturers_list)
# Save the wrong names to a file
with open('verified_iot_manufacturers_no_categories_names_only.txt', 'w') as file:
    for name in final_manufacturers_list:
        file.write(f"{name}\n")
# Check if the final list is comprehensive
missing_manufacturers = [manufacturer for manufacturer in test_list if not any(manufacturer in vendor or difflib.SequenceMatcher(None, manufacturer, vendor).ratio() > 0.8 for vendor in final_manufacturers_list)]

# Output the results
if missing_manufacturers:
    print("The final list is missing the following IoT manufacturers:")
    print(missing_manufacturers)
    print(len(missing_manufacturers))
else:
    print("The final list is comprehensive and includes all manufacturers from the test list.")

missing_by_cp = utils.read_csv_single_column_to_list("missing_manufacturers.txt")
test_list = missing_by_cp

# Check if the final list is comprehensive
missing_manufacturers = [manufacturer for manufacturer in test_list if not any(
    manufacturer in vendor or difflib.SequenceMatcher(None, manufacturer, vendor).ratio() > 0.9 for vendor in
    final_manufacturers_list)]

# Output the results
if missing_manufacturers:
    print("The final list is missing the following IoT manufacturers:")
    print(missing_manufacturers)
    print(len(missing_manufacturers))
else:
    print("The final list is comprehensive and includes all manufacturers from the test list.")