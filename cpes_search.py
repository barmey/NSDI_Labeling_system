import json
import xml.etree.ElementTree as ET
import openai
import requests
import secrets
import os


class CPEFilter:
    def __init__(self, cache_file='cpe_cache.json'):
        file_path = 'official-cpe-dictionary_v2.3.xml'
        self.openai_api_key = secrets.api_key_openai
        self.nvd_api_key = secrets.api_key_nvd
        self.root = self.load_xml(file_path)
        self.namespace = {'cpe': 'http://cpe.mitre.org/dictionary/2.0'}
        self.cache_file = cache_file
        self.cache = self.load_cache()

    def load_xml(self, file_path):
        tree = ET.parse(file_path)
        return tree.getroot()

    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        else:
            return {}

    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=4)

    def filter_cpe_by_vendor(self, vendor_name):
        filtered_items = []
        for cpe_item in self.root.findall('cpe:cpe-item', self.namespace):
            name = cpe_item.get('name')
            title_element = cpe_item.find('cpe:title', self.namespace)
            title = title_element.text if title_element is not None else ""

            if vendor_name.lower() in name.lower() or vendor_name.lower() in title.lower():
                filtered_items.append(cpe_item)
        return filtered_items

    def get_cache_key(self, cpe_item, device_type):
        parts = cpe_item.split(':')
        key = f"{parts[1]}:{parts[2]}:{parts[3]}_{device_type}"
        return key

    def check_cpe_types_with_gpt4(self, cpe_items, device_type):
        openai.api_key = self.openai_api_key
        prompt = (
            f"Determine if the following CPE items are related to or an essential component of a device type: {device_type}. "
            f"Each item includes its name and title. Provide an explicit 'yes' or 'no' for each item, followed by a brief explanation. "
            f"Respond in JSON format with the following keys: 'item', 'name', 'title', 'answer', and 'explanation'.\n\n"
            f"Don't add any characters before or after the JSON output. Ensure that the response is always in a valid JSON list format.\n\n"
        )
        for cpe_item in cpe_items:
            name = cpe_item.get('name')
            parts = name.split(':')[0:4]
            prompt += f"Name: {parts[0]}:{parts[1]}:{parts[2]}:{parts[3]}\n"

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            n=1,
            stop=None,
            temperature=0
        )
        results = json.loads(response.choices[0].message['content'].strip())
        matching_items = []

        for result in results:
            item = result['item']
            key = self.get_cache_key(result['item'], device_type)
            if "yes" in result['answer'].lower():
                matching_items.append(item)
            # Update cache with the result
            self.cache[key] = {
                'answer': result['answer'],
                'explanation': result['explanation']
            }
        self.save_cache()
        return matching_items

    def filter_and_check_cpe_items(self, vendor_name, device_type):
        filtered_items1 = self.filter_cpe_by_vendor(vendor_name)
        unique_items = {}

        # Filter out unique items based on the cache key
        for item in filtered_items1:
            cache_key = self.get_cache_key(item.get('name'), device_type)
            if cache_key not in unique_items:
                unique_items[cache_key] = item

        matching_keys = []
        batch_size = 13

        unique_items_list = list(unique_items.values())

        for i in range(0, len(unique_items_list), batch_size):
            batch = unique_items_list[i:i + batch_size]
            cached_results = []
            new_items = []

            for item in batch:
                cache_key = self.get_cache_key(item.get('name'), device_type)
                if cache_key in self.cache:
                    if self.cache[cache_key]['answer'].lower() == 'yes':
                        matching_keys.append(cache_key)
                else:
                    new_items.append(item)

            if new_items:
                matching_keys.extend(self.check_cpe_types_with_gpt4(new_items, device_type))

        # Create a set of matching keys for quick lookup
        matching_keys_set = set(matching_keys)

        # Return the original filtered names of the CPEs
        matching_items = [item.get('name') for item in filtered_items1 if
                          self.get_cache_key(item.get('name'), device_type) in matching_keys_set]

        return matching_items

    def get_cves_for_cpe(self, cpe_uri):
        base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
        params = {
            'cpeName': cpe_uri,
            'resultsPerPage': 2000
        }
        if self.nvd_api_key:
            params['apiKey'] = self.nvd_api_key

        response = requests.get(base_url, params=params)
        response.raise_for_status()

        cves = response.json().get('vulnerabilities', [])
        return cves


# Example usage
if __name__ == "__main__":
    cpe_filter = CPEFilter()

    vendor_name = 'blink'
    device_type = 'camera'
    pairs = [('blink','camera'),('blink','hub'),('belkin','camera'),('belkin','sensor'),('belkin','light'),('belkin','plug')]
    for (vendor_name,device_type) in pairs:
        matching_cpe_items = cpe_filter.filter_and_check_cpe_items(vendor_name, device_type)
        print(f"These are the relevance CPEs for the vendor: {vendor_name} and the function: {device_type}:")
        # Print the matching items
        for item in matching_cpe_items:
            print(f"    CPE: {item}")