import collections
from itertools import chain

import numpy
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import data_augmentaion
from collections import Counter
import seaborn as sns
import numpy as np

import data_classification_RB
import utils
from utils import valid_external_domain
import plotly.graph_objects as go


def sankey_digram():
    import pandas as pd
    import plotly.graph_objects as go

    # Your data
    y_true = ['weight scale', 'television', 'doorbell', 'camera', 'kitchen aid', 'bulb', 'sensor', 'Plug', 'camera',
              'camera', 'speaker', 'camera', 'hub', 'camera', 'sensor', 'hub', 'siren', 'plug', 'sensor',
              'sensor', 'camera', 'camera', 'plug', 'hub', 'echo', 'echo', 'echo', 'hub', 'television', 'speaker',
              'speaker', 'printer', 'hub', 'hub', 'Plug', 'kitchen aid', 'light', 'light', 'light', 'hub',
              'streamer', 'camera', 'camera', 'lock', 'smoke detector', 'thermostat', 'smoke detector', 'camera',
              'weather station', 'camera', 'hub', 'camera', 'camera', 'photo display', 'humidifier', 'doorbell',
              'vacuum', 'streamer', 'camera', 'television', 'router', 'hub', 'hub', 'speaker', 'kitchen aid',
              'plug', 'bulb', 'camera', 'plug', 'Plug', 'camera', 'plug', 'Plug', 'hub', 'hub', 'weight scale',
              'camera', 'sensor', 'camera', 'vacuum', 'hub', 'camera', 'bulb', 'camera', 'speaker', 'kitchen aid',
              'camera', 'camera', 'weight scale', 'bulb', 'camera', 'hub', 'sensor', 'camera', 'hub', 'garage door']
    y_pred = ['weight scale', 'television', 'lock', 'camera', 'sensor', 'hub', 'sensor', 'sensor', 'camera',
              'camera', 'speaker', 'camera', 'Plug', 'camera', 'sensor', 'hub', 'siren', 'plug', 'sensor',
              'sensor', 'camera', 'camera', 'camera', 'hub', 'echo', 'echo', 'echo', 'hub', 'television', 'speaker',
              'speaker', 'printer', 'light', 'hub', 'speaker', 'kitchen aid', 'light', 'light', 'light', 'hub',
              'streamer', 'camera', 'camera', 'thermostat', 'thermostat', 'lock', 'doorbell', 'camera',
              'weather station', 'camera', 'light', 'camera', 'camera', 'photo display', 'weight scale', 'doorbell',
              'vacuum', 'streamer', 'camera', 'television', 'router', 'bulb', 'hub', 'speaker', 'kitchen aid', 'plug',
              'bulb', 'camera', 'plug', 'Plug', 'camera', 'camera', 'camera', 'camera', 'hub', 'weight scale',
              'sensor', 'sensor', 'camera', 'vacuum', 'hub', 'camera', 'vacuum', 'camera', 'speaker', 'kitchen aid',
              'hub', 'camera', 'weight scale', 'bulb', 'hub', 'hub', 'sensor', 'camera', 'hub', 'garage door']

    # Step 1: Create a DataFrame
    df = pd.DataFrame({'Real Function': y_true, 'System Labeled Function': y_pred})

    # Step 2: Filter out correct predictions (only keep mismatches)
    df_mismatches = df[df['Real Function'] != df['System Labeled Function']]

    # Step 3: Compute the counts of each mismatched mapping
    counts = df_mismatches.groupby(['Real Function', 'System Labeled Function']).size().reset_index(name='Count')

    # Step 4: Prepare data for the Sankey diagram
    # Create internal unique labels to differentiate between real and predicted functions
    counts['Real_Label_Internal'] = 'Real_' + counts['Real Function']
    counts['Pred_Label_Internal'] = 'Pred_' + counts['System Labeled Function']

    # Create lists of internal labels
    real_labels_internal = counts['Real_Label_Internal'].unique().tolist()
    pred_labels_internal = counts['Pred_Label_Internal'].unique().tolist()

    # Combine all internal labels
    all_labels_internal = real_labels_internal + pred_labels_internal

    # Create display labels (without prefixes)
    real_labels_display = counts['Real Function'].unique().tolist()
    pred_labels_display = counts['System Labeled Function'].unique().tolist()
    all_labels_display = real_labels_display + pred_labels_display

    # Create a mapping from internal labels to indices
    label_indices = {label: idx for idx, label in enumerate(all_labels_internal)}

    # Map the labels to indices for source and target
    counts['Source'] = counts['Real_Label_Internal'].map(label_indices)
    counts['Target'] = counts['Pred_Label_Internal'].map(label_indices)

    sources = counts['Source'].tolist()
    targets = counts['Target'].tolist()
    values = counts['Count'].tolist()

    # Prepare node positions
    # x positions: 0.1 for real function labels, 0.9 for system-labeled function labels
    node_x = []
    for label in all_labels_internal:
        if label.startswith('Real_'):
            node_x.append(0.1)
        else:
            node_x.append(0.9)

    # The labels for display
    node_labels = real_labels_display + pred_labels_display

    # Step 5: Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        arrangement='fixed',  # Fix the node arrangement to respect x positions
        node=dict(
            pad=10,
            thickness=15,
            line=dict(color="black", width=0.8),
            label=node_labels,
            x=node_x,
            # y positions are optional; Plotly will arrange them vertically
            # Optionally, you can adjust the font size here
            # color="blue",  # You can set node colors if desired
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            # Optionally, set link colors
            # color="rgba(0, 100, 200, 0.5)",
        ))])

    # Add annotations for the columns
    fig.update_layout(
        #title_text="Sankey Diagram of Mismatched Labels",
        font_size=10,
        annotations=[
            dict(
                x=0.1,
                y=1.05,
                xref='paper',
                yref='paper',
                text='Function',
                showarrow=False,
                xanchor='center',
                yanchor='bottom',
                font=dict(size=12)
            ),
            dict(
                x=0.9,
                y=1.05,
                xref='paper',
                yref='paper',
                text='System\'s Label',
                showarrow=False,
                xanchor='center',
                yanchor='bottom',
                font=dict(size=12)
            )
        ],
        width=400,  # Adjust the width as needed
        height=400,  # Adjust the height as needed
        margin=dict(l=10, r=10, t=80, b=10),  # Increased top margin to accommodate annotations
    )

    fig.show()


def create_device_histogram(devices_dict):
    # Initialize a dictionary to store the counts
    len_dict = collections.defaultdict(int)

    # Iterate over the lists in the input_dict
    for key, value in devices_dict.items():
        # Count the length of each list
        len_dict[len(value)] += 1

    all_lengths = list(range(1, 11 + 1))

    # Get the counts for these lengths, defaulting to 0 if a length does not appear in len_dict
    all_counts = [len_dict.get(length, 0) for length in all_lengths]
    # Plot histogram
    lengths = list(len_dict.keys())
    counts = list(len_dict.values())
    plt.bar(all_lengths, all_counts, tick_label=[str(length) for length in all_lengths])
    plt.xlabel('Frequency')
    plt.ylabel('Count')
    plt.title('Devices Frequency Histogram')
    for i in range(len(all_lengths)):
        if all_counts[i] != 0:
            plt.text(all_lengths[i], all_counts[i] + 0.09, str(all_counts[i]), ha='center')

    plt.show()

def get_top_external_services_by_vendor(df, vendor_name, top_n=5):
    services_counter = Counter()
    for _, row in df[df['vendor'].apply(lambda x: x[0].lower() == vendor_name.lower())].iterrows():
        for domain in row['valid_external_domains']:
            services_counter[domain] += 1
    return services_counter.most_common(top_n)

def check_how_many_vendors_in_dict(dict_devices):
    vendors_array = []
    vendor_counter = 0
    for device in dict_devices:
        if len(numpy.intersect1d(dict_devices[device]['vendor'],vendors_array)) == 0:
            vendor_counter = vendor_counter + 1
        vendors_array.extend(dict_devices[device]['vendor'])
    print(vendor_counter)

def data_classification_string_matching_type(dict_enriched_devices,fields_to_search,data_fields=utils.fields_data):
    field_to_search_in = fields_to_search #[dns_field, hostname_field, hostnames_google_field, domains_google_field]
    vendors = utils.read_csv_single_column_to_list(utils.vendors_path)
    vendors = utils.acquired_by_gpt_vendors_list_gpt_filtered_one_by_one
    dict_results = {}
    index = -1
    print("data_classification_string_matching_vendor_equally")
    for device in tqdm(dict_enriched_devices.keys()):
        index = index + 1
        vec = data_classification_RB.calculate_vector_counter_word_in_text_equal_fields(dict_enriched_devices[device], field_to_search_in, vendors)
        dict_results[device] = vec

    results = dict_results
    sorted_results = dict(
        sorted(results.items(), key=lambda x: x[1][0][1], reverse=True))

    first_counters = [v[0][1] for _, v in sorted_results.items()]
    second_counters = [v[1][1] if len(v) > 1 else 0 for _, v in sorted_results.items()]

    plt.figure(figsize=(10, 6))

    # Plotting first and second counters for each device
    plt.scatter(range(len(sorted_results)), first_counters, color='blue', label='First Result', marker='o', s=100)
    plt.scatter(range(len(sorted_results)), second_counters, color='red', label='Second Result', marker='^', s=100)

    # Aesthetic tweaks
    plt.xlabel('Device')
    plt.ylabel('Confidence Score')
    #plt.title('Counters for First and Second Results per Device')
    plt.legend()
    plt.xticks([])  # Remove x-labels
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.tight_layout()

    plt.show()

    # Compute the ratio for each device and avoid division by zero
    ratios = {k: v[0][1] / v[1][1] if len(v) > 1 and v[1][1] != 0 else 0 for k, v in results.items()}

    # Sort the devices based on their ratios in descending order
    sorted_ratios = dict(sorted(ratios.items(), key=lambda x: x[1], reverse=True))

    plt.figure(figsize=(10, 6))

    # Plotting the ratios
    plt.scatter(range(len(sorted_ratios)), list(sorted_ratios.values()), color='purple', s=100)

    # Aesthetic tweaks
    plt.xlabel('Device')
    plt.ylabel('Ratio (First Result/Second Result)')
    #plt.title('Ratio of First to Second Result per Device')
    plt.xticks([])  # Remove x-labels
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.tight_layout()

    plt.show()

    # 1. Calculate the average and SD of the first results per device
    first_results = [device[0][1] for device in results.values() if type(device)==type([])]
    avg_first_result = np.mean(first_results)
    sd_first_result = np.std(first_results)

    print(f"Average of first results: {avg_first_result}")
    print(f"Standard deviation of first results: {sd_first_result}")

    # 2. Calculate the average and SD of the ratio between the first and second results counter
    ratios = []
    for device in results.values():
        if type(device)==type([]) and len(device) >= 2:  # Ensure there's at least two results to calculate a ratio
            ratios.append(device[0][1] / device[1][1])
        else:
            ratios.append(0)

    avg_ratio = np.mean(ratios)
    sd_ratio = np.std(ratios)

    print(f"Average of ratios between first and second results: {avg_ratio}")
    print(f"Standard deviation of ratios: {sd_ratio}")
sankey_digram()
ext_dict = data_augmentaion.build_enriched_dict()
ext_dict = utils.filter_search_results(ext_dict, max_results_dict=utils.max_results_dict)
experiment_filtered_fields = utils.update_experiment_fields(utils.experiment_fields, utils.max_results_dict)
ext_dict_unique = {key: ext_dict[key] for key in utils.devices_unique_array if key in ext_dict}

#data_classification_string_matching_type(ext_dict_unique,experiment_filtered_fields,data_fields=utils.fields_data)

#check_how_many_vendors_in_dict(ext_dict_unique)


data = pd.DataFrame.from_dict(ext_dict,orient='index')

create_device_histogram(utils.devices_dict_unique)

import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import json

with open(utils.vendor_type_list_chat_gpt4_path, 'r') as f:
    dict_vendor_type = json.load(f)
    dict_vendor_type = utils.lowercase_keys(dict_vendor_type)
# Compute types count
types_count = [len(v) for v in dict_vendor_type.values()]

# Compute histogram
values, base = np.histogram(types_count, bins=np.arange(1, max(types_count) + 2) - 0.5, density=True)

# Compute cumulative
cumulative = np.cumsum(values)
plt.figure(figsize=(6, 4))

# Plot
plt.plot(base[:-1], [100*x for x in cumulative], c='blue')

plt.ylim(0,100)
plt.xlabel('Number of Functions')
plt.ylabel('% of Vendors')
#plt.title('')
#plt.grid(True)
plt.tight_layout()
plt.show()




# Define your mapping of old type names to new type names
type_mapping = {
    'Switch': 'Switch/Plug',
    'Plug': 'Switch/Plug',
    'Gateway': 'Hub/Gateway',
    'Hub': 'Hub/Gateway',
    'Echo': 'Speaker',
    'Bulb': 'Light/Bulb',
    'Light': 'Light/Bulb',
    'Baby Monitor': 'Camera',
    'motion':'Motion',
    #'Garage Door':'Lock',
    # add any other mappings you need here...
}

# Update the device types in the dictionary
for vendor, device_types in dict_vendor_type.items():
    for i in range(len(device_types)):
        if device_types[i] in type_mapping:
            device_types[i] = type_mapping[device_types[i]]



# Compute type frequency
type_freq = defaultdict(int)
for types in dict_vendor_type.values():
    for type in types:
        type_freq[type] += 1
del type_freq['Blood Pressure Monitor']
del type_freq['Water Sensor']
del type_freq['Door Sensor']
#del type_freq['Router']

sorted_data = dict(sorted(type_freq.items(), key=lambda item: item[1], reverse=True))

# Extracting keys and values from the defaultdict
categories = list(sorted_data.keys())
values = list(sorted_data.values())

plt.figure(figsize=(6, 4))

# Creating the bar graph
plt.bar(categories, values, color='skyblue')

# Aesthetic tweaks
plt.xlabel('Functions')
plt.ylabel('Number of Vendors')
#plt.title('Counts for Each Category')
plt.xticks(rotation=90)
plt.tight_layout()

plt.show()
# Compute histogram
values, base = np.histogram(list(type_freq.values()), bins=np.arange(1, max(type_freq.values()) + 2))

# Compute cumulative
cumulative = np.cumsum(values)

# Calculate cumulative percentage
cumulative_percentage = cumulative / cumulative[-1]

# Create a new figure with a specified size (width, height)
plt.figure(figsize=(5, 2))
# Plot
plt.plot(base[:-1], [100*x for x in cumulative_percentage], c='blue')

plt.xlabel('Number of Vendors')
plt.ylabel('% of Functions')
#plt.title('CDF of Types by Number of Vendors')
#plt.grid(True)
plt.ylim(0,100)
plt.xticks(np.arange(1, max(type_freq.values()) + 1)[::4])

# Adjust layout
plt.tight_layout()

plt.show()

# Extracting labels and their corresponding values
labels = list(type_freq.keys())
values = list(type_freq.values())

# Sorting data based on values for better visualization
labels, values = zip(*sorted(zip(labels, values), key=lambda x: x[1], reverse=True))

# Plotting the bar chart
plt.figure(figsize=(15, 10))
plt.bar(labels, values, color='skyblue')
plt.xlabel('Device Type')
plt.ylabel('Count')
plt.title('Distribution of Device Types')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# Extract valid external domains
data['valid_external_domains'] = data['dns.qry.name'].apply(lambda x: [domain for domain in x if valid_external_domain(domain.lower())])

# Count the occurrences of each domain in the 'dns.qry.name' column
domain_counter = Counter()
for domains in data['valid_external_domains']:
    unique_domains = set(domains)
    for domain in unique_domains:
        domain_counter[domain] += 1

# Get the top 50 domains
top_50_domains = domain_counter.most_common(10)
top_50_domains_set = set(domain[0] for domain in top_50_domains)

# Get unique vendors
unique_vendors = sorted(set(row['vendor'][0].capitalize() for _, row in data.iterrows()))

# Count vendors for each domain
vendor_domain_counts = {domain: {vendor: 0 for vendor in unique_vendors} for domain in top_50_domains_set}

for _, row in data.iterrows():
    for domain in top_50_domains_set:
        if domain in row['valid_external_domains']:
            vendor = row['vendor'][0].capitalize()
            vendor_domain_counts[domain][vendor] = 1

# Count the number of vendors for each domain
vendors_count_per_domain = {domain: sum(vendor_domain_counts[domain].values()) for domain in top_50_domains_set}

# Sort the domains by the number of vendors using them
sorted_domains = sorted(vendors_count_per_domain.keys(), key=lambda x: vendors_count_per_domain[x], reverse=True)

# Create a bar chart to visualize the data
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(top_50_domains))

# Plot the stacked bars
bottom = np.zeros(len(top_50_domains))
vendors_in_graph = set()
for vendor in unique_vendors:
    heights = [vendor_domain_counts[domain][vendor] for domain in sorted_domains]
    if any(heights):
        ax.bar(x, heights, width=0.8, bottom=bottom, label=vendor)
        bottom += heights
        vendors_in_graph.add(vendor)

ax.set_xticks(x)
ax.set_xticklabels(sorted_domains, rotation=45, ha='center')

plt.title('Number of Different Vendors Using Top 10 Domains')
plt.xlabel('Domains')
plt.ylabel('Number of Vendors')
plt.legend(title='Vendors', ncol=2, bbox_to_anchor=(0.68, 1), loc='upper left', labels=vendors_in_graph)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Define device categories based on device types
device_categories = {
    'Camera/Doorbell': ['Camera','Doorbell'],
    'Bulb/Light': ['Bulb', 'Light'],
    'Thermostat': ['Thermostat'],
    'Television/Streamer': ['Television','Streamer'],
    'Speaker/Voice Assitant': ['Speaker', 'Echo'],
    'Hub/Gateway': ['Hub', 'Gateway'],
    'Switch/Plug': ['Switch','Plug'],
    'Vacuum': ['Vacuum'],
    'Siren/Alarm/Lock': ['Siren', 'Alarm','Lock'],
    'Weather Station':['Weather Station'],
    'Sensor': ['Smoke Detector', 'Water Sensor', 'Door Sensor','motion','sensor'],
    'Weight Scale': ['Weight Scale','Scale'],
    'Printer': ['Printer'],
    'Humidifier': ['Humidifier'],
    'Crockpot': ['Crockpot'],
    'Photo Display':['Photo Display']}

# Create a dictionary to store the count of devices in each category
category_count = {category: 0 for category in device_categories.keys()}

# Loop through the devices in the dataset and count the number of devices in each category
for index, row in data.iterrows():
    for category, types in device_categories.items():
        flag = False
        if any(device_type.lower() in [x.lower() for x in row['type']] for device_type in types):
            category_count[category] += 1
            flag = True
            break

# Sort the categories by their count values in descending order
sorted_categories = sorted(category_count.items(), key=lambda x: x[1], reverse=True)

# Create a horizontal bar plot of the device categories and their counts
sns.set()
palette = sns.color_palette("husl", len(sorted_categories))
plt.figure(figsize=(10, 5))
plt.barh([cat[0] for cat in sorted_categories], [cat[1] for cat in sorted_categories], color=palette)
plt.yticks(fontsize=14)  # Adjust the y-axis label font size
plt.xticks(fontsize=14)  # Adjust the y-axis label font size
plt.xlim(0,90)
#plt.title('Device Categories')
#plt.xlabel('Number of Devices')
#plt.ylabel('Category')
# Add total count annotation
total_count = sum(category_count.values())
plt.text(0.2, 0.95, f'Total Count: {total_count}', ha='right', va='top', transform=plt.gca().transAxes, fontsize=14)

plt.grid(axis='x', linestyle='--', alpha=0.7)

# Extract valid external domains
data['valid_external_domains'] = data['dns.qry.name'].apply(lambda x: [domain for domain in x if valid_external_domain(domain.lower())])

# Count common external services
external_services = Counter()
for domains in data['valid_external_domains']:
    for domain in domains:
        external_services[domain] += 1

# Plot common external services as a pie chart
top_services = external_services.most_common(10)
services, counts = zip(*top_services)
plt.figure(figsize=(10, 8))
colors = plt.cm.plasma(np.linspace(0, 1, len(top_services)))
plt.pie(counts, labels=services, colors=colors)
plt.title('Top 10 Common External Services')
plt.axis('equal')

# Count common external services
vendors_counter = Counter()
for vendor in data['vendor']:
    vendors_counter[vendor[0].capitalize()] += 1
# Get the top 10 most common vendors
top_10_vendors = vendors_counter.most_common(100)
top_vendors = [vendor for vendor, _ in top_10_vendors]
# Adjust the vendor's names
# Adjust the vendor's names
data['vendor'] = data['vendor'].apply(lambda x: x[0].capitalize())

# Generate a list of unique vendors
unique_vendors = data['vendor'].unique()

# Create a dictionary to store the count of devices from each vendor for each category
vendor_device_counts = {vendor: {category: 0 for category in device_categories.keys()} for vendor in unique_vendors}

# Loop through the devices in the dataset and count the number of devices from each vendor in each category
for index, row in data.iterrows():
    for category, types in device_categories.items():
        if any(device_type.lower() in [x.lower() for x in row['type']] for device_type in types):
            vendor_device_counts[row['vendor']][category] += 1
            break  # Count each device only once in a category

# Calculate the number of non-zero categories for each vendor
category_counts = {vendor: sum(1 for count in counts.values() if count > 0) for vendor, counts in vendor_device_counts.items()}

# Sort the vendors by the number of non-zero categories
sorted_vendors = sorted(unique_vendors, key=lambda vendor: category_counts[vendor], reverse=True)

# Create a stacked bar chart of device category distribution for each vendor
fig, ax = plt.subplots(figsize=(16, 9))  # Increase the figure width to make more space for the legend
x = np.arange(len(sorted_vendors))

# Set the background color to white
ax.set_facecolor('white')
fig.set_facecolor('white')

# Plot the stacked bars
colors = plt.cm.tab20.colors
bottom = np.zeros(len(sorted_vendors))
for i, category in enumerate(device_categories.keys()):
    counts = [vendor_device_counts[vendor][category] for vendor in sorted_vendors]
    ax.bar(x, counts, width=0.8, bottom=bottom, label=category, color=colors[i % len(colors)])
    bottom += counts

ax.set_xticks(x)
ax.set_xticklabels(sorted_vendors, rotation=40, ha='right')
plt.title('Device Function Distribution for Each Vendor')
plt.xlabel('Vendors')
plt.ylabel('Number of Devices')
plt.legend(loc='upper left', bbox_to_anchor=(0.8, 1), title='Device Functions', ncol=1)  # Move the legend box to the right
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



def get_top_external_services_by_vendor(df, vendor_name, top_n=5):
    services_counter = Counter()
    for _, row in df[df['vendor'].apply(lambda x: x[0].lower() == vendor_name.lower())].iterrows():
        for domain in row['valid_external_domains']:
            services_counter[domain] += 1
    return services_counter.most_common(top_n)



# Extract valid external domains
data['valid_external_domains'] = data['dns.qry.name'].apply(lambda x: [domain for domain in x if valid_external_domain(domain.lower())])

# Count common external services
external_services = Counter()
for domains in data['valid_external_domains']:
    for domain in domains:
        external_services[domain] += 1

# Plot common external services
top_services = external_services.most_common(10)
services, counts = zip(*top_services)
plt.figure(figsize=(10, 5))
plt.bar(services, counts)
plt.xlabel('External Services')
plt.ylabel('Count')
plt.title('Top 10 Common External Services')
plt.xticks(rotation=25, ha='right')
plt.show()


# Count common external services
vendors_counter = Counter()
for vendor in data['vendor']:
    vendors_counter[vendor[0]] += 1

# Plot common external services
top_vendors = vendors_counter.most_common(100)
vendors, counts = zip(*top_vendors)
plt.figure(figsize=(10, 5))
plt.bar(vendors, counts)
plt.xlabel('Vendors')
plt.ylabel('Count')
plt.title('Vendor Distribution')
plt.xticks(rotation=45, ha='right')
plt.show()

fields_to_present = ["dhcp.option.hostname", "http.user_agent", "dns.qry.name", "x509sat.printableString",utils.user_agents_google_field,utils.tls_issuers_google_field,utils.domains_google_field,utils.hostnames_google_field]

# Group the data by 'origin_dataset'
grouped_data = data.groupby('origin_dataset')

# Initialize an empty DataFrame to store the availability percentages
availability_percentages = pd.DataFrame()

# Calculate the availability for each group
for origin_dataset, dataset in grouped_data:
    non_null_percentages = dataset.apply(lambda x: sum(1 for i in x if i), axis=0) / len(dataset) * 100
    non_null_percentages = non_null_percentages.loc[fields_to_present]
    availability_percentages[origin_dataset] = non_null_percentages

# Calculate the weighted average availability across datasets
total_devices = len(data)
weighted_availability = availability_percentages.apply(lambda x: x * (grouped_data.size()[x.name] / total_devices), axis=0)
availability_percentages['Weighted Average'] = weighted_availability.sum(axis=1)

# Use readable labels for the index
readable_labels = {
    "dhcp.option.hostname": "DHCP Hostname",
    "http.user_agent": "HTTP User Agent",
    "dns.qry.name": "DNS Query Name",
    "x509sat.printableString": "X509 Printable String"
}
availability_percentages.index = availability_percentages.index.map(lambda x: readable_labels.get(x, x))

# Plot the data availability percentages for each field/column per dataset using a bar graph
ax = availability_percentages.plot.bar(figsize=(12, 6), stacked=False)
plt.title('Data Availability for Each Field/Column (Percentage) per Dataset')
plt.xlabel('Field/Column Name')
plt.ylabel('Percentage of Devices with Data')
plt.xticks(range(len(fields_to_present)), availability_percentages.index, rotation=45, ha='right')
plt.subplots_adjust(bottom=0.25)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Datasets', bbox_to_anchor=(1.05, 1), loc='upper left')

# Add data labels on top of the bars
#for container in ax.containers:
#    ax.bar_label(container)

plt.show()

# Calculate the percentage of non-empty values for each column
non_null_percentages = data.apply(lambda x: sum(1 for i in x if i), axis=0).sort_values(ascending=False) / len(data) * 100
non_null_percentages = non_null_percentages[non_null_percentages > 0]  # Exclude empty fields

# Select the required fields using loc method
fields_to_present = ["dhcp.option.hostname", "http.user_agent", "dns.qry.name", "x509sat.printableString","oui_vendor"]
# Create a dictionary to map the original field names to more readable labels
readable_labels = {
    "dhcp.option.hostname": "DHCP Hostname",
    "http.user_agent": "HTTP User Agent",
    "dns.qry.name": "DNS Query Name",
    "x509sat.printableString": "X509 Printable String"
}

# Update the index with readable labels
non_null_percentages = non_null_percentages.loc[fields_to_present]
non_null_percentages.index = non_null_percentages.index.map(lambda x: readable_labels.get(x, x))

# Create a colormap and assign colors for each bar
# Plot the data availability percentages for each field/column with different colors
fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size
non_null_percentages.plot(kind='bar', ax=ax)
plt.title('Data Availability for Each Field/Column (Percentage)')
plt.xlabel('Field/Column Name')
plt.ylabel('Percentage of Devices with Data')
plt.xticks(rotation=45, ha='right')  # Rotate labels and adjust their alignment
plt.subplots_adjust(bottom=0.25)  # Adjust bottom margin to prevent x-labels from being cut off
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add horizontal grid lines
plt.show()

import math



def truncate_label(label, max_length=25):
    return label[-max_length:] if len(label) > max_length else label


top_vendors_to_plot = 6
top_vendors = vendors_counter.most_common(top_vendors_to_plot)

n_vendors = len(top_vendors)
ncols = 2
nrows = math.ceil(n_vendors / ncols)


def get_top_external_services_by_vendor_percent(data, vendor, top_n=5, percentage=False):
    vendor_data = data[data['vendor'].apply(lambda x: x[0].lower() == vendor.lower())]
    total_devices = len(vendor_data)

    external_services_counter = Counter()
    for services in vendor_data['valid_external_domains']:
        for service in services:
            external_services_counter[service] += 1

    top_services = external_services_counter.most_common(top_n)

    if percentage and total_devices > 0:
        top_services = [(service, count / total_devices * 100) for service, count in top_services]

    return top_services

fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows), sharey=True)
plt.subplots_adjust(hspace=0.7, wspace=0.5)  # Increase hspace to avoid overlapping xticks

for idx, (vendor, _) in enumerate(top_vendors):
    ax = axes[idx // ncols, idx % ncols]
    top_services = get_top_external_services_by_vendor_percent(data, vendor,percentage=True)
    services, percentages = zip(*top_services)

    ax.bar(services, percentages)
    ax.set_title(f'Top 5 Domains for {vendor.capitalize()}')
    ax.set_ylabel('Percentage')
    ax.set_xlabel('Domains')
    ax.set_xticks(range(len(services)))
    ax.set_ylim(0, 110)  # Set the y-axis limits to be between 0 and 100
    ax.set_xticklabels([truncate_label(service) for service in services], rotation=25, ha='right')

# Remove empty subplots
for idx in range(n_vendors, nrows * ncols):
    fig.delaxes(axes[idx // ncols, idx % ncols])

plt.show()



# Group data by 'origin_dataset'
grouped_data = data.groupby('origin_dataset')

fig, axes = plt.subplots(len(grouped_data), 1, figsize=(10, 5 * len(grouped_data)), sharex=True)
plt.subplots_adjust(hspace=0.5)

for idx, (origin_dataset, dataset) in enumerate(grouped_data):
    non_null_counts = dataset.apply(lambda x: sum(1 for i in x if i), axis=0).sort_values(ascending=False)
    non_null_counts = non_null_counts[non_null_counts > 0]  # Exclude empty fields
    non_null_counts = non_null_counts.loc[fields_to_present]

    ax = axes[idx]
    non_null_counts.plot(kind='bar', ax=ax)
    ax.set_title(f'Data Availability for Each Field/Column ({origin_dataset})')
    ax.set_xlabel('Field/Column Name')
    ax.set_ylabel('Number of Devices with Data')
    ax.set_xticklabels(non_null_counts.index, rotation=45, ha='right')

plt.subplots_adjust(bottom=0.2)  # Adjust bottom margin to prevent x-labels from being cut off
plt.show()
