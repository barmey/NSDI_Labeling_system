import search_api
import utils
from tqdm import tqdm

def collect_data_from_google(queries):
    dict_hostnames = {}
    print('Collect Data From Google')
    for query in tqdm(queries):
        dict_hostnames[query] = search_api.send_search_request(query)
    return dict_hostnames


def get_and_save_search_tofile(file_to_save_path,queries):
    utils.save_dict_to_file_pkl(collect_data_from_google(queries),file_to_save_path)








