import json

def load_json(file_path):
    with open(file_path, 'r') as file:
        dict_list=json.load(file)
    return dict_list

def load_jsonl(file_path):
    dict_list=[]
    with open(file_path, 'r') as file:
        for line in file:
            dict_list.append(json.loads(line))
    return dict_list

def save_json(data, file_path):
    json_data = json.dumps(data)
    with open(file_path, 'w') as file:
        file.write(json_data)

def get_advtext_filename(
        attack_name='',
        dataset_name='',
        victim_name='',
        num_examples=0,
        file_type='.json'
    ):

    return '_'.join([
        attack_name, dataset_name, victim_name, str(num_examples)
    ])+file_type