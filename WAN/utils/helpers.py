import os
import json

def save_data_to_json(data, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)