import json

def get_dataset(N_DATA = -1, type = "test"):
    data_dir = "/mnt/zhangchen/S3Precision/AIGC/WAN/Datasets"
    if type == "test":
        test_json = data_dir + "/MSR-VTT/msrvtt_test_1k.json"

    with open(test_json, "r") as f:
        msrvtt_items = json.load(f)

    if N_DATA >= 0: # 裁切data
        msrvtt_items = msrvtt_items[:N_DATA]
    return msrvtt_items
