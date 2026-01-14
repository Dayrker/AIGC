# utils
from utils.helpers import save_data_to_json
from utils.parse import parse_args
# Inference
from Inference.infer_batch import infer_batch, infer_batch_multiprocessing
# Dataset
from Datasets.load_data import get_dataset

if __name__ == "__main__":
    # Get parameters first.
    args = parse_args()

    # Get datasets
    dataset = get_dataset(N_DATA=24, type="test")

    # get outputs
    if len(args.cuda.split(",")) <= 1:
        outputs = infer_batch(args, dataset=dataset, device_id=int(args.cuda))
    else:
        outputs = infer_batch_multiprocessing(args, dataset=dataset)

    # save outputs
    save_data_to_json(
        outputs,
        f"./output/{args.model}/{args.size}/{args.arch}/{args.precision}/result_all.json"
    )
