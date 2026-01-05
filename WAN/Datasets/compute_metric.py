import subprocess
import sys

def run_vbench_evaluation(metadata_path, VBENCH_EVAL_SCRIPT):
    """
    Spawns a NEW shell that:
    1. deactivates current conda env (optional)
    2. activates conda env `vbench_eval`
    3. runs VBench evaluator with metadata
    """

    cmd = f"""
    bash -lc "
        conda deactivate;
        conda activate vbench_eval;
        python {VBENCH_EVAL_SCRIPT} --metadata {metadata_path}
    "
    """

    print("\n==============================")
    print("Launching VBench evaluation in vbench_eval environment...")
    print("==============================\n")

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print("VBench evaluation failed.")
    else:
        print("VBench evaluation completed successfully.")

if __name__ == "__main__":
    VBENCH_EVAL_SCRIPT = (
        "/mnt/alvaro/projects/Transformer-Engine-Finetune-Inference-main/"
        "inference/wan2.2/vbench_runner.py"
    )

    run_vbench_evaluation(OUTPUT_METADATA_PATH, VBENCH_EVAL_SCRIPT)