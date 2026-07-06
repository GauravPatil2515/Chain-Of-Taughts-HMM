"""
Batch runner to execute the ensemble pipeline on large datasets.
"""
import os
import time
import subprocess
from pathlib import Path

def monitor_and_run(input_file: str, output_dir: str):
    print(f"Waiting for {input_file} to be ready...")
    
    while not os.path.exists(input_file):
        time.sleep(10)
        
    print(f"Found {input_file}. Waiting for generation to finish...")
    
    # Simple check: wait until file size stops changing for 1 minute
    prev_size = -1
    stable_count = 0
    
    while stable_count < 6: # 1 minute stability
        current_size = os.path.getsize(input_file)
        if current_size == prev_size and current_size > 0:
            stable_count += 1
        else:
            stable_count = 0
            if current_size > 0:
                 print(f"File growing: {current_size} bytes...")
        prev_size = current_size
        time.sleep(10)
        
    print("File stable. Starting pipeline...")
    
    cmd = [
        "python", "run_ensemble_pipeline.py",
        "--input", input_file,
        "--output-dir", output_dir
    ]
    
    subprocess.run(cmd)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    
    monitor_and_run(args.input, args.output_dir)
