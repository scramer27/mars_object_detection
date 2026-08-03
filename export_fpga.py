import glob
import os
from ultralytics import YOLO

# 1. Automatically find your dataset YAML file
yaml_files = glob.glob("**/*.yaml", recursive=True)
# Filter out venv or hidden folders
yaml_files = [f for f in yaml_files if ".venv" not in f and "venv" not in f]

if not yaml_files:
    raise FileNotFoundError("Could not find any dataset .yaml file in your project!")

# Prioritize files with 'mars' or 'data' in the name
dataset_yaml = yaml_files[0]
for y in yaml_files:
    if "mars" in y.lower() or "ai4mars" in y.lower():
        dataset_yaml = y
        break

print(f"--> Found calibration dataset config: {dataset_yaml}")