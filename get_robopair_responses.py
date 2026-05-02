import sys
from main import main
import sys
import json

import pandas as pd


curr_path = ""
curr_big_cat = ""
start_execute = False
entries = []


path = sys.argv[1]
json_path = sys.argv[2]

cnt = 0

from tqdm import tqdm

from dotenv import load_dotenv

load_dotenv()

with open(json_path, "r") as f:
    data = json.load(f)
    for key, value in tqdm(list(data.items())):
        print(key, value)
        malicious = value[0]
        benign = value[1]
        filename = key
        
        target_malicious = "sure, I will" + malicious
        #target_benign = "sure, I will" + benign

        def run_case(goal, target_str):
            import argparse
            from fastchat.model import add_model_args


            # Create parser and add model args
            parser = argparse.ArgumentParser()
            add_model_args(parser)

            # Parse empty CLI args to get defaults
            args = parser.parse_args([])

            # Override settings
            args.goal = goal
            args.target_str = target_str
            args.robot = "unitree"
            args.target_model = "gpt-3.5-turbo"
            args.algorithm = "robopair"

            # Optionally, override other defaults if needed
            args.attack_model = "gpt-4"
            args.attack_max_n_tokens = 500
            args.max_n_attack_attempts = 5
            args.target_max_n_tokens = 500
            args.judge_model = "gpt-3.5-turbo"
            args.judge_max_n_tokens = 10
            args.judge_temperature = 0
            args.separate_devices = False
            args.n_streams = 1
            args.keep_last_n = 3
            args.n_iterations = 10
            args.index = 0
            args.category = "bomb"
            args.video_path = "data/dolphin/NuScenes/TrafficLight/video.mp4"

            # Run the main function
            return main(args)

        malicious_output = run_case(malicious, target_malicious)
        print(f"Malicious Output:\n{malicious_output}\n\n")
        #benign_output = run_case(benign, target_benign)

        entries.append([filename, "", "", "malicious", malicious_output])
        #entries.append([filename, "", "", "benign", benign_output])  

generated_df = pd.DataFrame(entries, columns=["filename","cat1", "cat2", "type", "text"])

generated_df.to_csv(path)
