import os
import json
import random
import tqdm
import re
import argparse
import pandas as pd
from collections import OrderedDict
from gpt3_api import make_requests as make_gpt3_requests


random.seed(42)

def output_prompt(inputs):
    return inputs + '\n' + "Output:"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        default="data/gpt3_generations/",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="machine_generated_inputs.jsonl",
        help="The path to the machine generated data.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="machine_generated_outputs.jsonl",
        help="The path to the machine generated data.",
    )
    parser.add_argument(
        "--num_outputs",
        type=int,
        default=100000,
        help="th",
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=5,
        help="The number of requests to send to GPT-3 at a time."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="The API key to use. If not specified, the key will be read from the environment variable OPENAI_API_KEY."
    )
    parser.add_argument(
        "--organization",
        type=str,
        help="The organization to use. If not specified, the default organization id will be used."
    )
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()

    with open(os.path.join(args.batch_dir, args.input_file)) as fin:
        lines = fin.readlines()
        if args.num_outputs is not None:
            lines = lines[:args.num_outputs]
        tasks = []
        for line in lines:
            data = json.loads(line)
            tasks.append(data)

    output_path = os.path.join(args.batch_dir, args.output_file)
    existing_requests = {}
    if os.path.exists(output_path):
        with open(output_path) as fin:
            for line in tqdm.tqdm(fin):
                try:
                    data = json.loads(line)
                    existing_requests[data["generation_input"]] = data
                except:
                    pass
        print(f"Loaded {len(existing_requests)} existing requests")
    if len(existing_requests) < args.num_outputs:
        progress_bar = tqdm.tqdm(total=len(tasks))
        tasks = tasks[len(existing_requests):]
        progress_bar.update(len(existing_requests))
        with open(os.path.join(args.batch_dir, "machine_generated_outputs.jsonl"), "a") as fout:
            for batch_idx in range(0, len(tasks), args.request_batch_size):
                batch = tasks[batch_idx: batch_idx + args.request_batch_size]
                
                prompts = []
                for task in batch:
                    prompts.append(output_prompt(task['generation_input']))
                results = make_gpt3_requests(
                    prompts=prompts,
                    max_tokens=1024,
                    temperature=0,
                    top_p=1,
                    stop_sequences=["\n", "\n\n"],
                    n=1,)
                for i in range(len(batch)):
                    response = ""
                    if results[i]["response"] is not None:
                        response = results[i]["response"]["choices"][0]["text"]
                    if not (prompts[i].count("Instruction: ")==1 and prompts[i].count("Input: ")==1 and prompts[i].count("Constraints: ")==1):
                        response = ""
                    generation_input = prompts[i].split('Constraints: ')[0] + 'Output:'
                    fout.write(json.dumps({
                        "generation_input": generation_input,
                        "generation_output": response
                    }, ensure_ascii=False) + "\n")
                progress_bar.update(len(batch))
    else:
        print("Generation Finished!!!")
                    
                    