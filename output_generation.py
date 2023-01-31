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
    return inputs + '\n\n' + "Output:"

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
        "--num_inputs",
        type=int,
        default=5000,
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
        if args.num_inputs is not None:
            lines = lines[:args.num_inputs]
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

    progress_bar = tqdm.tqdm(total=len(tasks))
    with open(output_path, "w") as fout:
        for batch_idx in range(0, len(tasks), args.request_batch_size):
            batch = tasks[batch_idx: batch_idx + args.request_batch_size]
            if all(d["generation_input"] in existing_requests for d in batch):
                for d in batch:
                    data = existing_requests[d["generation_input"]]
                    fout.write(json.dumps({
                        "generation_input": data["generation_input"],
                        "generation_output": data["generation_output"]
                    }, ensure_ascii=False) + "\n")
                progress_bar.update(len(batch))
            else:
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
                    if results[i]["response"] is None:
                        continue
                    response = results[i]["response"]["choices"][0]["text"]
                    if response == "":
                        continue
                    fout.write(json.dumps({
                        "generation_input": prompts[i],
                        "generation_output": response
                    }, ensure_ascii=False) + "\n")
                    progress_bar.update(1)
                    
                    