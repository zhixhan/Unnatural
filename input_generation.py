import os
import json
import random
import re
import string
import tqdm
import argparse
import numpy as np
import pandas as pd
from gpt3_api import make_requests as make_gpt3_requests


random.seed(42)

def consrtuct_demonstrations(seed_tasks):
    demonstrations = []
    for task in seed_tasks:
        demonstration = ""
        demonstration += "Instruction: " + task['Instruction'] + '\n'
        demonstration += "Input: " + task['Input'] + '\n'
        demonstration += "Constraints: " + task['Constraints']
        demonstrations.append(demonstration)
    return demonstrations

def encode_prompt(demonstrations):
    """Encode multiple prompt instructions into a single string."""
    prompt = ""
    for idx, demonstration in enumerate(demonstrations):
        prompt += "Example" + str(idx+1) + '\n'
        prompt += "Instruction: " + demonstration['Instruction'] + '\n'
        prompt += "Input: " + demonstration['Input'] + '\n'
        prompt += "Constraints: " + demonstration['Constraints'] + '\n'
        prompt += '\n'
    prompt += "Example" + str(idx+2) + '\n'
    return prompt


def find_word_in_string(w, s):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search(s)


def post_process_gpt3_response(responses):
    results = []
    for response in responses:
        if response is None or response["finish_reason"] == "length":
            continue
        text = response['text']
        if not (text.count("Instruction: ")==1 and text.count("Input: ")==1 and text.count("Constraints: ")==1):
            continue
        _instruction = text.split("Instruction: ")[1].split("Input: ")[0]
        _input = text.split("Input: ")[1].split("Constraints: ")[0]
        _constraints = text.split("Constraints: ")[1]
        
        # filter based on keywords that are not suitable for language models.
        if any(find_word_in_string(word, _instruction) for word in ["image", "images", "graph", "graphs", "picture", "pictures", "file", "files", "map", "maps", "draw", "plot", "go to"]):
            continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit comfusing whether the model need to write a program or directly output the result. 
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if _instruction.startswith("Write a program"):
            continue
        
        if text in results:
            continue
       
        results.append(text)
    return results
 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        default="data/gpt3_generations/",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--seed_tasks",
        type=str,
        default="1, 2, 3, 4, 5",
        help="The path to the human written data.",
    )
    parser.add_argument(
        "--num_inputs_to_generate",
        type=int,
        default=100000,
        help="th",
    )
    
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=20,
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


if __name__ == "__main__":
    args = parse_args()
    all_seed_tasks = []
    all_seed_demonstrations = []
    seed_tasks_num = [int(s.strip()) for s in args.seed_tasks.split(",")]
    for i in seed_tasks_num:
        seed_tasks = [json.loads(l) for l in open(os.path.join('data', 'seed'+str(i)+'.jsonl'), "r")]
        all_seed_tasks.append(seed_tasks)
        seed_demonstrations = consrtuct_demonstrations(seed_tasks)
        all_seed_demonstrations.extend(seed_demonstrations)
    print(f"Loaded {len(all_seed_demonstrations)} human-written seed demonstrations")
    os.makedirs(args.batch_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_instructions = []
    if os.path.exists(os.path.join(args.batch_dir, "machine_generated_inputs.jsonl")):
        with open(os.path.join(args.batch_dir, "machine_generated_inputs.jsonl"), "r") as fin:
            for line in fin:
                instruction_info = json.loads(line)
                machine_instructions.append(instruction_info["generation_input"])
                request_idx = instruction_info["request_idx"] + 1
        print(f"Loaded {len(machine_instructions)} machine-generated inputs")

    if len(machine_instructions) < args.num_inputs_to_generate:
        
        # now let's generate new inputs!
        progress_bar = tqdm.tqdm(total=args.num_inputs_to_generate)
        if machine_instructions:
            progress_bar.update(len(machine_instructions))

        with open(os.path.join(args.batch_dir, "machine_generated_inputs.jsonl"), "a") as fout:
            while len(machine_instructions) < args.num_inputs_to_generate:
                seed_tasks = random.choice(all_seed_tasks)
                prompts = [encode_prompt(seed_tasks)]
                results = make_gpt3_requests(
                    prompts=prompts,
                    max_tokens=1024,
                    temperature=1,
                    top_p=0.99,
                    stop_sequences=["\n\n", "\n16", "16.", "16 ."],
                    n=args.request_batch_size,
                )
                results = [r['response'] for r in results][0]['choices']
                results = post_process_gpt3_response(results)
                post_results = []
                for res in results:
                    if res in all_seed_demonstrations or res in machine_instructions:
                        continue
                    machine_instructions.append(res)
                    fout.write(json.dumps({
                        "generation_input": res,
                        "request_idx": request_idx
                    }) + "\n")
                    progress_bar.update(1)

                request_idx += 1
    else:
        print("Generation Finished!!!")