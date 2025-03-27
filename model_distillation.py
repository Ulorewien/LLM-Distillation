import os
import pandas as pd
from datasets import load_dataset
from transformers import pipeline
from utils import *


data = load_dataset("mteb/tweet_sentiment_extraction")

start_idx, end_idx = 0, 5000
output_folder = "data\\train"
to_csv(data, start_idx, end_idx, output_folder)
raw_data = pd.read_csv(os.path.join(output_folder, f"twitter_sentiment_{start_idx}_{end_idx}.csv"))
final_data = raw_data[raw_data["text"].notna() & (raw_data["text"].str.strip() != "")]
final_data.to_csv(os.path.join(output_folder, f"twitter_sentiment_{start_idx}_{end_idx}_cleaned.csv"), index=False)

start_idx, end_idx = 5001, 6000
output_folder = "data\\test"
to_csv(data, start_idx, end_idx, output_folder)
raw_data = pd.read_csv(os.path.join(output_folder, f"twitter_sentiment_{start_idx}_{end_idx}.csv"))
final_data = raw_data[raw_data["text"].notna() & (raw_data["text"].str.strip() != "")]
final_data.to_csv(os.path.join(output_folder, f"twitter_sentiment_{start_idx}_{end_idx}_cleaned.csv"), index=False)


chain_of_thought_prompt = get_prompt("prompts/llama_3_1_prompt.json")
pipe = pipeline("text-generation", model="meta-llama/Llama-3.1-405B-Instruct")

messages = [
    {
        "role": "system",
        "content": chain_of_thought_prompt
    },
    {
        "role": "user",
        "content": "I'm so happy today!"
    }
]