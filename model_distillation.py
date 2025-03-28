import os
import pandas as pd
from datasets import load_dataset
from transformers import pipeline
from utils import *


data = load_dataset("mteb/tweet_sentiment_extraction")

start_idx, end_idx = 0, 5000
trin_output_folder = "data\\train"
clean_data(data, start_idx, end_idx, trin_output_folder)

start_idx, end_idx = 5001, 6000
test_output_folder = "data\\test"
clean_data(data, start_idx, end_idx, test_output_folder)


chain_of_thought_prompt = get_prompt("prompts/llama_3_1_prompt.json")
pipe = pipeline("text-generation", model="meta-llama/Llama-3.1-405B-Instruct")

llama_pipeline(
    pipe, 
    input_file=os.path.join(test_output_folder, f"twitter_sentiment_{start_idx}_{end_idx}_cleaned.csv"),
    output_file=os.path.join(test_output_folder, f"twitter_sentiment_{start_idx}_{end_idx}_llama_3_1.csv"),
    system_prompt=chain_of_thought_prompt
)