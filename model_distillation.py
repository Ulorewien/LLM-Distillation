import os
import pandas as pd
from datasets import load_dataset
from transformers import pipeline
from utils import *


data = load_dataset("mteb/tweet_sentiment_extraction")

start_idx, end_idx = 0, 5000
output_folder = "data\\train"
clean_data(data, start_idx, end_idx, output_folder)

start_idx, end_idx = 5001, 6000
output_folder = "data\\test"
clean_data(data, start_idx, end_idx, output_folder)


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