import os
import csv
import json
import pandas as pd

def to_csv(data, start, end, output_folder):
    output_path = os.path.join(output_folder, f"twitter_sentiment_{start}_{end}.csv")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    with open(output_path, mode='w', newline='', encoding="utf-8") as file:
        cols = ["id", "text", "label", "label_text"]
        writer = csv.DictWriter(file, fieldnames=cols)
        writer.writeheader()
        for i in range(start, min(end, len(data["train"]))):
            row = data["train"][i]
            writer.writerow(row)
            
    print(f"Saved {output_path}")
    
def clean_data(data, start_idx, end_idx, output_folder):
    to_csv(data, start_idx, end_idx, output_folder)
    raw_data = pd.read_csv(os.path.join(output_folder, f"twitter_sentiment_{start_idx}_{end_idx}.csv"))
    final_data = raw_data[raw_data["text"].notna() & (raw_data["text"].str.strip() != "")]
    final_data.to_csv(os.path.join(output_folder, f"twitter_sentiment_{start_idx}_{end_idx}_cleaned.csv"), index=False)
    
def get_prompt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        
    return data["prompt"]
    