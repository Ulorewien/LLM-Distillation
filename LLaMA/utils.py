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
    
def llama_pipeline(pipe, input_file, output_file, system_prompt):    
    with open(input_file, "r", newline="", encoding="utf-8") as infile, \
         open(output_file, "w", newline="", encoding="utf-8") as outfile:
             
        reader = csv.DictReader(infile)
        cols = reader.fieldnames + ["llama_3.1_reasoning", "llama_3.1_label"]
        
        writer = csv.DictWriter(outfile, fieldnames=cols)
        writer.writeheader()
        
        for i, row in enumerate(reader):
            user_prompt = row["text"]
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
            
            try:
                #TODO: Check the reponse and update the code once we get access to LLaMA 3.1 API
                response = pipe(messages)
            except Exception as e:
                print(f"Error processing row {i}: {e}")
        
    print(f"Processed {input_file} and saved to {output_file}")