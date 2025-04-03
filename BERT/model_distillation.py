import torch
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from utils import *

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset_path = "shawhin/phishing-site-classification"
model_path = "google-bert/bert-base-uncased"

id2label = {0: "Safe", 1: "Not Safe"}
label2id = {"Safe": 0, "Not Safe": 1}

dataset_dict = load_dataset(dataset_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, id2label=id2label, label2id=label2id).to(device)

model = freeze_parameters(model, unfreeze_layers=["pooler"])

print("Model parameters after freezing:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.requires_grad}")
        
tokenized_dataset = dataset_dict.map(lambda x: preprocess_text(x, tokenizer), batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

lr = 2e-4
batch_size = 8
n_epochs = 10

trainer_args = TrainingArguments(
    output_dir="bert-distillation-teacher",
    eval_strategy="epoch",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=n_epochs,
    logging_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=trainer_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
predictions = trainer.predict(tokenized_dataset["validation"])
logits = predictions.predictions
labels = predictions.label_ids
metrics = compute_metrics((logits, labels))
print(f"Metrics: {metrics}")

trainer.push_to_hub("bert-distillation-teacher")