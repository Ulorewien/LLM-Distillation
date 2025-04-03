import numpy as np
import evaluate

def freeze_parameters(model, unfreeze_layers=[]):
    """
    Freeze all parameters in the model except for the specified layers.
    
    Args:
        model: The model whose parameters are to be frozen.
        unfreeze_layers: List of layer names to unfreeze. If empty, all layers will be frozen.
    """
    for name, param in model.base_model.named_parameters():
        if any(layer in name for layer in unfreeze_layers):
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    return model

def preprocess_text(data, tokenizer):
    """
    Preprocess the text data using the provided tokenizer.
    
    Args:
        data: The text data to be preprocessed.
        tokenizer: The tokenizer to use for preprocessing.
        
    Returns:
        tokenized_data: The preprocessed data.
    """
    tokenized_data = tokenizer(data["text"], truncation=True)
    return tokenized_data

def compute_metrics(eval_pred):
    """
    Compute the metrics for the evaluation predictions.
    
    Args:
        eval_pred: The evaluation predictions.
        
    Returns:
        metrics: The computed metrics.
    """
    accuracy = evaluate.load("accuracy")
    auc_score = evaluate.load("roc_auc")

    logits, labels = eval_pred
    probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    pos_probs = probs[:, 1]
    auc = np.round(auc_score.compute(predictions_scores=pos_probs, references=labels)["roc_auc"], 3)
    pred = np.argmax(logits, axis=1)
    acc = np.round(accuracy.compute(predictions=pred, references=labels)["accuracy"], 3)
    metrics = {"accuracy": acc, "auc": auc}
    return metrics