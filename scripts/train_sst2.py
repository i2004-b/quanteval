# scripts/train_sst2.py
from datasets import load_dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments

# 1. Load SST-2 dataset (from the GLUE benchmark via HuggingFace Datasets)
datasets = load_dataset('glue', 'sst2')
# This returns a dict with 'train', 'validation', and 'test' splits. GLUE's test split has no labels, 
# so we'll use 'validation' for evaluation.
train_dataset = datasets['train']
val_dataset = datasets['validation']

# 2. Initialize DistilBERT tokenizer and model for sequence classification
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 3. Preprocess the text data: tokenize the sentences
def preprocess_function(examples):
    # Tokenize the texts (input could be two sentences for pair tasks, but SST-2 has single sentence)
    return tokenizer(examples['sentence'], truncation=True, padding=True, max_length=128)
    
train_enc = train_dataset.map(preprocess_function, batched=True)
val_enc = val_dataset.map(preprocess_function, batched=True)
# The above adds 'input_ids', 'attention_mask' fields to the dataset dicts.

# Set the format for PyTorch
train_enc.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_enc.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# 4. Define training arguments for the Trainer API
training_args = TrainingArguments(
    output_dir="../models/distilbert_baseline",   # directory to save model files
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=5e-5,
    evaluation_strategy="epoch",  # evaluate at end of each epoch
    save_strategy="epoch",        # save at each epoch
    logging_strategy="epoch",
    logging_dir="../models/distilbert_baseline/logs",
    report_to="none",            # disable default logging to WandB or others
    no_cuda=False               # will use CUDA if available
)

# 5. Define a simple compute_metrics function to compute accuracy
import numpy as np
from datasets import load_metric
metric_accuracy = load_metric('accuracy')
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = metric_accuracy.compute(predictions=preds, references=labels)
    return {"accuracy": acc["accuracy"]}

# 6. Initialize Trainer and train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_enc,
    eval_dataset=val_enc,
    compute_metrics=compute_metrics
)

print("Starting training DistilBERT on SST-2...")
trainer.train()
# Evaluate on validation set after training
eval_results = trainer.evaluate()
val_accuracy = eval_results.get("eval_accuracy")
print(f"Baseline DistilBERT Accuracy on SST-2 validation set: {val_accuracy*100:.2f}%")

# 7. Save the fine-tuned model (this saves to training_args.output_dir)
trainer.save_model()
tokenizer.save_pretrained(training_args.output_dir)
print(f"Saved DistilBERT baseline model to {training_args.output_dir}/")
