from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import torch
import numpy as np

# export CUDA_VISIBLE_DEVICES=0,2
# device = "cuda:1" # if torch.cuda.is_available() else "cpu"

chkp = "Kyleiwaniec/PTC_TAPT_RoBERTa_large"
model = AutoModelForSequenceClassification.from_pretrained(chkp, use_auth_token='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', num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(chkp, use_auth_token='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')

dataset = load_dataset('Kyleiwaniec/PTC_Corpus', use_auth_token='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')

# def preprocess_function(examples):
#     return tokenizer(examples["text"], padding=True, truncation=True, return_tensors="pt")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

classification = 'binary' #'binary'

def update_labels(example):
    example['labels'] = example['labels'][0] if len(example['labels']) else 14
    return example


if classification == 'multi':
    # For multiclass classification use the technique classification as labels
    dataset = dataset.rename_column("labels", "binary_labels")
    dataset = dataset.rename_column("technique_classification", "labels")
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(['article_id', 'text', 'binary_labels', 'offsets'])
    
    # use only the first label.
    tokenized_dataset = tokenized_dataset.map(update_labels, num_proc=4)
    
else: #binary
    # binary classification
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(['article_id', 'text', 'technique_classification', 'offsets'])



tiny_train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(100))
tiny_eval_dataset = tokenized_dataset["validation"].shuffle(seed=42).select(range(10))

small_train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_dataset["validation"].shuffle(seed=42).select(range(1000))

train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["validation"]

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


#,"matthews_correlation","f1","precision","recall"
metrics = load_metric("f1","matthews_correlation")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return  metrics.compute(predictions=predictions, references=labels)

model_name = chkp.split("/")[-1]
out_dir = "../models/PTC_TAPT_n_RoBERTa_SLC_PTC/"

#no_cuda=True
training_args = TrainingArguments(
    output_dir=out_dir,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01
)

# compute_metrics=compute_metrics,
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model()