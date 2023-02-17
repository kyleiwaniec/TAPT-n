from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import torch
import numpy as np

# export CUDA_VISIBLE_DEVICES=0,2
# device = "cuda:1" # if torch.cuda.is_available() else "cpu"

chkp = "models/PTC_TAPT_n_RoBERTa"

model = AutoModelForSequenceClassification.from_pretrained(chkp, num_labels=2)
model = model #.to(device)

tokenizer = AutoTokenizer.from_pretrained(chkp)
tokenizer = tokenizer


dataset = load_dataset('Kyleiwaniec/PTC_Corpus', use_auth_token='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')

def preprocess_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True, return_tensors="pt") #.to(device)

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


#no_cuda=True
training_args = TrainingArguments(
    output_dir=chkp+"_tiny_PTC/",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=48,
    per_device_eval_batch_size=48,
    num_train_epochs=3,
    weight_decay=0.01
)

# compute_metrics=compute_metrics,
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tiny_train_dataset,
    eval_dataset=tiny_eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print(torch.cuda.current_device())

trainer.train()
trainer.save_model()