from transformers import pipeline
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
from transformers.pipelines.pt_utils import KeyDataset, KeyPairDataset

classification_type = "multi" #binary

def update_labels(example):
    example['labels'] = example['labels'][0] if len(example['labels']) else 18
    return example


dataset = load_dataset('Kyleiwaniec/PTC_Corpus',
                       split="validation",
                       use_auth_token='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')

if classification_type == "multi":
    dataset = dataset.rename_column("labels", "binary_labels")
    dataset = dataset.rename_column("technique_classification", "labels")

    # use only the first label.
    dataset = dataset.map(update_labels, num_proc=4)


chkp = "../models/TAPT_n_RoBERTa_TC_PTC/"

tokenizer = AutoTokenizer.from_pretrained(chkp)
pipe = pipeline("text-classification", 
                      model=chkp, 
                      tokenizer=tokenizer, 
                      batch_size=64, 
                      device=0,
                      num_workers=160)

predictions=[]
t = tqdm(dataset)
for i in t:
    t.set_description("processing:", refresh=True)
    pred = pipe(i['text']) # 'LABEL_0'
    y_hat = int(pred[0]['label'].split('_')[1])
#     score = pred[0]['score'] if y_hat == 1 else 1-pred[0]['score']
    y = i['labels']
#     predictions_binary.append([y_hat,y,score])
    predictions.append([y_hat,y])
    
print(len(predictions))

y_true_b = np.array(predictions)[:,1]
y_pred_b = np.array(predictions)[:,0]

print(classification_report(y_true_b, y_pred_b))

