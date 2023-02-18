from transformers import pipeline
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report

dataset = load_dataset('Kyleiwaniec/PTC_Corpus', use_auth_token='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')

chkp = "../models/PTC_TAPT_RoBERTa_large_SLC/"

tokenizer = AutoTokenizer.from_pretrained(chkp)
classifier = pipeline("text-classification", model=chkp, tokenizer=tokenizer, batch_size=64, device=0,num_workers=160)

predictions_binary=[]
t = tqdm(dataset['test'])
for i in t:
    t.set_description("processing:", refresh=True)
    pred = classifier(i['text']) # 'LABEL_0'
    y_hat = int(pred[0]['label'].split('_')[1])
    score = pred[0]['score'] if y_hat == 1 else 1-pred[0]['score']
    y = i['labels']
    predictions_binary.append([y_hat,y,score])
    
print(len(predictions_binary))

y_true_b = np.array(predictions_binary)[:,1]
y_pred_b = np.array(predictions_binary)[:,0]

print(classification_report(y_true_b, y_pred_b))