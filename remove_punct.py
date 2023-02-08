import spacy
from spacy.lang.en import English
import pandas as pd
from tqdm import tqdm
import argparse

def main(data_path):
    training_data = pd.read_csv(data_path,names=['text'])

    nlp = spacy.load('en_core_web_sm')

    with open(data_path+'_noPunct.csv','w') as _file:
        t = tqdm(training_data['text'])
        for text in t:
            t.set_description('processing line', refresh=True)
            new_text=[]
            doc = nlp(text)
            if len(doc) > 64:
                for token in doc:
                    if not token.is_punct and not token.is_stop:
                        new_text.append(token.text)
                _file.writelines(' '.join(new_text)+'\n')

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path",
                        default=None,
                        type=str,
                        required=True,
                        help="path to data")
    

    
    args = parser.parse_args()
    data_path = args.data_path

    
    main(data_path)