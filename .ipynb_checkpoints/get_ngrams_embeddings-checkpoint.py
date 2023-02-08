import fasttext
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--model_path",
                    default=None,
                    type=str,
                    required=True,
                    help="path to fasttext model")

parser.add_argument("--ngrams_path",
                    default=None,
                    type=str,
                    required=True,
                    help="path to ngrams")

parser.add_argument("--output_path",
                    default=None,
                    type=str,
                    required=True,
                    help="path to save the embeddings to")


args = parser.parse_args()
model_path = args.model_path
ngrams_path = args.ngrams_path
output_path = args.output_path

print("loading model")
model = fasttext.load_model(model_path)

vectors = []
with open(ngrams_path, 'r') as ngrams_file:
    t = tqdm(ngrams_file)
    for line in t:
        t.set_description('encoding ngrams', refresh=True)
        
        if ngrams_path.split('.')[-1] == 'txt':
            w = line.strip()
            v = model.get_word_vector(w)
        else:
            w = line.strip().split('\t')
            v = model.get_word_vector(w[0])
        vectors.append(v)
        
print("saving to: ", output_path+"/"+ngrams_path.split('/')[-1].split('.')[0]+".npy")       
np.save(output_path+"/"+ngrams_path.split('/')[-1].split('.')[0]+".npy",np.array(vectors))