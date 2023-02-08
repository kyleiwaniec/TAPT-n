
'''
train a model from the 'TAPT' data so we can extract warm-start embeddings for the ngrams to feed the T-DNA model training code.
the dimension of the vectors must be the same as the LLM we will be continuing to train. In this case roberta-large, which has dim=1024
since we will only be using unigrams and bigrams, we only need wordNgrams set to 2
'''


import fasttext
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data_path",
                    default=None,
                    type=str,
                    required=True,
                    help="path to data")

parser.add_argument("--output_path",
                    default=None,
                    type=str,
                    required=True,
                    help="path to save the model to")

parser.add_argument("--num_epochs",
                    default=3,
                    type=int,
                    required=True,
                    help="number of epochs to train for")

parser.add_argument("--dimension",
                    default=1024,
                    type=int,
                    required=True,
                    help="the dimension of the embeddings 1024 for roberta-large, and 768 for roberta-base")

args = parser.parse_args()
data_path = args.data_path
output_path = args.output_path
num_epochs = int(args.num_epochs)
dimension = int(args.dimension)

model = fasttext.train_unsupervised(data_path, 
                                    model='skipgram', 
                                    lr=0.05, 
                                    dim=dimension, 
                                    ws=4, 
                                    wordNgrams=2, 
                                    epoch=num_epochs, 
                                    thread=12)

print("saving to: ", output_path+"/"+data_path.split('/')[-1]+"_fasttext.bin")
model.save_model(output_path+"/"+data_path.split('/')[-1]+"_fasttext.bin")