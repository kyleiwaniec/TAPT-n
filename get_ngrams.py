#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import sys
import os
import argparse
import json
import vocabulary_utils as vu
import importlib
importlib.reload(vu)

logger = logging.getLogger(__name__)

"""

data: list or Series of strings, ex) df['snippet']
LLMvocab: list containing the LLM vocabulary

"""

def main(data_path,
         stopwords,
         spacy_model,
         LLMvocab_path,
         tokenizer,
         language,
         max_n,
         pmi,
         output_path):
    

    # read the LLM vocabulary file
    with open(LLMvocab_path) as json_file:
        vocab = json.load(json_file)
        
    # read the data file
    data = []
    with open(data_path) as _file:
        for line in _file.readlines():
            data.append(line.strip())
    
    if tokenizer == "sentencepiece":
        # xlm-roberta uses the sentencepiece tokenizer, 
        # which adds an underscore to word chunks
        spec_char = '▁'
    else:
        # roberta prepends Ġ to word chunks
        spec_char = 'Ġ'
        
    # remove special charcaters so we can later compare the LLM vocab
    # to the ngrams and remove ngrams already in the LLM vocab
    vocab_list = [s.replace(spec_char,'') for s in list(vocab.keys())]
    

    # returns [(ngram,freq),...]
    ngrams_list,_ = vu.getNgramsSpacy(data,
                                     stopwords=stopwords,
                                     spacy_model=spacy_model,
                                     LLMvocab=vocab_list,
                                     language=language,
                                     max_n=max_n,
                                     pmi=pmi)

    # save ngrams to file
    print('saving to: ', output_path+'/'+language+'_ngrams_'+str(max_n)+'.tsv')
    with open(output_path+'/'+language+'_ngrams_'+str(max_n)+'.tsv', 'w') as f:
        for item in ngrams_list:
            f.write("%s\t%s\n" % (item[0], item[1]))
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path",
                        default=None,
                        type=str,
                        required=True,
                        help="path to data")
    
    parser.add_argument("--model",
                        default='en_core_web_sm',
                        type=str,
                        required=False,
                        help="SpacCy model for tokenization")
    
    parser.add_argument("--stopwords_path",
                        default=None,
                        type=str,
                        required=False,
                        help="path to stopwords file if not using spacy")
    
    parser.add_argument("--LLMvocab_path",
                        default=None,
                        type=str,
                        required=False,
                        help="path to LLM vocabulary JSON file. You can usually find this in the language model directory called vocab.json")
    
    parser.add_argument("--tokenizer",
                        default='wordpiece',
                        type=str,
                        required=False,
                        help="language code")
    
    parser.add_argument("--language",
                        default='en',
                        type=str,
                        required=False,
                        help="tokenizer type")
    
    parser.add_argument("--max_n",
                        default=32768,
                        type=int,
                        required=False,
                        help="maximum number of ngrams to return")
    
    parser.add_argument("--pmi",
                        default=False,
                        type=bool,
                        required=False,
                        help="use pmi if true, otherwise use frequencies")
    
    parser.add_argument("--output_path",
                        default="output",
                        type=str,
                        required=True,
                        help="path to save the ngrams to")
    
    
    
    
    # Parse cli arguments
    args = parser.parse_args()
    
    data_path = args.data_path
    spacy_model = args.model
    stopwords = args.stopwords_path
    LLMvocab_path = args.LLMvocab_path
    tokenizer = args.tokenizer
    language = args.language
    max_n = int(args.max_n)
    pmi = args.pmi
    output_path = args.output_path
    
    main(data_path,
         stopwords,
         spacy_model,
         LLMvocab_path,
         tokenizer,
         language,
         max_n,
         pmi,
         output_path)
    