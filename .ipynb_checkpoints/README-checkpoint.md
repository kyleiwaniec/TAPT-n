# Task Adaptive Pre-Training with N-grams (TAPT-n)

This code is based on the T-DNA repo (https://github.com/shizhediao/T-DNA) with the following changes: 

* Corrections were made to get the code to run.
* The ngrams script referenced in T-DNA was missing. To generate ngrams use the t-dna-ngrams-*.ipynb notebooks in this repo.
    * The original paper calls for the use of PMI to decide which ngrams to keep. Since all ngrams in our dataset have a negative PMI, we simply use the ones with the highest frequency.
    * The embeddings are generated using FASTTEXT, and saved as a numpy array (in the code this numpy array is refered to as the "model", but it is not the actual fasttext model bin file). To train FASTTEXT use fasttext-train*.ipynb notebooks.
    * If you need to generate ngrams, you will need to insall SpaCy - it is not included in the requirements.txt file.
* The tokenizer class for xlm-roberta was added to the tokenization.py file 
    * The source of the vocabulary was also updated to play nice with xlm
    * And the run-language-modeling-xlm.py file was also modified accordingly
    * All the other classes for xlm inherit from roberta without any changes 🎉
* To perform TAPT (task adaptive pre-training) run the train-mlm-xlm.sh script
    * Training takes approximately 6 hours (3 epochs) on a p2 or G4 GPU in Sagemaker 
    * The model output from this phase will need to be fine-tuned on a downstream task such as sentence similarity or classification
    
* Finally, you will need to download the relevant model files from huggingface and change the paths in the scripts accordingly.


### T-DNA Citation:
```
@inproceedings{DXSJSZ2021,
    title = "Taming Pre-trained Language Models with N-gram Representations for Low-Resource Domain Adaptation",
    author = "Diao, Shizhe  and
      Xu, Ruijia  and
      Su, Hongjin  and
      Jiang, Yilei  and
      Song, Yan  and
      Zhang, Tong",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.259",
    doi = "10.18653/v1/2021.acl-long.259",
    pages = "3336--3349",
}
```
