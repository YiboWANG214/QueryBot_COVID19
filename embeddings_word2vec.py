import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import torch
from torch.optim import AdamW
from transformers import AutoModel, AutoConfig, BertForMaskedLM, BertTokenizer
from transformers import AutoTokenizer
import nltk
from nltk.corpus import stopwords
import json
import csv
import string

nltk.download('punkt')
nltk.download('stopwords')
words = stopwords.words('english')
words.append('patient')

def clean(sent):
    sent = sent.translate(str.maketrans('', '', string.punctuation))
    remove_digits = str.maketrans('', '', string.digits)
    sent = sent.translate(remove_digits)
    sent = sent.lower().split()
    sent = [word for word in sent if word not in words]
    sent = ' '.join(sent)

    return sent

# calculate embeddings
data = pd.read_csv('sentences_clean_noteID_3.csv', encoding='utf-8')
notes_line = data['clean']

embeddings = []

corpus = []
for sent in notes_line:
    corpus.append(nltk.word_tokenize(clean(sent)))

tmp_model = KeyedVectors.load_word2vec_format('PubMed-and-PMC-w2v.bin', binary=True)

model = Word2Vec(size=200, min_count=1)
model.build_vocab(corpus)
total_examples = model.corpus_count
model.build_vocab([list(tmp_model.vocab.keys())], update=True)
model.intersect_word2vec_format('PubMed-and-PMC-w2v.bin', binary=True)
model.train(corpus, total_examples=total_examples, epochs=3)
#model.wv.save_Word2Vec_format("finetuned_word2vec.bin",binary=True)
model.save("finetuned_word2vec.model")

for line in notes_line:
    line = nltk.word_tokenize(clean(line))
    tmp = [[0]*200]
    for word in line:
        if word in model.wv.vocab:
            tmp.append(model[word])
    tmp = list(np.mean(tmp, axis=0))
    embeddings.append(tmp)

data['embeddings'] = embeddings
data.to_csv('embeddings_word2vec_finetuned.csv', index=False)


