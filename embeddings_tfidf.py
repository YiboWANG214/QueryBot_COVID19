import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import torch
from torch.optim import AdamW
from transformers import AutoModel, AutoConfig, BertForMaskedLM, BertTokenizer
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# calculate embeddings
data = pd.read_csv('sentences_clean_noteID_3.csv', encoding='utf-8')
notes_line = data['clean']
print(notes_line[:5])

embeddings = []
  
vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, max_features=1000)
X = vectorizer.fit_transform(notes_line)
print(X)
X = X.toarray()
pickle.dump(vectorizer.vocabulary_,open("feature.pkl","wb"))

embeddings = list(X)
print(embeddings[:5])


data['embeddings'] = embeddings
data.to_csv('embeddings_tfidf.csv', index=False)
