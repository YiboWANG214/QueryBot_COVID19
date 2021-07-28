#pip install transformers
import pandas as pd
import numpy as np
from gensim.models import Word2Vec

# prepare bert
import torch
from torch.optim import AdamW
from transformers import AutoModel, AutoConfig, BertForMaskedLM, BertTokenizer
from transformers import AutoTokenizer
import nltk
from nltk.corpus import stopwords
import json
import csv
import string

output_dir = './model_save_clinical/'

nltk.download('punkt')
nltk.download('stopwords')
words = stopwords.words('english')
words.append('patient')


if torch.cuda.is_available():        
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Load a trained model and vocabulary that you have fine-tuned
model = BertForMaskedLM.from_pretrained(output_dir,
                                        output_attentions = False, # Whether the model returns attentions weights.
                                        output_hidden_states = True, # Whether the model returns all hidden-states.
                                        )
tokenizer = AutoTokenizer.from_pretrained(output_dir)

# Copy the model to the GPU.
model.to(device)
model.eval()


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

t = 0
for i in range(len(notes_line)):
    sentence = notes_line[i]
    t = t+1
    marked_text = "[CLS] " + sentence + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text,
                                        # max_length = 128
                                        )
    if len(tokenized_text) > 500:
       data = data.drop([i])
       continue
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    segments_ids = [1] * len(tokenized_text)
    segments_tensors = torch.tensor([segments_ids])
    segments_tensors = torch.tensor([segments_ids]).to(device)

    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        # outputs = pretrained_model(tokens_tensor, segments_tensors)
        hidden_states = outputs[1]
        token_vecs = hidden_states[-2][0]
        # Calculate the average of all token vectors.
        sentence_embedding = torch.mean(token_vecs, dim=0)
    embeddings.append(sentence_embedding.tolist())


data['embeddings'] = embeddings
data.to_csv('embeddings_clinical.csv', index=False)
