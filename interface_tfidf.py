import time
import os
import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import ttk
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, BertForMaskedLM, BertTokenizer
import nltk
from nltk.corpus import stopwords
import json
import csv
import string
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
import pickle

start = time.clock()

nltk.download('punkt')
nltk.download('stopwords')
words = stopwords.words('english')
words.append('patient')

transformer = TfidfTransformer()
loaded_vec = TfidfVectorizer(decode_error='replace', vocabulary=pickle.load(open("feature.pkl", "rb")))

def base_path(path):
    if getattr(sys, 'frozen', None):
        basedir = sys._MEIPASS
    else:
        basedir = os.path.dirname(__file__)
    return os.path.join(basedir, path)


def clean(sent):
    sent = sent.translate(str.maketrans('', '', string.punctuation))
    remove_digits = str.maketrans('', '', string.digits)
    sent = sent.translate(remove_digits)
    sent = sent.lower().split()
    sent = [word for word in sent if word not in words]
    sent = ' '.join(sent)
    return sent


def embedding(sent):
    my_array = np.array(clean(sent).split())
    sentence_embedding = transformer.fit_transform(loaded_vec.fit_transform(my_array)).todense()
    sentence_embedding = np.array(np.mean(sentence_embedding, axis=0)).reshape(1000,)
    return sentence_embedding


def calculate(original_q, data_):
    q = clean(original_q)
    files = os.listdir('./records_tfidf/json/relevant')
    sentence_embedding_relevant = np.zeros(1000)
    sentence_embedding_non = np.zeros(1000)
    for json_file in files:
        if q.replace(' ', '') + '.json' == json_file:
            if not os.path.exists('./records_tfidf/json/relevant/' + json_file):
                os.makedirs('./records_tfidf/json/relevant/' + json_file)
                with open('./records_tfidf/json/relevant/' + json_file, 'a') as ffile:
                    ffile.write(original_q)
            with open('./records_tfidf/json/relevant/' + json_file) as load_f:
                ff = load_f.readlines()
                new_dict = json.loads(ff[-1])
                for s in new_dict[q]:
                    sentence_embedding_relevant += np.array(embedding(clean(s)))
                if len(new_dict[q]) == 0:
                    sentence_embedding_relevant = np.zeros(1000)
                else:
                    sentence_embedding_relevant = np.array(sentence_embedding_relevant) / len(new_dict[q])
            break
    files2 = os.listdir('./records_tfidf/json/non')
    for json_file in files2:
        if q.replace(' ', '')+'.json' == json_file:
            if not os.path.exists('./records_tfidf/json/non/'+json_file):
                os.makedirs('./records_tfidf/json/non/'+json_file)
                with open('./records_tfidf/json/non/'+json_file, 'a') as ffile2:
                    ffile2.write(original_q)
            with open('./records_tfidf/json/non/'+json_file) as load_f2:
                ff = load_f2.readlines()
                new_dict = json.loads(ff[-1])
                for s in new_dict[q]:
                    sentence_embedding_non += np.array(embedding(clean(s)))
                if len(new_dict[q]) == 0:
                    sentence_embedding_non = np.zeros(1000)
                else:
                    sentence_embedding_non = np.array(sentence_embedding_non) / len(new_dict[q])

    print(len(sentence_embedding_relevant), len(sentence_embedding_non))
    query_en = 1.0 * np.array(embedding(q)) + 0.75 * sentence_embedding_relevant - 0.35*sentence_embedding_non
    query_en = np.array(query_en).reshape(-1, 1000)

    result_feature = []
    result_sentence = []
    result_value = []
    result_id = []
    note_ID = []
    for j in range(len(data_)):
        sentence = data_.loc[j]['embeddings'].replace('\n', ' ')
        sentence = sentence[1:-1].split()
        sentence = np.array(sentence).reshape(-1, 1000)
        value = cosine_similarity(sentence, query_en)
        result_sentence.append(data_['Sentences'][j])
        note_ID.append(data_['Note_ID'][j])
        result_id.append(data_['Patient_ID'][j])
        result_feature.append(sentence[0])
        result_value.append(value)

    my_df = pd.DataFrame(data=result_sentence, columns=['sentence'])
    my_df['feature'] = result_feature
    my_df['score'] = result_value
    my_df['note_ID'] = note_ID
    my_df['id'] = result_id
    my_df = my_df.sort_values(by='score', ascending=False)

    sentences = my_df['sentence'].tolist()
    ids = my_df['id'].tolist()
    note_ids = my_df['note_ID'].tolist()

    sent = []
    with open('./records_word2vec/csv/'+q+'.csv', "a") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["query", "rank", "note_id", "patient_id", "sentences"])
        for m in range(min(20, len(ids))):
            input_contents = (note_ids[m], ids[m], sentences[m])
            sent.append(input_contents)
            writer.writerows([[original_q, m, note_ids[m], ids[m], sentences[m]]])
    print(sentences[:20])
    print(my_df['score'][:20])
    return sent, set(sentences[:min(20, len(ids))])


def calculate2(original_q, data_):
    q = clean(original_q)
    files = os.listdir('./records_tfidf/json/relevant')
    sentence_embedding_relevant = np.zeros(1000)
    sentence_embedding_non = np.zeros(1000)
    for json_file in files:
        if q.replace(' ', '') + '.json' == json_file:
            print(json_file, '1')
            if not os.path.exists('./records_tfidf/json/relevant/' + json_file):
                os.makedirs('./records_tfidf/json/relevant/' + json_file)
                with open('./records_tfidf/json/relevant/' + json_file, 'a') as ffile:
                    ffile.write(original_q)
            with open('./records_tfidf/json/relevant/' + json_file) as load_f:
                ff = load_f.readlines()
                new_dict = json.loads(ff[-1])
                for s in new_dict[q]:
                    sentence_embedding_relevant += np.array(embedding(clean(s)))
                if len(new_dict[q]) == 0:
                    sentence_embedding_relevant = np.zeros(1000)
                else:
                    sentence_embedding_relevant = np.array(sentence_embedding_relevant) / len(new_dict[q])
            break

    files2 = os.listdir('./records_tfidf/json/non')
    for json_file in files2:
        if q.replace(' ', '')+'.json' == json_file:
            print(json_file, '2')
            if not os.path.exists('./records_tfidf/json/non/'+json_file):
                #if not os.path.exists('./records_tfidf/json/non/'+json_file):
                os.makedirs('./records_tfidf/json/non/' + json_file)
                with open('./records_tfidf/json/non/' + json_file, 'a') as ffile2:
                    ffile2.write(original_q)
            with open('./records_tfidf/json/non/'+json_file) as load_f2:
                ff = load_f2.readlines()
                new_dict = json.loads(ff[-1])
                for s in new_dict[q]:
                    my_array = np.array(embedding(clean(s)).tolist())
                    sentence_embedding_non += my_array
                if len(new_dict[q]) == 0:
                    sentence_embedding_non = np.zeros(1000)
                else:
                    sentence_embedding_non = np.array(sentence_embedding_non) / len(new_dict[q])

    query_en = 1.0 * np.array(embedding(q)) + 0.75 * sentence_embedding_relevant - 0.35*sentence_embedding_non
    query_en = np.array(query_en).reshape(-1, 1000)

    result_value = []
    label = []
    for j in range(len(data_)):
        sentence = data_.loc[j]['embedding']
        sentence = sentence.replace('\n', '')
        sentence = sentence.replace('  ', ' ')
        sentence = sentence[1:-1].split()
        sentence = np.array(sentence).reshape(-1, 1000)
        value = cosine_similarity(sentence, query_en)
        label.append(data_['label'][j])
        result_value.append(value)

    my_df = pd.DataFrame({'score': result_value, 'label':label})
    my_df = my_df.sort_values(by='score', ascending=False)

    labels = my_df['label'].tolist()

    return labels[0]


def func():
    str1 = entry_query.get()
    str2 = entry_patient.get()
    sentences_ = []
    if not str1:
        messagebox.askokcancel('Alert', 'Please enter query.')
        return

    # with patient information
    if str2 != '':
        data = pd.read_csv("./embeddings_tfidf.csv")
        patient_id = data['Patient_ID']
        index_ = []
        for j in range(len(data)):
            if str(patient_id[j]).strip() == str(str2).strip():
                index_.append(j)
        data_patient = data.iloc[index_]
        data_patient = data_patient.reset_index(drop=True)
        sentences_ = calculate(str1, data_patient)

    # without patient information
    else:
        
        data_mean = pd.read_csv('./data_tfidf/mean.csv')
        label = calculate2(str1, data_mean)
        print(label)
        file_name = './data_tfidf/data'+str(label)+'.csv'
        data = pd.read_csv(file_name)
        sentences_ = calculate(str1, data)
    
    print(len(sentences_))

    if len(sentences_) == 0:
        messagebox.askokcancel('Alert', 'No Matching')
    else:
        func2(sentences_, str1)


def func2(sentences_, s):
    str1 = clean(s)
    sentences = sentences_[0]
    s_set = sentences_[1]
    def func3():
        tmp_dict = dict()
        tmp_list = []
        for item in tree.selection():
            tmp_list.append(tree.item(item, "values")[2])
        tmp_dict[str1] = tmp_list
        with open('./records_tfidf/json/relevant/' + str1.replace(' ', '') + '.json', "a", encoding='utf-8') as f:
            json.dump(tmp_dict, f)
            f.write('\n')

        tmp_dict2 = dict()
        tmp_list2 = s_set-set(tmp_list)
        tmp_dict2[str1] = list(tmp_list2)
        with open('./records_tfidf/json/non/' + str1.replace(' ', '') + '.json', "a", encoding='utf-8') as f:
            json.dump(tmp_dict2, f)
            f.write('\n')
        tree.delete(*tree.get_children())

    tree = ttk.Treeview(window, selectmode=EXTENDED)
    tree["columns"] = ("note_id", "patient_id", "sentence")
    tree.column("#0", width=10)
    tree.column("note_id", width=50)
    tree.column("patient_id", width=80)
    tree.column("sentence", width=800)
    tree.heading("note_id", text="note_id")
    tree.heading("patient_id", text="patient_id")
    tree.heading("sentence", text="sentence")
    tree.place(x=10, y=80)
    i = 0
    for sentence in sentences:
        tree.insert('', i, values=sentence)
        i += 1
    bt = tk.Button(window, text='Complete', command=func3)
    bt.place(x=300, y=300)
    window.mainloop()


file = open('file.txt', 'a')
folder = os.path.exists('./records_tfidf/json/relevant')
if not folder:
    os.makedirs('./records_tfidf/json/relevant')
if not os.path.exists('./records_tfidf/json/non'):
    os.makedirs('./records_tfidf/json/non')
folder2 = os.path.exists('./records_tfidf/csv')
if not folder2:
    os.makedirs('./records_tfidf/csv')

# create window
window = tk.Tk()
window.title('Clinical Query Bot')
window.geometry("700x350")

# create entry box
l1 = tk.Label(window, text='Query:').place(x=10, y=15)
query = tk.StringVar()
entry_query = tk.Entry(window, width=50, textvariable=query)
entry_query.place(x=60, y=15)

l2 = tk.Label(window, text='Patient:').place(x=5, y=52)
patient = tk.StringVar()
entry_patient = tk.Entry(window, width=50, textvariable=patient)
entry_patient.place(x=60, y=50)

# create button
btn = tk.Button(window, text='Search', command=func)
btn.place(x=550, y=30)

window.mainloop()
eclapsed2 = time.clock() - start
print("Time used for retreival: ", eclapsed2)
