# preprocess data
import pandas as pd

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
words = stopwords.words('english')
words.append('patient')

data = pd.read_excel('clean_notes_sentences.xlsx', encoding='utf-8')
print(data.columns)
notes = data['Sentences']
id = data['Patient_ID']
note_id = data['Note_id']
date = data['Note_date']
type_ = data['Note_type']
sentences = data['Sentences']
print(notes[0])
print(len(data))
patient_ = []
note_id_ = []
date_ = []
type__ = []
notes_line = []
sentences_ = []

num = 0
i = 0
length = len(data)
print(length)
while i < length:
    # print(i)
    if not isinstance(notes[i], float) and not isinstance(notes[i], int):
        note = notes[i]
        print(i)
        print(note)
        if 4 <= len(note.split(' ')) <= 200:
            # print(i)
            note = note.lower()
            note = note.split(' ')
            filtered_words = [word for word in note if word not in stopwords.words('english')]
            note = ' '.join(filtered_words)
            notes_line.append(note)
            patient_.append(id[i])
            note_id_.append(note_id[i])
            date_.append(date[i])
            type__.append(type_[i])
            sentences_.append(sentences[i])

    i += 1

new_data = pd.DataFrame({'Patient_ID':patient_, 'Note_ID':note_id_, 'Note_date':date_, 'Note_type':type__, 'Sentences':sentences_, 'clean': notes_line})

new_data.to_csv('sentences_clean_noteID_3.csv', index=False)
