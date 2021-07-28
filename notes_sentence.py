#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from nltk.tokenize import sent_tokenize

data = pd.read_csv('./Covid_Unstructured_Notes(2).csv', encoding='utf-8',
                   engine='python')

uni_id = list(data['Patient ID'].unique())
pat_id = []
date = []
note = []
notetype = []

for i in uni_id:
    temp = data[data['Patient ID'] == i]
    flg = 0
    print('i:' + str(i))
    for j in range(temp.shape[0]):
        if flg == 0:
            pat_id.append(i)
            date.append(temp.iloc[j]['Result Verification Day'])
            notetype.append(temp.iloc[j]['Document Name'])
            note.append(str(temp.iloc[j]['Note Result']))
            flg = 1
        else:
            if (temp.iloc[j]['Result Verification Day'] == temp.iloc[j - 1]['Result Verification Day']) and (
                    temp.iloc[j]['Document Name'] == temp.iloc[j - 1]['Document Name']):
                note[-1] = note[-1] + ' ' + str(temp.iloc[j]['Note Result'])
            else:
                pat_id.append(i)
                date.append(temp.iloc[j]['Result Verification Day'])
                notetype.append(temp.iloc[j]['Document Name'])
                note.append(str(temp.iloc[j]['Note Result']))
        if i % 1000 == 0:
            notes_combine = pd.DataFrame(
                {'Patient_ID': pat_id, 'Note_date': date, 'Note_type': notetype, 'Notes': note})
            notes_combine.to_csv('./clean_notes.csv')

notes_combine = pd.DataFrame({'Patient_ID': pat_id, 'Note_date': date, 'Note_type': notetype, 'Notes': note})
notes_combine.to_csv('./clean_notes.csv', index=False)
