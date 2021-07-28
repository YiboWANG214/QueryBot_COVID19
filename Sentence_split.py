import pandas as pd
from nltk.tokenize import sent_tokenize


notes = pd.read_excel('./clean_notes.xlsx')

pat_id = []
notes_id = []
date = []
sentences = []
notetype = []

for i in range(notes.shape[0]):
    test = str(notes.iloc[i]['Notes'])
    content1 = ''
    content2 = ''
    test = test.replace('\r','\n')
    test = test.split('\n')
    for line in test:
        if line != '':
            content1 += (line+'. ')
            content2 += line
    content2 = sent_tokenize(content2)
    print(i)
    for sent in content2:
        pat_id.append(notes.iloc[i]['Patient_ID'])
        notes_id.append(i)
        date.append(notes.iloc[i]['Note_date'])
        sentences.append(sent)
        notetype.append(notes.iloc[i]['Note_type'])
        
notes_combine = pd.DataFrame({'Patient_ID':pat_id, 'Note_id':notes_id, 'Note_date':date, 'Note_type':notetype, 'Sentences':sentences})
notes_combine.to_excel('clean_notes_sentences.xlsx')
