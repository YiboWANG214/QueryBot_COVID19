# QueryBot_COVID19
This is the codes for paper "Query bot for retrieving patients’ clinical history: A COVID-19 use-case" https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8454191/


1. Preprocess:  
   clean notes: notes_sentence.py  
   split notes into sentneces: Sentence_split.py  
   filter meaningful sentences: preprocess1.py  

3. Training:  
   bioBERT: python training_bioBERT.py
   clinicalBERT: python training_clinicalBERT.py

3. Obtain sentence embeddings: python embeddings_xx.py  

4. Cluster: python split_embedding_xx.py  

5. Run QueryBot interface: python interface_xx.py 

   interface_BERT.py: interface for QueryBot using clinicalBERT  
   interface_bioBERT.py: interface for QueryBot using bioBERT  
   interface_tfidf.py: interface for QueryBot using tfidf  
   interface_word2vec.py: interface for QueryBot using word2vec  
