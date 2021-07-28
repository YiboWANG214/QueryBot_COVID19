# QueryBot_COVID19


1. Preprocess:  
   clean notes: notes_sentence.py  
   split notes into sentneces: Sentence_split.py  
   filter meaningful sentences: preprocess1.py  

3. Training:  
   bioBERT/clinical: python training.py  

3. Obtain sentence embeddings: python embeddings_xx.py  

4. Cluster: python split_embedding_xx.py  

5. Run QueryBot interface: python interface_xx.py 

   interface_BERT.py: interface for QueryBot using clinicalBERT  
   interface_bioBERT.py: interface for QueryBot using bioBERT  
   interface_tfidf.py: interface for QueryBot using tfidf  
   interface_word2vec.py: interface for QueryBot using word2vec  
