import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import csv
from sklearn.cluster import KMeans
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time

def optimalK(data, nrefs=3, maxClusters=100):
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})
    for gap_index, k in enumerate(range(1, maxClusters)):
        refDisps = np.zeros(nrefs)
        for i in range(nrefs):
            randomReference = np.random.random_sample(size=data.shape)
            km = KMeans(k)
            km.fit(randomReference)

            refDisp = km.inertia_
            refDisps[i] = refDisp
        km = KMeans(k)
        km.fit(data)

        origDisp = km.inertia_
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)
        gaps[gap_index] = gap

        resultsdf = resultsdf.append({'clusterCount': k, 'gap': gap}, ignore_index=True)

    return (gaps.argmax() + 1,
            resultsdf)


my_df = pd.read_csv('./embeddings_bio.csv')
embeddings = []
for j in range(len(my_df)):
    tmp = my_df.loc[j]['embeddings'][1:-1].replace('\n', ' ').split(', ')
    # tmp = my_df.loc[j]['embeddings'][1:-1].split() 
    sentence = [float(x) for x in tmp]
    #sentence = list(map(float, tmp))
    sentence = np.array(sentence).reshape(-1, 768)
    embeddings.append(sentence[0])
X = np.array(embeddings)
print(X.shape)

'''
pca = PCA(n_components=2)
X = pca.fit_transform(X)
'''
#X = TSNE(n_components=2, n_iter=250).fit_transform(X)
#np.savetxt('tsne_split.csv', X)

bestKValue, gapdf = optimalK(X, nrefs=5, maxClusters=100)
#print(bestKValue)

km = KMeans(bestKValue)
result = km.fit_predict(X)
clusters = km.labels_.tolist()
my_df['label'] = clusters

folder = os.path.exists("./data_bio")
if not folder:
    os.makedirs("./data_bio")

my_df.to_csv("my_bio.csv", index=False)

label = []
embeddings_ = []
for ii in range(bestKValue):
    b = []
    a = my_df[my_df.label == ii]
    a.to_csv('./data_bio/' + 'data' + str(ii) + '.csv')
    tmp = my_df[my_df.label == ii].embeddings.tolist()
    for t in tmp:
        t0 = t[1:-1].replace('\n', ' ')
        t0 = list(map(float, t0.split(', ')))
        b.append(t0)
    b = np.mean(b, axis=0)
    label.append(ii)
    embeddings_.append(b)

mean_ = pd.DataFrame({'label': label, 'embedding': embeddings_})
mean_.to_csv('./data_bio/' + 'mean' + '.csv')
