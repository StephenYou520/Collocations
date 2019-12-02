import os
import nltk
import re
import numpy as np
from collections import Counter
import pandas  as pd

if __name__=='__main__':

    def readInChunks(fileObj, chunkSize=4096):
        """
        Lazy function to read a file piece by piece.
        Default chunk size: 4kB.
        """
        while 1:
            data = fileObj.read(chunkSize)
            if not data:
                break
            yield data


    text_= open("D:/nlp/corpora/sogout.txt", 'r', encoding="utf-8")

    #file_path='D:/nlp/corpora/sogout.txt'
    #file=open(os.path.join(file_path), "r", encoding='UTF8')
    #text=file.read()

    for chunk in readInChunks(text_):
        print(type(chunk))
        #text=re.sub('[.．：。；;！!?？——%\]\[\t\n\"\')(】【）（+\-\*/<>《》]', '', chunk)
        #text = re.sub('[a-zA-Z]', '', chunk)
        final_text=[x for x in chunk.split(" ") if len(x) > 3]
    #token=nltk.wordpunct_tokenize(final_text)
    #bigram=nltk.bigrams(token)
    #print (list(bigram)[0:])
    #print(text)
    #final_text=re.sub('[.．：。；;！!?？——%\]\[\t\n\"\')(】【）（+\-\*/<>《》]', '', final_text)
    #final_text = re.sub('[a-zA-Z]', '', final_text)
    c=Counter(final_text)
    n=len(c)
    print(c.keys())
    finder=nltk.collocations.BigramCollocationFinder.from_words(final_text)
    bigram_measures=nltk.collocations.BigramAssocMeasures()
    finder.nbest(bigram_measures.pmi,10)
    res=sorted(finder.ngram_fd.items(),key=lambda t:(-t[1],t[0]))
    print("res已经计算")
    d=np.zeros([n,2],int)
    matrix = pd.DataFrame(d, index=c.keys(), columns=[0,1])
    tp=[]

    for i in range(0,len(res)):
        matrix.loc[res[i][0][0],0]+=res[i][1]
        matrix.loc[res[i][0][1],1]+=res[i][1]
    print("矩阵matrix已生成")
    '''
    for i in range(0,len(res)):
        #t_test
        n_ii=res[i][1]
        n_ix=0
        n_xi=0
        n_xx=len(res)
        for j in range(0,len(res)):
            if res[j][0][0]==res[i][0][0]: n_ix+=1
            if res[j][0][1]==res[i][0][1]: n_xi+=1
    '''
    for i in range(0,len(res)):
        n_ii=res[i][1]
        n_ix=matrix.loc[res[i][0][0],0]
        n_xi=matrix.loc[res[i][0][0],1]
        n_xx=len(res)
        if i%100000==0:
            print("已计算：")
            print(i*10000)
        x_2=bigram_measures.chi_sq(n_ii,(n_ix,n_xi),n_xx)**2
        tp.append(x_2)

    # print(tp[0])
    ty = tp.copy()
    # ty=list(set(ty))
    ty = sorted(ty, reverse=True)
    final = []
    final.append(tp.index(ty[0]))
    j = 1
    while len(final) <= 10:
        k = 0
        j1 = j
        while tp.index(ty[j1]) == tp.index(ty[j1 - 1]):
            k = k + 1
            j1 = j1 + 1
        final.append(tp.index(ty[j]) + k)
        j = j + 1
    for f in range(10):
        print(res[final[f]][0],res[final[f]][1], tp[final[f]], "\n")