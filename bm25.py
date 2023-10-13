# -*- coding: utf-8 -*-

from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
import numpy as np
from collections import Counter
import pandas as pd
#from collections import OrderedDict
import math


# get text and tokenization
def getTextToken(bs, type):
    texts = []
    stopWords = stopwords.words('english')
    for txt in bs.find_all(type):
        words = re.findall(r'\w+', txt.text.strip().lower())
        no_stops = [w for w in words if (w not in stopWords) and (w != 'description')]
        texts.append(no_stops)
    return texts


#Myclass is implemented for calcualting BM25 scores
class MyClass:
    def __init__(self, k1=1.5, b=0.75):
            self.b = b
            self.k1 = k1

    # Fitting the parameters to calculate the BM25 Ranking. The corpus is a list of lists
    def _fit(self, corpus):
        corpus = getTextToken(soup_content, 'text')
        tf = []
        df = {}
        idf = {}
        doc_len = []
        corpus_size = 0
        for document in corpus:
            corpus_size += 1
            doc_len.append(len(document))
        
            # Per document Term Frequencies computed below
            frequencies = {}
            for term in document:
                term_count = frequencies.get(term, 0) + 1
                frequencies[term] = term_count
        
            tf.append(frequencies)
        
            # Document Frequencies per term computed below
            for term, _ in frequencies.items():
                df_count = df.get(term, 0) + 1
                df[term] = df_count
        
        for term, freq in df.items():
            idf[term] = math.log(1 + (corpus_size - freq + 0.5) / (freq + 0.5))
        
        self.tf_ = tf
        self.df_ = df
        self.idf_ = idf
        self.doc_len_ = doc_len
        self.corpus_ = corpus
        self.corpus_size_ = corpus_size
        self.avg_doc_len_ = sum(doc_len) / corpus_size
        return self
    
    def look(self, query):
        scores = [self.Score(query, index) for index in range(self.corpus_size_)]
        return scores
    
    def Score(self, query, index):
        score = 0.0
        
        doc_len = self.doc_len_[index]
        frequencies = self.tf_[index]
        for term in query:
            if term not in frequencies:
                continue

            freq = frequencies[term]
            numerator = self.idf_[term] * freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len_)
            score += (numerator / denominator)
    
        return score


with open("trec_documents.xml", "r") as file:
    # Read each line in the file, readlines() returns a list of lines
    content = file.readlines()
    # Combine the lines in the list into a string
    content = "".join(content)
    soup_content = BeautifulSoup(content, "lxml")
text_tokens = getTextToken(soup_content, 'text')


bm25 = MyClass()

# A word count dictionary to extract unique words and counts
word_count_dict = {}
for text in text_tokens:
    for token in text:
        word_count = word_count_dict.get(token, 0) + 1
        word_count_dict[token] = word_count

text_tokens = [[token for token in text if word_count_dict[token] > 1] for text in text_tokens]
bm25._fit(text_tokens)



# Preprocessing the dataset to remove punctuations and form tokens+
def preprocess(data):
    contents = ' '.join(map(lambda l: ''.join(l), data))
    tokens = re.findall(r'\w+', contents.strip().lower())
    tokens = ["'" + item + "'" for item in tokens]
    return tokens



# Reading the datafile and converting it into a dataframe
def readFile(filename):
    with open(filename) as file:
        content = file.read().splitlines()
        content = "".join(content)
        bs_content = BeautifulSoup(content, "lxml")
        doccount = len([str(t.text) for t in (bs_content.findAll('docno'))])
        pairs = {}
        for doc in bs_content.select('doc'):
            key = doc.docno.text
            texts = [p.text for p in doc.select('p') or doc.select('text')]
            pairs[key] = texts
        dataframe = pd.DataFrame(pairs.items(), columns=['Doc number', 'Text'])
        dataframe['Tokens'] = dataframe['Text'].apply(preprocess)
        corpus = [' '.join(ele) for ele in dataframe['Tokens']]
        corpus = [x.replace("'", "") for x in corpus]
        doc_id = dataframe['Doc number'].tolist()
        return corpus, doc_id

corpus, doc_id = readFile('trec_documents.xml')



docs = BeautifulSoup(open('test_questions.txt'), "lxml")
docs_desc = getTextToken(docs, 'desc')
scores, mean_scores = [], []
for i in range(len(docs_desc)):
    score = bm25.look(docs_desc[i])
    scores.append(score)
    # scores += scores
for term in scores:
    mean_scores.append(sum(term)/len(term))

zipped = zip(doc_id, mean_scores)
for doc, score in zipped:
    # score = round(score, 3)
    print('Doc ID : ---->> ' + str(doc) + ' :::  BM25 Mean ---->> ' + str(score))





