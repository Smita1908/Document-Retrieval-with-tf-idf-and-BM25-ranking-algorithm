# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup as bs
import pandas as pd
import re
import numpy as np
import math
from nltk.corpus import stopwords
from collections import Counter
from collections import OrderedDict

pd.set_option('display.expand_frame_repr', False)


# preprocessing the data by lower-casing,removing punctuation and forming tokens
def preprocess(data):
    contents = ' '.join(map(lambda l: ''.join(l), data))
    tokens = re.findall(r'\w+', contents.strip().lower())
    tokens = ["'" + item + "'" for item in tokens]
    return tokens


# Calculating IDF values
def Idf(doc, N):
    idfDict = {}
    for token in doc.split():
            idfDict[token] = idfDict.get(token, 0) + 1
    for term in idfDict.keys():
        idfDict[term] = math.log(N/idfDict[term])
    return idfDict



# Calculating TF values of the terms per document
def Doctf(doc):
    counts, tf = {}, {}
    for item in doc.split():
        if item in counts:
            counts[item] += 1
        else:
            counts[item] = 1
    max_freq = counts.get(max(counts, key=counts.get))
    for term, frequency in counts.items():
        if max_freq != 0:
            tf[term] = frequency/max_freq
    return tf

# Calculating tf-idf value of terms of one document
def DocTfIdf(doc, idf, tf):
    tf_idf = {}
    for term in doc.split():
        tf_idf[term] = tf.get(term, 0)*idf.get(term, 0)
    return tf_idf

#reading the xml file using beautifulsoup and saving it as a dataframe
content=[]
with open("trec_documents.xml") as file:
    content = file.read().splitlines()
    content = "".join(content)
    bs_content = bs(content, "lxml")
    doccount = len([str(t.text) for t in (bs_content.findAll('docno'))])
    pairs = {}
    for doc in bs_content.select('doc'):
        key = doc.docno.text
        texts = [p.text for p in doc.select('p') or doc.select('text')]
        pairs[key] = texts
    dataframe = pd.DataFrame(pairs.items(), columns=['Doc number', 'Text'])
    dataframe['Tokens'] = dataframe['Text'].apply(preprocess)
    #print(dataframe)
    corpus = [' '.join(ele) for ele in dataframe['Tokens']]
    corpus = [x.replace("'", "") for x in corpus]
    N = len(dataframe.index)
    for doc in corpus:
        idf_dict = Idf(doc, N)
        print(idf_dict)
        tf_dict = Doctf(doc)
        print(tf_dict)
        tfidf_dict = DocTfIdf(doc, idf_dict, tf_dict)
        print(tfidf_dict)


# Finding the Cosine Similarity score between two vectors
def cosineSimilarity(vec1, vec2):
    numerator = np.dot(vec1, vec2)
    denominator = np.linalg.norm(vec1)*np.linalg.norm(vec2)
    if denominator != 0:
        co_sim = numerator/denominator
    return co_sim

#reading the qustions text file and asving it as a dataframe
questions=[]
with open("test_questions.txt") as file:
    questions = file.read().splitlines()
    questions = "".join(questions)
    bs_questions = bs(questions, "lxml")
    ques_nums = {}
    for ques in bs_questions.select('top'):
        number = ques.find('num').text.split('Description')[0]
        question = ques.desc.text.split('Description')[1].replace(":", "")
        ques_nums[number] = question
    dataframe = pd.DataFrame(ques_nums.items(), columns=['Question number', 'Question'])

