from utils import *
import os

path = 'C:/Users/DELL/Desktop/freela/dataset/theses100/'
all_files = os.listdir(path + 'docsutf8')
all_keys = os.listdir(path + 'keys')
print(
    len(all_files), ' files n', all_files, 'n', all_keys
)  # won't necessarily be sorted

all_documents = []
all_keys = []
all_files_names = []
for i, fname in enumerate(all_files):
    with open(path + 'docsutf8/' + fname) as f:
        lines = f.readlines()
    key_name = fname[:-4]
    with open(path + 'keys/' + key_name + '.key') as f:
        k = f.readlines()
    all_text = ' '.join(lines)
    keyss = ' '.join(k)
    all_documents.append(all_text)
    all_keys.append(keyss.split('n'))
    all_files_names.append(key_name)

# ------------------------------------------------------------------------------

import pandas as pd

dtf = pd.DataFrame({'goldkeys': all_keys, 'text': all_documents})
print(dtf.head())

import re
import string

import nltk
import nltk.data
import numpy as np
import regex as re
import spacy
from nltk import pos_tag, sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# nltk.download()
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')
# nltk.download('wordnet')


def convert_to_lower(text):
    return text.lower()


def remove_numbers(text):
    text = re.sub(r'd+', '', text)
    return text


def remove_http(text):
    text = re.sub('https?://t.co/[A-Za-z0-9]*', ' ', text)
    return text


def remove_short_words(text):
    text = re.sub(r'bw{1,2}b', '', text)
    return text


def remove_short_words(text):
    text = re.sub(r'bw{1,2}b', '', text)
    return text


def remove_punctuation(text):
    punctuations = """!()[]{};«№»:'",`./?@=#$-(%^)+&[*_]~"""
    no_punctuation = ''

    for char in text:
        if char not in punctuations:
            no_punctuation = no_punctuation + char
    return no_punctuation


def remove_white_space(text):
    text = text.strip()
    return text


def toknizing(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)

    ## Remove Stopwords from tokens

    result = [i for i in tokens if not i in stop_words]

    return result


def preprocess_text(text):

    # 1. Tokenise to alphabetic tokens
    text = remove_numbers(text)
    text = remove_http(text)
    text = remove_punctuation(text)
    text = convert_to_lower(text)
    text = remove_white_space(text)
    text = remove_short_words(text)

    # 2. POS tagging
    pos_map = {'J': 'a', 'N': 'n', 'R': 'r', 'V': 'v'}
    tokens = toknizing(text)
    pos_tags_list = pos_tag(tokens)
    # print(pos_tags)

    # 3. Lowercase and lemmatise
    lemmatiser = WordNetLemmatizer()
    tokens = [
        lemmatiser.lemmatize(w.lower(), pos=pos_map.get(p[0], 'v'))
        for w, p in pos_tags_list
    ]
    return tokens


# clean text applying all the text preprocessing functions
dtf['cleaned_text'] = dtf.text.apply(lambda x: ' '.join(preprocess_text(x)))
print(dtf.head())


# Clean the basic keywords and remove the spaces and noise
def clean_orginal_kw(orginal_kw):
    orginal_kw_clean = []
    for doc_kw in orginal_kw:
        temp = []
        for t in doc_kw:
            tt = ' '.join(preprocess_text(t))
            if len(tt.split()) > 0:
                temp.append(tt)
        orginal_kw_clean.append(temp)
    return orginal_kw_clean


orginal_kw = clean_orginal_kw(dtf['goldkeys'])

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    use_idf=True, max_df=0.5, min_df=1, ngram_range=(1, 3)
)
vectors = vectorizer.fit_transform(dtf['cleaned_text'])

dict_of_tokens = {i[1]: i[0] for i in vectorizer.vocabulary_.items()}

tfidf_vectors = []  # all deoc vectors by tfidf
for row in vectors:
    tfidf_vectors.append(
        {
            dict_of_tokens[column]: value
            for (column, value) in zip(row.indices, row.data)
        }
    )

doc_sorted_tfidfs = []  # list of doc features each with tfidf weight
# sort each dict of a document
for dn in tfidf_vectors:
    newD = sorted(dn.items(), key=lambda x: x[1], reverse=True)
    newD = dict(newD)
    doc_sorted_tfidfs.append(newD)

tfidf_kw = []   # get the keyphrases as a list of names without tfidf values
for doc_tfidf in doc_sorted_tfidfs:
    ll = list(doc_tfidf.keys())
    tfidf_kw.append(ll)


def apk(kw_actual, kw_predicted, top_k=10):
    if len(kw_predicted) > top_k:
        kw_predicted = kw_predicted[:top_k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(kw_predicted):
        if p in kw_actual and p not in kw_predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not kw_actual:
        return 0.0
    return score / min(len(kw_actual), top_k)


def mapk(kw_actual, kw_predicted, top_k=10):
    return np.mean([apk(a, p, top_k) for a, p in zip(kw_actual, kw_predicted)])


for k in [5, 10, 20, 40]:
    mpak = mapk(orginal_kw, tfidf_kw, k)
    print('mean average precession  @', k, '=  {0:.4g}'.format(mpak))

TopN = 5
print(tfidf_kw[0][0:TopN])