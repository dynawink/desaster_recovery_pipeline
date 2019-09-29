#!/usr/bin/python3

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# download nltk packages
nltk.download(['punkt', 'wordnet'])

"""This modules is part of the desaster recovery pipeline.
It provides the tokenizer.
"""

def tokenize(text):

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
