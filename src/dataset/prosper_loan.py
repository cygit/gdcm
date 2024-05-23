import os
import re
import string

import nltk
#from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction import _stop_words

nltk.download('punkt')
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from dataset.base_dataset import BaseDataset

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir)), "data", "prosper_loan")
os.makedirs(DATA_DIR, exist_ok=True)
MIN_DF = 0.01
MAX_DF = 0.8
TEST_RATIO = 0.15

TAG_RE = re.compile(r'<[^>]+>')


def remove_tags(text):
    return TAG_RE.sub('', text)


trans_table = {ord(c): None for c in string.punctuation + string.digits}
stemmer = nltk.SnowballStemmer("english")


def tokenize(text):
    tokens = [word for word in nltk.word_tokenize(text.translate(trans_table)) if len(word) >= 3 and
              word not in _stop_words]
    stems = [stemmer.stem(item) for item in tokens]
    return stems


x_df = pd.read_csv(os.path.join(DATA_DIR, "prosper_loan.csv"))
x_df['Description'] = x_df['Description'].fillna('')
documents = x_df['Description'].tolist()
labels = x_df['LoanStatus'].isin(['Paid', 'Defaulted (PaidInFull)', 'Defaulted (SettledInFull)']).astype(
    int).to_numpy()
x_df = x_df.drop(columns=['Key', 'LoanStatus', 'Description'])
# explanatory vars
expvars = x_df.to_numpy()
doc_train, doc_test, y_train, y_test, expvars_train, expvars_test = \
    train_test_split(documents, labels, expvars, test_size=TEST_RATIO)


class ProsperDataset(BaseDataset):
    def __init__(self):
        super().__init__(doc_train, doc_test, y_train, y_test, expvars_train, expvars_test)

    def get_data_filename(self, params):
        window_size = params["window_size"]  # context window size
        vocab_size = params["vocab_size"]  # max vocabulary size
        min_df = params.get("min_df", MIN_DF)  # min document frequency of vocabulary, defaults to MIN_DF
        max_df = params.get("max_df", MAX_DF)  # max document frequency of vocabulary, defaults to MAX_DF
        return os.path.join(DATA_DIR, "prosper_w%d_v%d_min%.0E_max%.0E.pkl" % (window_size, vocab_size,
                                                                               min_df, max_df))

    def load_data(self, params):
        print("Loading data...")
        window_size = params["window_size"]  # context window size
        vocab_size = params["vocab_size"]  # max vocabulary size
        min_df = params.get("min_df", MIN_DF)  # min document frequency of vocabulary, defaults to MIN_DF
        max_df = params.get("max_df", MAX_DF)  # max document frequency of vocabulary, defaults to MAX_DF

        vectorizer = CountVectorizer(tokenizer=tokenize, stop_words='english', min_df=min_df, max_df=max_df,
                                     max_features=vocab_size)
        return self.get_data_dict(self.get_data_filename(params), vectorizer, window_size)
