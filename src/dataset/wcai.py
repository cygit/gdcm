import os

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from dataset.base_dataset import BaseDataset

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir)), "data", "wcai")
os.makedirs(DATA_DIR, exist_ok=True)
MIN_DF = 0.01
MAX_DF = 0.8
TEST_RATIO = 0.15
df = pd.read_csv(os.path.join(DATA_DIR, "wcai.csv"), encoding="utf-8")
labels = df["labels"].values
concatenated_documents = df["docs"].values
# explanatory vars
del df["labels"]
del df["docs"]
expvars = df.values

doc_train, doc_test, y_train, y_test, expvars_train, expvars_test = \
    train_test_split(concatenated_documents, labels, expvars, test_size=TEST_RATIO)

class WcaiDataset(BaseDataset):
    def __init__(self):
        super().__init__(doc_train, doc_test, y_train, y_test, expvars_train, expvars_test)

    def get_data_filename(self, params):
        window_size = params["window_size"]  # context window size
        min_df = params.get("min_df", MIN_DF)  # min document frequency of vocabulary, defaults to MIN_DF
        max_df = params.get("max_df", MAX_DF)  # max document frequency of vocabulary, defaults to MAX_DF
        return os.path.join(DATA_DIR, "wcai_w%d_min%.0E_max%.0E.pkl" % (window_size, min_df, max_df))

    def load_data(self, params):
        window_size = params["window_size"]  # context window size
        min_df = params.get("min_df", MIN_DF)  # min document frequency of vocabulary, defaults to MIN_DF
        max_df = params.get("max_df", MAX_DF)  # max document frequency of vocabulary, defaults to MAX_DF
        vectorizer = CountVectorizer(min_df=min_df, max_df=max_df)
        return self.get_data_dict(self.get_data_filename(params), vectorizer, window_size)
