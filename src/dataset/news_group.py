import os

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

from dataset.base_dataset import BaseDataset

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir)), "data", "news_group")
os.makedirs(DATA_DIR, exist_ok=True)
MIN_DF = 0.01
MAX_DF = 0.8
newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True, remove=("headers", "footers", "quotes"),
                                      categories=["alt.atheism", "comp.graphics"])
newsgroups_test = fetch_20newsgroups(subset='test', shuffle=True, remove=("headers", "footers", "quotes"),
                                     categories=["alt.atheism", "comp.graphics"])

class NewsDataset(BaseDataset):
    def __init__(self):
        super().__init__(newsgroups_train.data, newsgroups_test.data, newsgroups_train.target, newsgroups_test.target)

    def get_data_filename(self, params):
        window_size = params["window_size"]  # context window size
        min_df = params.get("min_df", MIN_DF)  # min document frequency of vocabulary, defaults to MIN_DF
        max_df = params.get("max_df", MAX_DF)  # max document frequency of vocabulary, defaults to MAX_DF
        return os.path.join(DATA_DIR, "20_news_group_w%d_min%.0E_max%.0E.pkl" % (window_size, min_df, max_df))

    def load_data(self, params):
        window_size = params["window_size"]  # context window size
        min_df = params.get("min_df", MIN_DF)  # min document frequency of vocabulary, defaults to MIN_DF
        max_df = params.get("max_df", MAX_DF)  # max document frequency of vocabulary, defaults to MAX_DF
        vectorizer = CountVectorizer(min_df=min_df, max_df=max_df)
        return self.get_data_dict(self.get_data_filename(params), vectorizer, window_size)
