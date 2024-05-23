import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from dataset.base_dataset import BaseDataset


MIN_DF = 0.01
MAX_DF = 0.1
TEST_RATIO = 0.15
DATASET_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data", "amazon_review")


os.makedirs(DATASET_DIR, exist_ok=True)


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PAR_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir))
DATA_DIR = os.path.join(PAR_DIR, "data")

if not os.path.exists(os.path.join(DATA_DIR, "amazon_review", "amazon_review.csv")):
    nltk.download('punkt')
    nltk.download('stopwords')


    reviews_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Subscription_Boxes", trust_remote_code=True)
    reviews_df = pd.DataFrame(reviews_dataset['full'])

    metadata_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_Subscription_Boxes", split="full", trust_remote_code=True)
    metadata_df = pd.DataFrame(metadata_dataset)

    df = pd.merge(reviews_df, metadata_df, on='parent_asin', how='inner')
    df = df.drop(columns=["images_y", "videos", "subtitle", "author", "bought_together"])
    df = df.drop(columns=["images_x", "user_id", "timestamp", 'asin'])
    df['verified_purchase'] = df['verified_purchase'].astype(int)

    cols_to_combine = ['title_x', 'text', 'main_category', 'title_y', 'features', 'description']
    df['combined_text'] = df[cols_to_combine].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
    df['tokenized_text'] = df['combined_text'].apply(word_tokenize)

    stop_words = set(stopwords.words('english'))
    df['filtered_text'] = df['tokenized_text'].apply(lambda tokens: [word for word in tokens if word.lower() not in stop_words])
    df['filtered_text'] = df['filtered_text'].apply(lambda tokens: ' '.join(tokens))
    df['filtered_text'] = df['filtered_text'].apply(lambda text: re.sub(r'[^A-Za-z0-9\s]', '', text))

    df = df.drop(columns = ['price', 'title_x', 'text', 'parent_asin', 'main_category', 'title_y', 'features', 'description', 'store', 'categories', 'details', 'combined_text', 'tokenized_text'])
    df = df.dropna()
    # CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    # PAR_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir))
    # DATA_DIR = os.path.join(PAR_DIR, "data")


    # csv_filename = os.path.join(DATA_DIR, "amazon_review.csv")
    csv_filename = os.path.join(DATA_DIR, "amazon_review", 'amazon_review.csv')
    df.to_csv(csv_filename, index=False)


df = pd.read_csv(os.path.join(DATA_DIR, "amazon_review", "amazon_review.csv"), encoding="utf-8")
labels = df["rating"].values
concatenated_documents = df["filtered_text"].values
del df["rating"]
del df["filtered_text"]

expvars = df.values

doc_train, doc_test, y_train, y_test, expvars_train, expvars_test = \
    train_test_split(concatenated_documents, labels, expvars, test_size=TEST_RATIO)

class AmazonReviewDataset(BaseDataset):
    def __init__(self):
        super().__init__(doc_train, doc_test, y_train, y_test, expvars_train, expvars_test)

    def get_data_filename(self, params):
        window_size = params["window_size"]  # context window size
        min_df = params.get("min_df", MIN_DF)  # min document frequency of vocabulary, defaults to MIN_DF
        max_df = params.get("max_df", MAX_DF)  # max document frequency of vocabulary, defaults to MAX_DF
        return os.path.join(DATA_DIR, "amazon_review_w%d_min%.0E_max%.0E.pkl" % (window_size, min_df, max_df))

    def load_data(self, params):
        window_size = params["window_size"]  # context window size
        min_df = params.get("min_df", MIN_DF)  # min document frequency of vocabulary, defaults to MIN_DF
        max_df = params.get("max_df", MAX_DF)  # max document frequency of vocabulary, defaults to MAX_DF
        vectorizer = CountVectorizer(min_df=min_df, max_df=max_df)
        return self.get_data_dict(self.get_data_filename(params), vectorizer, window_size)

