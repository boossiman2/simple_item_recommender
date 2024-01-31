import os
import pandas as pd
import numpy as np
from tokenizers import Tokenizer, SentencePieceBPETokenizer, BertWordPieceTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class Reader():
    def __init__(self, data_path: str, dataset: str):
        self.data_path = data_path
        self.dataset = dataset

    def read_csv(self) -> pd.DataFrame:
        return pd.read_csv(self.data_path+self.dataset)

    def write_csv(self, df: pd.DataFrame, file_name: str) -> None:
        df.to_csv(self.data_path+file_name)

    def drop_columns(self, df: pd.DataFrame, drop_cols: list) -> pd.DataFrame:
        return df.drop(columns=drop_cols, errors='ignore')

    def shuffle_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.sample(frac=1).reset_index(drop=True)

    def write_vocabs(self, df: pd.DataFrame):
        with open(self.data_path+'vocabs.txt', 'w', encoding='utf-8') as f:
            for col in ['title', 'category1', 'category2']:
                for line in df[col].values:
                    try:
                        f.write(line + '\n')
                    except TypeError as e:
                        print(line, e)

class SentencePieceTokenizer():
    def __init__(self, data_path: str, model_path: str):
        self.data_path = data_path
        self.model_path = model_path
        self.__tokenizer = self.__load_model();

    def __load_model(self):
        try:
            return Tokenizer.from_file(self.model_path + 'SentencePieceBPETokenizer/' + 'tokenizer.json')
        except:
            return SentencePieceBPETokenizer()

    def train(self, file_name: str, vocab_size: int=25000) -> None:
        self.__tokenizer.train(self.data_path+file_name, vocab_size=vocab_size)

    def forward(self, data: str) -> list:
        return self.__tokenizer.encode(data)

    def save_model(self) -> None:
        model_path = self.model_path + 'SentencePieceBPETokenizer/'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.__tokenizer.save(model_path + 'tokenizer.json')

class BertWPTokenizer():
    def __init__(self, data_path: str, model_path: str):
        self.data_path = data_path
        self.model_path = model_path
        self.__tokenizer = self.__load_model();

    def __load_model(self):
        try:
            return Tokenizer.from_file(self.model_path + 'bertWordPieceTokenizer/' + 'tokenizer.json')
        except:
            # lowercase : 대소문자를 구분 여부. True일 경우 구분하지 않음.
            # strip_accents : True일 경우 악센트 제거. ex) é → e, ô → o
            return BertWordPieceTokenizer(lowercase=False, strip_accents=False)

    def train(self, file_name: str, vocab_size: int=25000) -> list:
        self.__tokenizer.train(self.data_path+file_name, vocab_size=vocab_size)

    def forward(self, data: str) -> list:
        return self.__tokenizer.encode(data)

    def save_model(self) -> None:
        model_path = self.model_path + 'BertWordPieceTokenizer/'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.__tokenizer.save(model_path + 'tokenizer.json')

    def write_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df['title_bertWP'] = df['title'].apply(lambda x : self.forward(x).tokens).apply(lambda x : " ".join(x))
        df.to_csv(self.data_path+'preprocessed.csv')
        return df

class Vectorizer():
    def __init__(self, vectorizer_type: str):
        if vectorizer_type == 'count':
            self.vectorizer = CountVectorizer(ngram_range=(1, 3))
        elif vectorizer_type == 'tf-idf':
            self.vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1,3))

    def vectorize(self, df: pd.DataFrame) -> np.ndarray:
        document_vec = self.vectorizer.fit_transform(
            df['title_bertWP'] + ' ' + df['category1'] + ' ' + df['category2']
        ).toarray()
        return document_vec
