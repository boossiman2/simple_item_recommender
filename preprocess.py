import os
import pandas as pd
from tokenizers import Tokenizer, SentencePieceBPETokenizer, BertWordPieceTokenizer

class Reader():
    def __init__(self, data_path: str, dataset: str):
        self.data_path = data_path
        self.dataset = dataset

    def read_csv(self) -> pd.DataFrame:
        return self.read_csv(self.data_path+self.dataset)

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

class BertWordPieceTokenizer():
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

class Preprocessor(object):
    def __init__(self, data_path):
        self.data_path = data_path

    def countVecotrize(self):