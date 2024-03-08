import os
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tokenizers import Tokenizer, SentencePieceBPETokenizer, BertWordPieceTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from utils import Similarity
from model.recommender import Candidate


# TODO : 추상화, 캡슐화
class Preprocessor:
    def __init__(self, data_path: str, file_name: str, model_path: str) -> None:
        self.data_path: str = data_path
        self.file_name: str = file_name
        self.model_path: str = model_path
        self.df: pd.DataFrame | None

    def _read_csv(self,  encoding: str = 'utf-8') -> pd.DataFrame:
        self.df = pd.read_csv(self.data_path + self.file_name, encoding=encoding)

    def drop_columns(self, df: pd.DataFrame, drop_cols: list) -> pd.DataFrame:
        return df.drop(columns=drop_cols, errors='ignore')

    def shuffle_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.sample(frac=1).reset_index(drop=True)

    def _write_dataframe(self, df: pd.DataFrame) -> None:
        df.to_csv(self.data_path + self.file_name.replace('.csv', '') + '_preprocessed.csv', index=False)

    def _write_vocabs(self, df: pd.DataFrame) -> None:
        with open(self.data_path+'vocabs.txt', 'w', encoding='utf-8') as f:
            for col in ['title', 'category1', 'category2']:
                for line in df[col].values:
                    try:
                        f.write(line + '\n')
                    except TypeError as e:
                        print(line, e)

    def preprocess_cluster(self) -> None:
        self._read_csv(encoding='latin1')
        # 중복 rows 제거
        self.df = self.df.drop_duplicates()
        # InvoiceDate의 data type을 Datetime 형식으로 변환
        self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'], format="%m/%d/%y %H:%M")

        # 1. Null 데이터 제거: 고객 식별 번호가 없는 데이터 삭제
        # 2. 오류 데이터 삭제: Quantity or UnitPrice <= 0 데이터 삭제
        # 3. CustomerID dtype을 int형으로 변환
        self.df = self.df[(self.df['CustomerID'].notnull()) & (self.df['Quantity'] > 0) & (self.df['UnitPrice'] > 0)]
        self.df['CustomerID'] = self.df['CustomerID'].astype(int)

        # RFM 기반 데이터 가공
        # RECENCY: 가장 최근 상품 구입 일에서 오늘까지의 기간
        # FREQUENCY: 상품 구매 횟수
        # MONETARY: 총 구매 금액 (UnitPrice * Quantity)
        self.df['OrderAmount'] = self.df['UnitPrice'] * self.df['Quantity']
        # 고객 레벨로 세그멘테이션을 수행해야하기 때문에 주문번호 기준의 데이터를 개별 고객 기준의 데이터로 groupby
        cust_df = self.df.groupby('CustomerID').agg({
            'InvoiceDate': 'max',
            'InvoiceNo': 'count',
            'OrderAmount': 'sum'
        }).reset_index()
        cust_df = cust_df.rename(columns={
            'InvoiceDate': 'Recency',
            'InvoiceNo': 'Frequency',
            'OrderAmount': 'Monetary'
        })
        # Recency 처리 (오늘 - 구매일).apply(lambda x: x.days+1)
        today = datetime.datetime.now()
        cust_df['Recency'] = (today - cust_df['Recency'])
        cust_df['Recency'] = cust_df['Recency'].apply(lambda x: x.days+1)

        # Recency는 평균이 92.7이지만, 50%인 51보다 크게 높음, max 는 374로 75%인 143보다 훨씬 커 왜곡 정도가 심함
        # Frequency의 경우 평균이 90.3이지만 max 값 7847를 포함한 상위의 몇 개의 큰 값으로 인해 75%가 99.25에 가까움
        # Monetary도 마찬가지로 상위의 큰 값으로 인해 평균은 1864.3으로 75%인 1576.5보다 높은 값이 확인 됨
        # -> 데이터 세트의 왜곡 정도를 낮추기 위해 로그변환 이용
        # --> Recency에 비해 Frequency, Monetary는 normal distribution에 유사하게 변함
        # --> Recency_log_scaled의 min=0, max=1에 비해 mean이 0.25로 너무 낮음 (sth 처리 필요)
        # -> StandardScaler, MinMaxScaler로 평균과 표준편차 재조정
        # --> StandardScaler와 다르게 MinMaxScaler로 재조정 시 Recency, Frequency, Monetary 모두 0~1 사이의 값을 가지게 됨
        cust_df[['Recency_log', 'Frequency_log', 'Monetary_log']] = \
            cust_df[['Recency', 'Frequency', 'Monetary']].apply(lambda x: np.log1p(x))
        cust_df[['Recency_log_scaled', 'Frequency_log_scaled', 'Monetary_log_scaled']] = MinMaxScaler().fit_transform(cust_df[['Recency_log', 'Frequency_log', 'Monetary_log']].values)
        # cust_df 저장
        self._write_dataframe(cust_df)

    def preprocess_recommender(self, top_k: int = 10) -> None:
        self._read_csv()
        df = self.df
        df = self.drop_columns(df, drop_cols=['image', 'link', 'category3', 'category4'])
        df = self.shuffle_dataframe(df)
        self._write_vocabs(df)

        tokenizer_type = 'BertWordPieceTokenizer'
        tokenizer = BertWPTokenizer(data_path=self.data_path, model_path=self.model_path)
        if not os.path.exists(self.model_path + tokenizer_type + '/tokenizer.json'):
            tokenizer.train(file_name='vocabs.txt')
            tokenizer.save_model()
        df['title_bertWP'] = df['title'].apply(lambda x: tokenizer.forward(x).tokens).apply(lambda x: " ".join(x))

        vectorizer_type = 'tf-idf'
        vectorizer = Vectorizer(vectorizer_type)
        # doc = title + ' ' + category1 + ' ' + category2
        doc_vec = vectorizer.vectorize(df)

        sim_matrix, sim_matrix_shape = Similarity().calculate_cos_sim(doc_vec)
        candidates = Candidate(topk=top_k).generate(sim_matrix)

        # candidates 저장
        pd.DataFrame(candidates,
                     index=range(1, len(candidates) + 1),
                     columns=range(1, top_k + 1)
                     ).to_csv(self.data_path + 'candidate.csv', sep='\t')


class SentencePieceTokenizer:
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

class BertWPTokenizer:
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

        print(model_path)
        self.__tokenizer.save(model_path + 'tokenizer.json')

    def write_df(self, df: pd.DataFrame) -> None:
        df.to_csv(self.data_path+'preprocessed.csv')


class Vectorizer:
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


if __name__ == '__main__':
    data_path = 'data/'
    file_name = 'Online_Retail.csv'
    preprocessor = Preprocessor(data_path=data_path, file_name=file_name, model_path='')
    preprocessor.preprocess_cluster()

    data_path = 'data/'
    file_name = 'naver_shopping.csv'
    model_path = 'model/models/'
    preprocessor = Preprocessor(data_path=data_path, file_name=file_name, model_path=model_path)
    preprocessor.preprocess_recommender(top_k=10)
