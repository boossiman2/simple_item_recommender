import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Preprocessor:
    def __init__(self, data_path: str, file_name: str) -> None:
        self.data_path: str = data_path
        self.file_name: str = file_name
        self.df: pd.DataFrame = self._read_csv()

    def _read_csv(self) -> pd.DataFrame:
        return pd.read_csv(self.data_path + self.file_name, encoding='latin1')

    def _write_dataframe(self, df: pd.DataFrame) -> None:
        df.to_csv(self.data_path + self.file_name.replace('.csv', '') + '_preprocessed.csv', index=False)

    def preprocess_dataframe(self) -> None:
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


if __name__ == '__main__':
    data_path = '../data/'
    file_name = 'Online_Retail.csv'
    preprocessor = Preprocessor(data_path=data_path, file_name=file_name)
    preprocessor.preprocess_dataframe()