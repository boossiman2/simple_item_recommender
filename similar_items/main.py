import os

import pandas as pd

from preprocess import Reader, BertWPTokenizer, Vectorizer
from customer_clustering.app.model.recommender import Similarity, Candidate

DATA_PATH = './data/'
MODEL_PATH = './model/'
DATA = 'naver_shopping.csv'
TOP_K = 10

def main():
    csv_reader = Reader(data_path=DATA_PATH, dataset=DATA)
    df = csv_reader.read_csv()
    df = csv_reader.drop_columns(df, drop_cols=['image', 'link', 'category3', 'category4'])
    df = csv_reader.shuffle_dataframe(df)
    csv_reader.write_vocabs(df) # './data/vocabs.txt'

    tokenizer_type = 'BertWordPieceTokenizer' # TODO : argparser 화
    tokenizer = BertWPTokenizer(data_path=DATA_PATH, model_path=MODEL_PATH)
    if not os.path.exists(MODEL_PATH + tokenizer_type + '/tokenizer.json'):
        tokenizer.train('vocabs.txt')
        tokenizer.save_model()
    df = tokenizer.write_df(df)

    vectorizer_type = 'tf-idf' # TODO : argparser 화
    vectorizer = Vectorizer(vectorizer_type)
    doc_vec = vectorizer.vectorize(df)

    sim_matrix, sim_matrix_shape = Similarity().calculate_cos_sim(doc_vec)
    candidates = Candidate(topk=TOP_K).generate(sim_matrix) # N(아이템 수) X K(topk) 차원의 테이블

    # candidates 저장
    pd.DataFrame(candidates,
                 index=range(1, len(candidates)+1),
                 columns=range(1, TOP_K+1)
                 ).to_csv(DATA_PATH+'candidate.csv', sep='\t')

if __name__ == '__main__':
    main()


