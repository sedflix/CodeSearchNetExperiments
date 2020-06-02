import tensorflow as tf
from typing import List

from data_reader import get_data_df


def get_generator(root_path: str, langs: List[str], splits: List[str],
                  max_len_query: int, max_len_code: int,
                  query_ft, code_ft):
    def temp():
        df = get_data_df(root_path, langs, splits)

        for i, row in df[['docstring_tokens', 'code_tokens']].iterrows():
            query = [query_ft[token] for token in row[0]]
            code = [code_ft[token] for token in row[1]]
            query = tf.keras.preprocessing.sequence.pad_sequences(
                [query], maxlen=max_len_query, dtype='float', padding='post', truncating='post',
                value=[0]
            )
            code = tf.keras.preprocessing.sequence.pad_sequences(
                [code], maxlen=max_len_code, dtype='float', padding='post', truncating='post',
                value=[0]
            )

            yield query[0], code[0]

    return temp


def get_dataset_at_once(root_path: str, langs: List[str], splits: List[str], split, max_len_query, max_len_code,
                        query_ft, code_ft):
    df = get_data_df(root_path, langs, splits)

    q_s, c_s = [], []
    for i, row in df[['docstring_tokens', 'code_tokens']].iterrows():
        _query = [query_ft[token] for token in row[0]]
        _code = [code_ft[token] for token in row[1]]

        q_s.append(_query)
        c_s.append(_code)

    q_s = tf.keras.preprocessing.sequence.pad_sequences(
        q_s, maxlen=max_len_query, dtype='float', padding='post', truncating='post',
        value=[0]
    )

    c_s = tf.keras.preprocessing.sequence.pad_sequences(
        c_s, maxlen=max_len_code, dtype='float', padding='post', truncating='post',
        value=[0]
    )

    query_dataset = tf.data.Dataset.from_tensor_slices(q_s)
    code_dataset = tf.data.Dataset.from_tensor_slices(c_s)

    dataset = tf.data.Dataset.zip((query_dataset, code_dataset))

    return dataset


def df_to_dataset(root_path: str, langs: List[str], splits: List[str]):
    df = get_data_df(root_path, langs, splits)

    querys = list(df['docstring_tokens'])
    codes = list(df['code_tokens'])

    querys_dataset = tf.data.Dataset.from_tensor_slices(querys)
    codes_dataset = tf.data.Dataset.from_tensor_slices(codes)
