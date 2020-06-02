import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LSTM, Bidirectional


def softmax_loss(y_true, y_pred):
    q, c = y_pred

    similarity_score = tf.matmul(q, K.transpose(c))
    per_sample_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=similarity_score,
        labels=tf.range(q.shape[0])
    )
    return tf.reduce_sum(per_sample_loss) / q.shape[0]


def mrr(y_true, y_pred):
    q, c = y_pred

    similarity_scores = tf.matmul(q, K.transpose(c))

    # extract the logits from the diagonal of the matrix, which are the logits corresponding to the ground-truth
    correct_scores = tf.linalg.diag_part(similarity_scores)

    # compute how many queries have bigger logits than the ground truth (the diagonal) -> which will be incorrectly ranked
    compared_scores = similarity_scores >= tf.expand_dims(correct_scores, axis=-1)

    compared_scores = tf.cast(compared_scores, tf.float16)
    # for each row of the matrix (query), sum how many logits are larger than the ground truth
    # ...then take the reciprocal of that to get the MRR for each individual query (you will need to take the mean later)
    return K.mean(tf.math.reciprocal(tf.reduce_sum(compared_scores, axis=1)))


def get_encoder_lstm(units, name):
    return Bidirectional(LSTM(units, dropout=0.5, recurrent_dropout=0.0), name=name)


def get_encoder_cnn(name):
    pass


def get_model_lstm(max_len_query, max_len_code, embeddings_dim_q, embeddings_dim_c):
    input_query = tf.keras.Input(shape=(max_len_query, embeddings_dim_q), name="in_query")
    input_code = tf.keras.Input(shape=(max_len_code, embeddings_dim_c), name="in_code")

    lstm_q = get_encoder_lstm(embeddings_dim_q, "query")(input_query)
    lstm_c = get_encoder_lstm(embeddings_dim_c, "code")(input_code)

    model = tf.keras.Model([input_query, input_code], [lstm_q, lstm_c])

    return model
