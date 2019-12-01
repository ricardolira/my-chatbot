"""Chatbot tutorial."""
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, BasicLSTMCell
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.seq2seq import (prepare_attention,
                                        attention_decoder_fn_train,
                                        attention_decoder_fn_inference,
                                        dynamic_rnn_decoder)
import re
import time


def import_dataset(data_path, encoding='utf-8', errors='ignore'):
    """Import data from file path nad return parsed data."""
    with open(
        data_path, encoding=encoding, errors=errors
    ) as data:
        return_data = data.read().split('\n')
    data.close()
    return return_data


def map_lines(all_lines):
    """Create a dict contaning all dialogs mapped to its id."""
    lines_splitter = re.compile(r'\+{3}\$\+{3}')
    lines_dict = dict()
    for line in all_lines:
        _key = re.split(lines_splitter, line)[0].strip()
        _value = re.split(lines_splitter, line)[-1].strip()
        lines_dict.update({_key: _value})
    return lines_dict


def get_conversations(all_conversations):
    """Get the ids of all conversations."""
    conversations_pattern = re.compile(r'L\d+')
    ids_of_conversations = list()

    for conversaation in all_conversations[:-1]:
        _one_conversation = re.findall(conversations_pattern, conversaation)
        ids_of_conversations.append(_one_conversation)
    return ids_of_conversations


def separate_questions_from_answers(all_conversations):
    """Return two lsts containing questions and answer, respectively."""
    questions = list()
    answers = list()
    for conversation in all_conversations:
        for question, answer in zip(conversation[0:], conversation[1:]):
            questions.append(question)
            answers.append(answer)
    return questions, answers


def clean_text(text):
    """Clear text based on different non-expected english grammar."""
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", "will", text)
    text = re.sub(r"\'ve", "have", text)
    text = re.sub(r"\'re's", "are", text)
    text = re.sub(r"\'d", "would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text


def model_inputs():
    """Initialize the tensor flow variables."""
    inputs = tf.placeholder(tf.int32, [None, None], name='input')
    target = tf.placeholder(tf.int32, [None, None], name='target')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return inputs, target, lr, keep_prob


def proprocessed_targets(targets, word2int, batch_size=10):
    lhs = tf.fill([batch_size, 1], word2int['<SOS>'])
    rhs = tf.strided_slice(targets, [0, 0, ], [batch_size, -1], [1, 1])
    preprocessed_targets = tf.concat([lhs, rhs], 1)
    return preprocessed_targets


def encoder_rnn(
    rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length
):
    """Create the Encoder RNN Layer."""
    lstm = BasicLSTMCell(rnn_size)
    lstm_dropout = DropoutWrapper(lstm, input_keep_prob=keep_prob)
    encoder_cell = MultiRNNCell([lstm_dropout] * num_layers)
    _, encoder_state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=encoder_cell,
        cell_bw=encoder_cell,
        sequence_length=sequence_length,
        inputs=rnn_inputs,
        dtype=tf.float32
    )
    return encoder_state


def decode_training_set(
    encoder_state, decoder_cell, decoder_embedded_input,
    sequence_length, decoding_scope, output_function,
    keep_prob, batch_size
):
    """Decode the training set."""
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    (attention_keys,
     attentions_values,
     attention_score_function,
     attention_construct_function) = prepare_attention(
        attention_states,
        attention_option='bahdanau',
        num_units=decoder_cell.output_size
    )
    training_decoder_function = attention_decoder_fn_train(
        encoder_state[0],
        attention_keys,
        attentions_values,
        attention_score_function,
        attention_construct_function,
        name='attn_dec_train'
    )
    decoder_output, _, _ = dynamic_rnn_decoder(
        decoder_cell, training_decoder_function, decoder_embedded_input,
        sequence_length, scope=decoding_scope
    )
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)

    return output_function(decoder_output_dropout)


def decode_test_set(
    encoder_state, decoder_cell, decoder_embeddings_matrix,
    sos_id, eos_id, maximum_length, num_words,
    sequence_length, decoding_scope, output_function,
    keep_prob, batch_size
):
    """Decode the validation set."""
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    (attention_keys,
     attentions_values,
     attention_score_function,
     attention_construct_function) = prepare_attention(
        attention_states,
        attention_option='bahdanau',
        num_units=decoder_cell.output_size
    )
    test_decoder_function = attention_decoder_fn_inference(
        output_function,
        encoder_state[0],
        attention_keys,
        attentions_values,
        attention_score_function,
        attention_construct_function,
        decoder_embeddings_matrix,
        sos_id,
        eos_id,
        maximum_length,
        num_words,
        name='attn_dec_inf'
    )
    test_predictions, _, _ = dynamic_rnn_decoder(
        decoder_cell, test_decoder_function, scope=decoding_scope
    )

    return test_predictions


def decoder_rnn(
    decoder_embedded_input, decoder_embeddings_matrix, encoder_state,
    num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob,
    batch_size
):
    with tf.variable_scope('decoding') as decoding_scope:
        lstm = BasicLSTMCell(rnn_size)
        lstm_dropout = DropoutWrapper(lstm, input_keep_prob=keep_prob)
        decoder_cell = MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_iniitializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(
            x, num_words, None, scope=decoding_scope,
            weights_initializer=weights, biases_initializer=biases
        )
        training_predictions = decode_training_set(
            encoder_state, decoder_cell, decoder_embedded_input,
            sequence_length, decoding_scope, output_function,
            keep_prob, batch_size
        )
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(
            encoder_state, decoder_cell, decoder_embeddings_matrix,
            word2int['<SOS>'], word2int['<EOS>'], sequence_length - 1,
            num_words, decoding_scope, output_function, keep_prob,
            batch_size
        )
    return training_predictions, test_predictions


def seq2seq_model(
    inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words,
    questions_num_words, encoder_embedding_size, decoder_embedding_size,
    rnn_size, num_layers, questionswords2int
):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(
        inputs, answers_num_words, encoder_embedding_size,
        initializer=tf.random_uniform_initializer(0, 1)
    )
    encoder_state = encoder_rnn(
        encoder_embedded_input, rnn_size, num_layers, keep_prob,
        sequence_length
    )
    preprocessed_targets = proprocessed_targets(
        targets, questionswords2int, batch_size
    )
    decoder_embeddings_matrix = tf.Variable(
        tf.random_uniform(
            [questions_num_words + 1, decoder_embedding_size], 0, 1
        )
    )
    decoder_embedded_input = tf.nn.embedding_lookup(
        decoder_embeddings_matrix, preprocessed_targets
    )
    training_predictions, test_predictions = decoder_rnn(
        decoder_embedded_input, decoder_embeddings_matrix, encoder_state,
        questions_num_words, sequence_length, rnn_size, num_layers,
        questionswords2int, keep_prob, batch_size
    )
    return training_predictions, test_predictions
