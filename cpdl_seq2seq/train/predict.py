import pickle
import sys

import tensorflow as tf

display_step = 300

epochs = 13
batch_size = 128

rnn_size = 128
num_layers = 3

encoding_embedding_size = 200
decoding_embedding_size = 200

learning_rate = 0.001
keep_probability = 0.5


def load_params():
    with open('params.p', mode='rb') as in_file:
        return pickle.load(in_file)


def load_preprocess():
    with open('preprocess.p', mode='rb') as in_file:
        return pickle.load(in_file)


_, (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = load_preprocess()
load_path = load_params()


def sentence_to_seq(sentence, vocab_to_int):
    results = []
    for word in sentence.split(" "):
        if word in vocab_to_int:
            results.append(vocab_to_int[word])
        else:
            results.append(vocab_to_int['<UNK>'])

    return results


def predict(translate_sentence, load_path):
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(load_path + '.meta')
        loader.restore(sess, load_path)

        input_data = loaded_graph.get_tensor_by_name('input:0')
        logits = loaded_graph.get_tensor_by_name('predictions:0')
        target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
        keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

        translate_logits = sess.run(logits, {input_data: [translate_sentence] * batch_size,
                                             target_sequence_length: [len(translate_sentence) * 2] * batch_size,
                                             keep_prob: 1.0})
        return translate_logits


def translate(sentence):
    sentence = sentence_to_seq(sentence, source_vocab_to_int)
    translate_logits = predict(sentence, load_path)
    return " ".join([target_int_to_vocab[i] for i in translate_logits[0]])


def suggest_translations(sentence):
    sentence = sentence_to_seq(sentence, source_vocab_to_int)
    translate_logits = predict(sentence, load_path)
    res = []
    for translate_logit in translate_logits:
        res.append(" ".join([target_int_to_vocab[i] for i in translate_logit]))
    return res


def main(translate_sentence):
    translate_sentence = sentence_to_seq(translate_sentence, source_vocab_to_int)
    translate_logits = predict(translate_sentence, load_path)
    print('  Lyrics: {}'.format([source_int_to_vocab[i] for i in translate_sentence]))

    print('\nPrediction')
    print('  Notes: {}'.format(" ".join([target_int_to_vocab[i] for i in translate_logits[0]])))


if __name__ == '__main__':
    print('params: source sequence')
    main(sys.argv[1])
