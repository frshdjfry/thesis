import pickle
import sys

import tensorflow as tf
from lxml import etree

display_step = 300

epochs = 13
batch_size = 32

rnn_size = 128
num_layers = 3

encoding_embedding_size = 200
decoding_embedding_size = 200

learning_rate = 0.001
keep_probability = 0.5

note_template = """
<note default-x="155.23" default-y="-45.00">
<pitch>
  <step>{step}</step>
  <octave>{octave}</octave>
  </pitch>
<duration>4</duration>
<voice>1</voice>
<type>{type}</type>
<stem>up</stem>
<lyric default-x="6.58" default-y="-45.34" relative-y="-30.00">
  <syllabic>single</syllabic>
  <text>{lyric}</text>
</lyric>
</note>
"""


barline_template = """
<barline location="right">
<bar-style>light-light</bar-style>
</barline>
"""

rest_template = """
<note>
<rest/>
<duration>4</duration>
<voice>1</voice>
<type>eighth</type>
</note>
"""


measure_start_template = """
<measure number="{number}">
"""

measure_end_template = """
</measure>
"""

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


def read_file(file_name):
    with open(file_name, 'r') as f:
        return [i.strip() for i in f]


def write_file(file_name, data):
    with open(file_name, 'w') as f:
        for i in data:
            f.write(i + '\n')
        f.close()


def format_xml(template_name, output_file, notes):
    with open(template_name, 'r') as f:
        xml = f.read()
        xml = bytes(bytearray(xml, encoding='utf-8'))
        root = etree.XML(xml)
        for part in root.findall('part'):
            # for measure in part.findall('measure'):
            # for note in notes:
                # measure.append(etree.fromstring(note))
            # print(' '.join(notes))
            # print(etree.fromstring(notes[0]))
            # part.append(etree.fromstring(notes[0] + notes[1]))
            for measure in notes:
                part.append(etree.fromstring(measure))

        tree = etree.ElementTree(root)
        tree.write(output_file + '.xml', pretty_print=True, xml_declaration=True, encoding="utf-8")


def format_note(lyric, note_features, template):
    step, octave, type = note_features.split('_')
    res = template.format(step=step.upper(), octave=octave, type=type, lyric=lyric)
    return res


def main(file_name):
    translate_sentences = read_file(file_name)
    predicted_notes = []
    for translate_sentence in translate_sentences:
        translate_sentence_seq = sentence_to_seq(translate_sentence, source_vocab_to_int)
        translate_logits = predict(translate_sentence_seq, load_path)
        print('  Lyrics: {}'.format([source_int_to_vocab[i] for i in translate_sentence_seq]))
        # print('  Lyrics: {}'.format(translate_sentence))

        print('\nPrediction')
        print('  Notes: {}'.format(" ".join([target_int_to_vocab[i] for i in translate_logits[0]])))
        predicted_notes.append([target_int_to_vocab[i] for i in translate_logits[0]])

    print(translate_sentences)
    print(predicted_notes)
    xml_notes = []
    counter = 1
    for sent, notes in zip(translate_sentences, predicted_notes):
        mes_start = measure_start_template.format(number=counter)

        words = sent.split()
        notes = notes[:-1]
        # print(words, notes)
        # for lyric, note in zip(words, notes):
        #     xml_notes.append(format_note(lyric, note, note_template))
        #     mes_start += format_note(lyric, note, note_template)
        for note in notes:
            mes_start += format_note('lyric', note, note_template)
        mes_start += rest_template
        mes_start += measure_end_template
        # xml_notes.append(measure_end_template)
        xml_notes.append(mes_start)
        counter += 1
    print(xml_notes)
    format_xml('template.xml', file_name.split('.')[0], xml_notes)
    # write_file(file_name.split('.')[0]+'.notes', predicted_notes)


if __name__ == '__main__':
    print('params: file name')
    main(sys.argv[1])
