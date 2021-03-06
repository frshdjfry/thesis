import os
import re
import sys
import traceback
import zipfile
from lxml import etree


def create_folder_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def extract(path_to_zip_file, directory_to_extract_to):
    create_folder_if_not_exists(directory_to_extract_to)
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)


def unzip_files(input_dir, out_dir):
    files = os.listdir(input_dir)
    counter = 0
    for f in files:
        try:
            extract(input_dir + '/' + f, out_dir)
            counter += 1
        except Exception as e:
            print(e)
            print('failed to unzip ', f)
    print('%s file from %s article unzipped' % (counter, len(files)))


class Note():
    def __init__(self, note):
        self.note = note

    @property
    def step(self):
        return self.note.find('pitch').find('step').text

    @property
    def octave(self):
        return self.note.find('pitch').find('octave').text

    @property
    def note_type(self):
        return self.note.find('type').text

    @property
    def lyric_text(self):
        return self.note.find('lyric').find('text').text

    @property
    def lyric_syllabic(self):
        return self.note.find('lyric').find('syllabic').text

    @property
    def note_features(self):
        return self.step + '_' + self.octave + '_' + self.note_type


def extract_lyric_note(out_dir, file):
    with open(os.path.join(out_dir, file), 'r') as f:
        xml = f.read()
        xml = bytes(bytearray(xml, encoding='utf-8'))
        root = etree.XML(xml)
        word_res = []
        note_res = []
        for part in root.findall('part'):
            for measure in part.findall('measure'):
                notes = []
                syllables = []
                for note in measure.findall('note'):
                    if note.find('lyric') and note.find('pitch'):
                        enote = Note(note)

                        if enote.lyric_syllabic == 'single':
                            note_res.append(enote.note_features)
                            word_res.append(enote.lyric_text)

                        elif enote.lyric_syllabic == 'begin':
                            notes.append(enote.note_features)
                            syllables.append(enote.lyric_text)

                        elif enote.lyric_syllabic == 'middle':
                            notes.append(enote.note_features)
                            syllables.append(enote.lyric_text)

                        elif enote.lyric_syllabic == 'end':
                            notes.append(enote.note_features)
                            syllables.append(enote.lyric_text)
                            note_res.append(' '.join(notes))
                            word_res.append(' '.join(syllables))
                            notes = []
                            syllables = []
        return word_res, note_res


def save_lyric_notes(lyrics, notes):
    f1 = open('lyrics', 'w')
    f2 = open('notes', 'w')

    sent_l = []
    sent_n = []
    for l, n in zip(lyrics, notes):
        sent_l.append(l)
        sent_n.append(n)
        if l.endswith('.') or l.endswith(',') or l.endswith(';') or l.endswith('!') or l.endswith('?'):
            sent_l_str = ' '.join(sent_l)
            f1.write(re.sub(r"[^A-Z\ a-z]+", '', sent_l_str) + '\n')
            sent_n_str = ' '.join(sent_n)
            f2.write(sent_n_str + '\n')
            sent_l = []
            sent_n = []

    # with open('lyrics', 'w') as f:
    #     for l in lyrics:
    #         f.write(re.sub(r"[^A-Z\ a-z]+", '', l) + '\n')
    # with open('notes', 'w') as f:
    #     for n in notes:
    #         f.write(n + '\n')


def convert(out_dir):
    files = os.listdir(out_dir)
    counter = 0
    lyrics_res = []
    notes_res = []
    for f in files:
        if not f.endswith('.xml'):
            continue
        try:
            print('processing ', f)
            lyric, note = extract_lyric_note(out_dir, f)
            lyrics_res += lyric
            notes_res += note

            counter += 1
        except Exception:
            print(traceback.format_exc())
            print('failed to extract lyrics', f)

    print(lyrics_res[:10])
    save_lyric_notes(lyrics_res, notes_res)
    print('%s file from %s article unzipped' % (counter, len(files)))


def main(action, input_dir, out_dir):
    if action == 'unzip':
        unzip_files(input_dir, out_dir)
    elif action == 'convert':
        convert(out_dir)


if __name__ == '__main__':
    print('params: action[unzip|convert], input directory, output directory')
    main(sys.argv[1], sys.argv[2], sys.argv[3])
