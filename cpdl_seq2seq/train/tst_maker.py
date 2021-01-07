import random
import sys


def write_data(source_path, data):
    with open(source_path, 'w') as f:
        for i in data:
            f.write(i + '\n')


def load_eval_data(source_path):
    with open(source_path, 'r') as f:
        return [i.strip() for i in f]


def main(source_path, target_path):
    sources = load_eval_data(source_path)
    targets = load_eval_data(target_path)
    source_tsts = []
    target_tsts = []

    tst_len = int(len(sources) / 10)
    for i in range(tst_len):
        source_tsts.append(sources.pop(random.randint(0, len(sources))))
        target_tsts.append(targets.pop(random.randint(0, len(targets))))

    write_data(source_path + '-tst', source_tsts)
    write_data(target_path + '-tst', target_tsts)
    write_data(source_path + '-train', sources)
    write_data(target_path + '-train', targets)


if __name__ == '__main__':
    print('params: source_path, target_path')
    main(sys.argv[1], sys.argv[2])
