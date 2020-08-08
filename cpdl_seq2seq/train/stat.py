import sys
from operator import itemgetter


def load_eval_data(source_path):
    with open(source_path, 'r') as f:
        return [i.strip().lower() for i in f]


def notes_stats(targets):
    result = {}
    total = 0
    for t in targets:
        for note in t.split():
            if result.get(note):
                result[note] += 1
            else:
                result[note] = 1
            total += 1
    return result, total


def main(target_path):
    targets = load_eval_data(target_path)
    notes_stat, total = notes_stats(targets)
    sorted_list = sorted(notes_stat.items(), key=itemgetter(1), reverse=True)
    for s in sorted_list:
        print(s[0], s[1])
        print(float(s[1])/total)
        print()


if __name__ == '__main__':
    print('params: target_path')
    main(sys.argv[1])
#
# b 69247
# 0.14741025679234157
#
# d 68959
# 0.1467971738579734
#
# a 67353
# 0.1433783849947952
#
# g 67191
# 0.14303352584421308
#
# c 66748
# 0.14209048508058422
#
# e 66531
# 0.14162854411961928
# 
# f 63728
# 0.1356616293104733
#
