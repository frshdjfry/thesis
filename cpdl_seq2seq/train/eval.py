import sys

from predict import suggest_translations


# Word Accuracy in Top-1
# Fuzziness in Top-1 )Mean F-score(
# Mean Reciprocal Rank )MRR(
# MAP ref

def load_eval_data(source_path):
    with open(source_path, 'r') as f:
        return [i.strip().lower() for i in f]

#
# def lcs(x, y, m, n):
#     if m == 0 or n == 0:
#         return 0
#     elif x[m - 1] == y[n - 1]:
#         return 1 + lcs(x, y, m - 1, n - 1)
#     else:
#         return max(lcs(x, y, m, n - 1), lcs(x, y, m - 1, n))
def lcs(s1, s2):
    matrix = [["" for x in range(len(s2))] for x in range(len(s1))]
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                if i == 0 or j == 0:
                    matrix[i][j] = s1[i]
                else:
                    matrix[i][j] = matrix[i-1][j-1] + s1[i]
            else:
                matrix[i][j] = max(matrix[i-1][j], matrix[i][j-1], key=len)

    cs = matrix[-1][-1]

    return len(cs)

def word_acc_top_1(predicteds, targets):
    correct = 0
    for predicted, target in zip(predicteds, targets):
        if predicted == target:
            correct += 1
    return float(correct) / len(targets)


def fuzziness_in_top_1(predicteds, targets):
    fscores = []
    for predicted, target in zip(predicteds, targets):
        clean_predicted = ''.join(predicted.split())
        clean_target = ''.join(target.split())
        precision = float(lcs(clean_predicted, clean_target)) / len(clean_predicted)
        recall = float(lcs(clean_predicted, clean_target)) / len(clean_target)
        if precision and recall:
            fscore = 2 * precision * recall / (precision + recall)
            fscores.append(fscore)
    return sum(fscores) / len(fscores)


def mean_reciprocal_rank(suggestions, targets):
    ranks = []
    for predicteds, target in zip(suggestions, targets):
        for i, p in enumerate(predicteds, 1):
            if p == target:
                ranks.append(i)
                break
    return float(sum(ranks)) / len(ranks) if len(ranks) else 0


def map_ref(suggested_predictions, targets):
    correct = 0
    for predictions, target in zip(suggested_predictions, targets):
        if target in predictions:
            correct += 1
    return float(correct) / len(targets)


def main(source_path, target_path):
    sources = load_eval_data(source_path)
    targets = load_eval_data(target_path)
    predictions = []
    suggested_predictions = []
    for i, source in enumerate(sources):
        suggestions = suggest_translations(source)
        predictions.append(suggestions[0].strip('<EOS>').strip())
        suggested_predictions.append([s.strip('<EOS>').strip() for s in suggestions])
        print('predicted: "%s"' % suggestions[0].strip('<EOS>').strip())
        print('target   : "%s"' % targets[i])
        print()

    wat1 = word_acc_top_1(predictions, targets)
    print('wat1: ', wat1)
    ft1 = fuzziness_in_top_1(predictions, targets)
    print('ft1: ', ft1)
    mrr = mean_reciprocal_rank(suggested_predictions, targets)
    print('mrr: ', mrr)
    mapr = map_ref(suggested_predictions, targets)
    print('mapr: ', mapr)


if __name__ == '__main__':
    print('params: source_path, target_path')
    main(sys.argv[1], sys.argv[2])
