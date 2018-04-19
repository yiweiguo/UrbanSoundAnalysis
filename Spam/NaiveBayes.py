import os
from FeatureLoading import Features


class NaiveBayesSpam:

    def __init__(self, features):
        self.features = features

    def testlize(self, test_dir):
        index_f = open(test_dir, 'r')
        corpus = [tuple((line.split('\n'))[0].split(' ..'))
                  for line in index_f]
        index_f.close()
        return corpus

    def classification(self, email_path):
        mail_ham = self.features.p_ham
        mail_spam = self.features.p_spam
        if os.path.exists(email_path):
            tokens = self.features.tokenlize(email_path)
            if tokens[1]:
                words = tokens[0]
                for i in range(len(words)):
                    word = words[i]
                    if word in self.features.v_uniham:
                        mail_ham += self.features.prob_ham[word]
                    else:
                        mail_ham += self.features.prob_ham['<UNK>']
                    if word in self.features.v_unispam:
                        mail_spam += self.features.prob_spam[word]
                    else:
                        mail_spam += self.features.prob_spam['<UNK>']
                    if i != len(words) - 1:
                        bigram = tuple([words[i], words[i + 1]])
                        if bigram in self.features.v_biham:
                            mail_ham += self.features.prob_biham[bigram]
                        else:
                            mail_ham += self.features.prob_biham['<UNK>']
                        if bigram in self.features.v_bispam:
                            mail_spam += self.features.prob_bispam[bigram]
                        else:
                            mail_spam += self.features.prob_bispam['<UNK>']
                    if len(word) in self.features.c_ham:
                        mail_ham += self.features.prob_cham[len(word)]
                    else:
                        mail_ham += self.features.prob_cham['<UNK>']
                    if len(word) in self.features.c_spam:
                        mail_spam += self.features.prob_cspam[len(word)]
                    else:
                        mail_spam += self.features.prob_cspam['<UNK>']
                if mail_ham > mail_spam:
                    return (True, True)
                return (True, False)
            else:
                return (False, False)
        else:
            return (False, False)


def main():
    head = 'trec/trec07p'
    level = 'partial'
    features = Features(head + '/' + level + '/index1000')
    features.prepare(head, features.taglize())
    features.train_log_prob()
    nbs = NaiveBayesSpam(features)

    while True:
        test_head = input('Input test head:')
        if test_head == 'q':
            return
        test_level = input('Input test level:')
        print('Testing...\n')
        cases = nbs.testlize(test_head + '/' + test_level + '/index')
        total = 0
        accurate = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for case in cases:
            res = nbs.classification(test_head + case[1])
            if res[0]:
                total += 1
                if res[1] == True and (case[0]).lower() == 'ham':
                    accurate += 1
                    TP += 1
                elif res[1] == False and (case[0]).lower() == 'spam':
                    accurate += 1
                    TN += 1
                elif res[1] == True and (case[0]).lower() == 'spam':
                    FP += 1
                elif res[1] == False and (case[0]).lower() == 'ham':
                    FN += 1
        acc_rate = float(accurate / total * 100)
        print('Tested ' + str(total) + ' cases.')
        print('Test result: accuracy = ' + str(acc_rate) + '%')


if __name__ == '__main__':
    main()
