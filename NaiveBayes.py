from collections import defaultdict
import email
import math
import os


class NaiveBayesSpam(object):

    def __init__(self, tag_dir):
        self.index_path = tag_dir
        self.sm_ham = 0.00000001
        self.sm_spam = 0.00000001
        self.num_ham = 0
        self.num_spam = 0
        self.v_uniham = defaultdict(int)
        self.v_unispam = defaultdict(int)
        self.v_biham = defaultdict(int)
        self.v_bispam = defaultdict(int)
        self.c_ham = defaultdict(int)
        self.c_spam = defaultdict(int)
        self.prob_ham = {}
        self.prob_spam = {}
        self.prob_biham = {}
        self.prob_bispam = {}
        self.prob_cham = {}
        self.prob_cspam = {}
        self.count_ham = 0
        self.count_spam = 0
        self.p_ham = 0.0
        self.p_spam = 0.0

    def taglize(self):
        index_f = open(self.index_path, 'r')
        corpus = [tuple((line.split('\n'))[0].split(' ..'))
                  for line in index_f]
        index_f.close()
        return corpus

    def tokenlize(self, email_path):
        tokens = []
        email_f = open(email_path, 'r')
        try:
            content = email.message_from_file(email_f)
        except UnicodeDecodeError:
            email_f.close()
            return tuple([[], False])
        email_f.close()
        lines = email.iterators.body_line_iterator(content)
        for line in lines:
            words = line.split()
            tokens.extend(words)
        return tuple([tokens, True])

    def prepare(self, head, corpus):
        for (tag, email) in corpus:
            if os.path.exists(head + email):
                tokens = self.tokenlize(head + email)
                if tokens[1]:
                    if tag == 'ham':
                        self.num_ham += 1
                        sum_len = len(tokens[0])
                        for i in range(sum_len):
                            word = tokens[0][i]
                            self.count_ham += 1
                            self.v_uniham[word] += 1
                            if i != sum_len - 1:
                                bigram = tuple(
                                    [tokens[0][i], tokens[0][i + 1]])
                                self.v_biham[bigram] += 1
                    else:
                        self.num_spam += 1
                        sum_len = len(tokens[0])
                        for i in range(sum_len):
                            word = tokens[0][i]
                            self.count_spam += 1
                            self.v_unispam[word] += 1
                            if i != sum_len - 1:
                                bigram = tuple(
                                    [tokens[0][i], tokens[0][i + 1]])
                                self.v_bispam[bigram] += 1

    def train(self):
        self.p_ham = float(self.num_ham / (self.num_ham + self.num_spam))
        self.p_spam = 1.0 - self.p_ham
        self.p_ham = math.log(self.p_ham)
        self.p_spam = math.log(self.p_spam)
        for word in self.v_uniham.keys():
            self.prob_ham[word] = math.log((self.v_uniham[word] + self.sm_ham) / (
                self.count_ham + self.sm_ham * (len(self.v_uniham) + 1)))
        for word in self.v_unispam.keys():
            self.prob_spam[word] = math.log((self.v_unispam[word] + self.sm_spam) / (
                self.count_spam + self.sm_spam * (len(self.v_unispam) + 1)))
        for bigram in self.v_biham.keys():
            self.prob_biham[bigram] = math.log(
                self.v_biham[bigram] * 1.0 / self.v_biham[bigram])
        for bigram in self.v_bispam.keys():
            self.prob_bispam[bigram] = math.log(
                self.v_bispam[bigram] * 1.0 / self.v_bispam[bigram])
        self.prob_ham['<UNK>'] = math.log(
            (self.sm_ham) / (self.count_ham + self.sm_ham * (len(self.v_uniham) + 1)))
        self.prob_spam['<UNK>'] = math.log(
            (self.sm_spam) / (self.count_spam + self.sm_spam * (len(self.v_unispam) + 1)))
        self.prob_cham['<UNK>'] = math.log(
            (self.sm_ham) / (self.count_ham + self.sm_ham * (len(self.c_ham) + 1)))
        self.prob_cspam['<UNK>'] = math.log(
            (self.sm_spam) / (self.count_spam + self.sm_spam * (len(self.c_spam) + 1)))

    def classification(self, email_path):
        mail_ham = self.p_ham
        mail_spam = self.p_spam
        if os.path.exists(email_path):
            tokens = self.tokenlize(email_path)
            if tokens[1]:
                words = tokens[0]
                for i in range(len(words)):
                    word = words[i]
                    if word in self.v_uniham:
                        mail_ham += self.prob_ham[word]
                    else:
                        mail_ham += self.prob_ham['<UNK>']
                    if word in self.v_unispam:
                        mail_spam += self.prob_spam[word]
                    else:
                        mail_spam += self.prob_spam['<UNK>']
                    if i != len(words) - 1:
                        bigram = tuple([words[i], words[i + 1]])
                        if bigram in self.v_biham:
                            mail_ham += self.prob_biham[bigram]
                        else:
                            mail_ham += self.prob_ham['<UNK>']
                        if bigram in self.v_bispam:
                            mail_spam += self.prob_bispam[bigram]
                        else:
                            mail_spam += self.prob_spam['<UNK>']
                    if len(word) in self.c_ham:
                        mail_ham += self.prob_cham[len(word)]
                    else:
                        mail_ham += self.prob_cham['<UNK>']
                    if len(word) in self.c_spam:
                        mail_spam += self.prob_cspam[len(word)]
                    else:
                        mail_spam += self.prob_cspam['<UNK>']
                if mail_ham > mail_spam:
                    return True
                return False
            else:
                print("Can't decode!")
                return False
        else:
            print('File not exist!')
            return False


def main():
    head = 'trec/trec07p'
    level = 'full'
    nbs = NaiveBayesSpam(head + '/' + level + '/indexp')
    nbs.prepare(head, nbs.taglize())
    nbs.train()
    print(nbs.prob_ham)


if __name__ == '__main__':
    main()
