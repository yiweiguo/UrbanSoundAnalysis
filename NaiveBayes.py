from builtins import int
from collections import defaultdict
from ctypes.wintypes import INT
import email
import math
import os
from uuid import int_


class NaiveBayesSpam(object):

    def __init__(self, dir):
        self.index_path = dir
        self.sm_ham = 0.00000001
        self.sm_spam = 0.00000001
        self.num_ham = 0
        self.num_spam = 0
        self.v_uniham = defaultdict(int)
        self.v_unispam = defaultdict(int)
        self.v_biham = defaultdict(int)
        self.v_bispam = defaultdict(int)
        self.prob_ham = {}
        self.prob_spam = {}
        self.count_ham = 0
        self.count_spam = 0
        self.count_biham = 0
        self.count_bispam = 0

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
                                self.count_biham += 1
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
                                self.count_bispam += 1


def main():
    head = 'trec/trec07p'
    level = 'full'
    nbs = NaiveBayesSpam(head + '/' + level + '/index')
    nbs.prepare(head, nbs.taglize())


if __name__ == '__main__':
    main()
