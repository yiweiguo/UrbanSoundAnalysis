from collections import defaultdict
import email
import math
import os
import tensorflow as tf
import numpy as np
import operator
import pandas


class Features:
    def __init__(self, tag_dir, is_naive_bayes=True):
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
        self.feature_words = []
        #self.database = None
        self.feature_vector = None
        self.label_vector = None
        self.regularization_param = 0.001
        self.is_naive_bayes = is_naive_bayes


    def taglize(self):
        index_f = open(self.index_path, 'r')
        print(self.index_path[0:11])
        if self.index_path[0:11] == "trec/train":
            corpus = [tuple((line.split('\n'))[0].split(' '))
                  for line in index_f]
        else:
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
            if head == "trec/train":
                path = email
            else:
                path = head + email
            if os.path.exists(path):
                tokens = self.tokenlize(path)
                if tokens[1]:
                    if tag.lower() == 'ham':
                        self.num_ham += 1
                        sum_len = len(tokens[0])
                        for i in range(sum_len):
                            word = tokens[0][i]
                            self.count_ham += 1
                            self.v_uniham[word] += 1
                            self.c_ham[len(word)] += 1
                            if i != sum_len - 1:
                                if(self.is_naive_bayes):
                                    bigram = tuple([tokens[0][i], tokens[0][i + 1]])
                                else:
                                    bigram = tokens[0][i] + " " + tokens[0][i + 1]
                                self.v_biham[bigram] += 1
                    else:
                        self.num_spam += 1
                        sum_len = len(tokens[0])
                        for i in range(sum_len):
                            word = tokens[0][i]
                            self.count_spam += 1
                            self.v_unispam[word] += 1
                            self.c_spam[len(word)] += 1
                            if i != sum_len - 1:
                                if (self.is_naive_bayes):
                                    bigram = tuple([tokens[0][i], tokens[0][i + 1]])
                                else:
                                    bigram = tokens[0][i] + " " + tokens[0][i + 1]
                                self.v_bispam[bigram] += 1
        if(not self.is_naive_bayes):
            feature_number = 1000
            self.feature_words = self.mutual_information(feature_number)
            feature_columns, label_vector, size = self.get_train_set(head, corpus)
            #print(feature_columns.keys())
            self.feature_vector = pandas.DataFrame(data=feature_columns, dtype=np.int32)
            #print(self.feature_vector)
            self.label_vector = pandas.Series(data=label_vector, dtype=np.int32)


    def get_train_set(self, head, corpus):
        label_vector = np.array([])
        #feature_column = dict.fromkeys(self.feature_words, np.array([]))
        feature_column = {}
        num = 0
        for word in self.feature_words:
            key = "String" + str(num)
            num += 1
            feature_column[key] = np.array([])
        counter = 0
        for (tag, email) in corpus:
            if head == "trec/train":
                path = email
            else:
                path = head + email
            if os.path.exists(path):
                tokens = self.tokenlize(path)
                if tokens[1]:
                    if tag.lower() == 'ham':
                        label_vector = np.hstack([label_vector, 0])
                    else:
                        label_vector = np.hstack([label_vector, 1])
                    sum_len = len(tokens[0])
                    tmp_dict = defaultdict(int)
                    for i in range(sum_len):
                        word = tokens[0][i]
                        tmp_dict[word] += 1
                        if i != sum_len - 1:
                            bigram = tokens[0][i] + " " + tokens[0][i + 1]
                            tmp_dict[bigram] += 1
                    #print(tmp_dict)
                    for key in feature_column:
                        feature_column[key] = np.hstack([feature_column[key], 0])
                    for key in tmp_dict:
                        if key in self.feature_words:
                            ind = self.feature_words.index(key)
                            key_string = "String" + str(ind)
                            feature_column[key_string][counter] = tmp_dict[key]
                    counter += 1
        #print(feature_column)
        #print(label_vector)
        #print(counter)
        return feature_column, label_vector, counter


    def mutual_information(self, word_number):
        p_ham = float(self.num_ham / (self.num_ham + self.num_spam))
        p_spam = 1.0 - p_ham
        #num_words_and_phrases = self.count_ham + self.count_spam
        words_and_phrases = defaultdict(int)
        num_words_and_phrases = 0
        num_words_and_phrases_ham = 0
        num_words_and_phrases_spam = 0
        mutual_information_value = {}
        for key in self.v_uniham:
            #key_string = "String: " + key
            key_string = key
            words_and_phrases[key_string] += self.v_uniham[key]
            num_words_and_phrases += self.v_uniham[key]
            num_words_and_phrases_ham += self.v_uniham[key]
        for key in self.v_unispam:
            #key_string = "String: " + key
            key_string = key
            words_and_phrases[key_string] += self.v_unispam[key]
            num_words_and_phrases += self.v_unispam[key]
            num_words_and_phrases_spam += self.v_unispam[key]
        for key in self.v_biham:
            #key_string = "String: " + key
            key_string = key
            words_and_phrases[key_string] += self.v_biham[key]
            num_words_and_phrases += self.v_biham[key]
            num_words_and_phrases_ham += self.v_biham[key]
        for key in self.v_bispam:
            #key_string = "String: " + key
            key_string = key
            words_and_phrases[key_string] += self.v_bispam[key]
            num_words_and_phrases += self.v_bispam[key]
            num_words_and_phrases_spam += self.v_bispam[key]

        for token in words_and_phrases:
            p_word = float(words_and_phrases[token] / num_words_and_phrases)
            p_ham_joint = float((self.v_uniham[token] + self.v_biham[token]) / num_words_and_phrases_ham) * p_ham
            p_spam_joint = float((self.v_unispam[token] + self.v_bispam[token]) / num_words_and_phrases_spam) * p_spam
            '''print(token)
            print(p_ham_joint)
            print(p_word)
            print(p_ham)
            print(p_spam_joint)
            print(p_spam)
            print(self.v_uniham[token])
            print(self.v_biham[token])
            print(self.v_unispam[token])
            print(self.v_bispam[token])
            print(num_words_and_phrases_ham)'''
            try:
                mutual_information_value[token] = p_ham_joint * math.log(p_ham_joint / (p_word * p_ham), 2) + p_spam_joint * math.log(p_spam_joint / (p_word * p_spam), 2)
            except ValueError:
                if(p_spam_joint == 0 or p_ham_joint == 0):
                    mutual_information_value[token] = float("-Inf")
                else:
                    mutual_information_value[token] = float("Inf")

        sorted_mutual_info_value = sorted(mutual_information_value.items(), key=operator.itemgetter(1), reverse=True)
        #print(sorted_mutual_info_value[0])
        return [sorted_mutual_info_value[ind][0] for ind in range(0, word_number)]



    def train_log_prob(self):
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
            #print(self.v_biham[bigram])
            #print(self.v_uniham[bigram[0]])
            self.prob_biham[bigram] = math.log(
                self.v_biham[bigram] * 1.0 / self.v_uniham[bigram[0]])
        for bigram in self.v_bispam.keys():
            self.prob_bispam[bigram] = math.log(
                self.v_bispam[bigram] * 1.0 / self.v_unispam[bigram[0]])
        for length in self.c_ham.keys():
            self.prob_cham[length] = math.log(
                (self.c_ham[length] + self.sm_ham) / (self.count_ham + self.sm_ham * (len(self.c_ham) + 1)))
        for length in self.c_spam.keys():
            self.prob_cspam[length] = math.log((self.c_spam[length] + self.sm_spam) / (
                self.count_spam + self.sm_spam * (len(self.c_spam) + 1)))
        self.prob_ham['<UNK>'] = math.log(
            (self.sm_ham) / (self.count_ham + self.sm_ham * (len(self.v_uniham) + 1)))
        self.prob_spam['<UNK>'] = math.log(
            (self.sm_spam) / (self.count_spam + self.sm_spam * (len(self.v_unispam) + 1)))
        self.prob_biham['<UNK>'] = math.log(
            (self.sm_ham) / (self.count_ham + self.sm_ham * (len(self.v_biham) + 1)))
        self.prob_bispam['<UNK>'] = math.log(
            (self.sm_spam) / (self.count_spam + self.sm_spam * (len(self.v_bispam) + 1)))
        self.prob_cham['<UNK>'] = math.log(
            (self.sm_ham) / (self.count_ham + self.sm_ham * (len(self.c_ham) + 1)))
        self.prob_cspam['<UNK>'] = math.log(
            (self.sm_spam) / (self.count_spam + self.sm_spam * (len(self.c_spam) + 1)))
