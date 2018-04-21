from collections import defaultdict
import email
import math
import os
import numpy as np
from FeatureLoading import Features
import tensorflow as tf
import pandas

class NeuralNetwork(object):
    def __init__(self, features, classifier):
        self.features = features
        self.classifier = classifier

    def testlize(self, test_dir):
        index_f = open(test_dir, 'r')
        corpus = [tuple((line.split('\n'))[0].split(' ..'))
                  for line in index_f]
        index_f.close()
        return corpus

    def eval_input_fn(self, feature_vectors, feature_labels):
        dataset = tf.data.Dataset.from_tensor_slices((dict(feature_vectors), feature_labels)).batch(1000)
        return dataset

    def classification(self, super_path, corpus):
        test_class_vector = np.array([])
        test_feature_columns = {}
        for ind in range(0, len(self.features.feature_words)):
            key_string = "String" + str(ind)
            test_feature_columns[key_string] = np.array([])
        counter = 0
        for label, path in corpus:
            for key in test_feature_columns:
                test_feature_columns[key] = np.hstack([test_feature_columns[key], 0])
            if label.lower() == "spam":
                test_class_vector = np.hstack([test_class_vector, 1])
            else:
                test_class_vector = np.hstack([test_class_vector, 0])
            if super_path == "trec/train":
                email_path = path
            else:
                email_path = super_path + path
            if os.path.exists(email_path):
                tokens = self.features.tokenlize(email_path)
                if tokens[1]:
                    words = tokens[0]
                    for i in range(len(words)):
                        word = words[i]
                        if word in self.features.feature_words:
                            index = self.features.feature_words.index(word)
                            key_string = "String" + str(index)
                            test_feature_columns[key_string][counter] += 1
                            if i != len(words) - 1:
                                bigram = tokens[0][i] + " " + tokens[0][i + 1]
                                if bigram in self.features.feature_words:
                                    index = self.features.feature_words.index(bigram)
                                    key_string = "String" + str(index)
                                    test_feature_columns[key_string][counter] += 1
            counter += 1
            '''
        for key in test_feature_columns:
            test_feature_columns[key] = tf.constant(test_feature_columns[key], tf.int64)
        test_class_vector = tf.constant(test_class_vector, tf.int64)'''
        test_feature_columns = pandas.DataFrame(data=test_feature_columns, dtype=np.int32)
        test_class_vector = pandas.Series(data=test_class_vector, dtype=np.int32)
        result = self.classifier.evaluate(
            input_fn=lambda : self.eval_input_fn(test_feature_columns, test_class_vector))
        print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**result))




def DNN_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    #print(final_database)
    return dataset

def main():
    head = 'trec/trec07p'
    level = 'train'
    features = Features(head + '/' + level + '/index', is_naive_bayes=False)
    features.prepare(head, features.taglize())
    columns = []
    for ind in range(0, len(features.feature_words)):
        key_string = "String" + str(ind)
        columns.append(tf.feature_column.numeric_column(key=key_string))
    classifier = tf.estimator.DNNClassifier(feature_columns=columns,
                                                 hidden_units=[2000, 500], n_classes=2)
    #print(len(columns))
    #print(features.label_vector)
    classifier.train(input_fn=lambda:DNN_input_fn(features.feature_vector, features.label_vector, 100), steps=1000)
    nnClassifier = NeuralNetwork(features, classifier)
    while True:
        test_head = input('Input test head:')
        if test_head == 'q':
            return
        test_level = input('Input test level:')
        print('Testing...\n')
        cases = nnClassifier.testlize(test_head + '/' + test_level + '/index')
        nnClassifier.classification(test_head, cases)


if __name__ == '__main__':
    main()