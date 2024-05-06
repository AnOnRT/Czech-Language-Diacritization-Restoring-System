import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

from itertools import product

import numpy as np

LETTERS_NODIA = "acdeeinorstuuyz"
LETTERS_DIA = "áčďéěíňóřšťúůýž"

parser = argparse.ArgumentParser()
parser.add_argument("--predict", default=False, type=bool, help="Test the model on the input text, otherwise train the model.")
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")
parser.add_argument("--seed", default=42, type=int, help="Random seed")

class Dictionary:
    def __init__(self,
                 name="fiction-dictionary.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2223/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            licence_name = name.replace(".txt", ".LICENSE")
            urllib.request.urlretrieve(url + licence_name, filename=licence_name)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dictionary to `variants`
        self.variants = {}
        with open(name, "r", encoding="utf-8-sig") as dictionary_file:
            for line in dictionary_file:
                nodia_word, *variants = line.rstrip("\n").split()
                self.variants[nodia_word] = variants



class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"

    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self,
                name="fiction-train.txt",
                url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2223/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)

            if name == "fiction-train.txt":
                licence_name = name.replace(".txt", ".LICENSE")
                urllib.request.urlretrieve(url + licence_name, filename=licence_name)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and split it into `data` and `target`.
        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            self.target = dataset_file.read()
        self.data = self.target.translate(self.DIA_TO_NODIA)


class Diacritization_Model:
    def __init__(self, X, y, window_size=5) -> None:
        self.model = MLPClassifier()
        self.X = X
        self.y = y

        self.processed_data = None 

        self.processed_target = None
        
        self.one_hot = OneHotEncoder(sparse=False)
        
        self.LETTERS_NODIA = "acdeeinorstuuyz"
        self.LETTERS_DIA = "áčďéěíňóřšťúůýž"
        self.window_size = window_size

        self.word_dict1 = np.unique([i for i in y.lower()])

        self.char2int = {c: i for i, c in zip(range(len(self.word_dict1)), self.word_dict1)}
        self.int2char = {i: c for i, c in zip(range(len(self.word_dict1)), self.word_dict1)}
        self.one_hot_fit()

    def one_hot_fit(self):
        self.one_hot.fit(np.expand_dims(range(len(self.word_dict1)), 1))
    
    def predict(self, test):
        pred_res = ""
        for i in range(len(test)):
            if not test[i] in self.LETTERS_NODIA:
                pred_res += test[i]
            else:
                word = " " * max(0, self.window_size - i) + \
                       test[max(0, i - self.window_size): min(len(test), i + self.window_size + 1)].lower() + \
                       " " * max(0, self.window_size - (len(test) - i - 1))
                list_of_data = []
                
                for each in word:
                    if each not in self.word_dict1:
                        each = ' '
                    list_of_data.append([self.char2int[each]])
                one_hoted = self.one_hot.transform(list_of_data)

                test_i = np.array(one_hoted).reshape(-1, (len(self.word_dict1) * (self.window_size * 2 + 1)))
                pred = self.model.predict(test_i)[0]
                if test[i].isupper():
                    pred_res += self.int2char[pred].upper()
                else:
                    pred_res += self.int2char[pred]
        return pred_res
    
    
    def preprocess_data(self):
        data = []
        target = []
        for i in range(len(self.X)):
            if self.X[i].lower() not in self.LETTERS_NODIA:
                continue
            word = " " * max(0, self.window_size - i) + \
                   self.X[max(0, i - self.window_size): min(len(self.X), i + self.window_size + 1)].lower() \
                   + " " * max(0, self.window_size - (len(self.X) - i - 1))
            list_of_data = []
            
            for each in word:
                list_of_data.append([self.char2int[each]])
                
            data.append(list_of_data)
            target.append(self.char2int[self.y[i].lower()])
        new_data = []
        
        for i in range(len(data)):
            new_data.append(self.one_hot.transform(data[i]))
        data = np.array(new_data).reshape(-1, (len(self.word_dict1) * (self.window_size * 2 + 1)))
        
        self.processed_data = data
        self.processed_target = target
        return data, target
    

    
    def fit(self, hidden_layer_sizes, activation, alpha, verbose=100, tol=0.0001, max_iter=200, learning_rate='invscaling'):
        self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, alpha=alpha, verbose=verbose, tol=tol, max_iter=max_iter, learning_rate=learning_rate)
        self.model.fit(self.processed_data, self.processed_target)

    def load(self, model_path):
        with lzma.open(model_path, "rb") as model_file:
            return pickle.load(model_file)
        
    def save(self, model_path):
        with lzma.open(model_path, "wb") as model_file:
            pickle.dump(self, model_file)
    


def predict(model, test):
    pred_res = ""
    for i in range(len(test)):
        if not test[i] in model.LETTERS_NODIA:
            pred_res += test[i]
        else:
            word = " " * max(0, model.window_size - i) + \
                    test[max(0, i - model.window_size): min(len(test), i + model.window_size + 1)].lower() + \
                    " " * max(0, model.window_size - (len(test) - i - 1))
            list_of_data = []
            
            for each in word:
                if each not in model.word_dict1:
                    each = ' '
                list_of_data.append([model.char2int[each]])
            one_hoted = model.one_hot.transform(list_of_data)

            test_i = np.array(one_hoted).reshape(-1, (len(model.word_dict1) * (model.window_size * 2 + 1)))
            pred = model.model.predict(test_i)[0]
            if test[i].isupper():
                pred_res += model.int2char[pred].upper()
            else:
                pred_res += model.int2char[pred]
    return pred_res

def accuracy(gold: str, system: str) -> float:
    assert isinstance(gold, str) and isinstance(system, str), "The gold and system outputs must be strings"

    gold, system = gold.split(), system.split()
    assert len(gold) == len(system), \
        "The gold and system outputs must have the same number of words: {} vs {}.".format(len(gold), len(system))

    words, correct = 0, 0
    for gold_token, system_token in zip(gold, system):
        words += 1
        correct += gold_token == system_token

    return correct / words

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def without_dia(s):
    for i in range(len(LETTERS_DIA)):
        occurences = find(s, LETTERS_DIA[i])
        if len(occurences) != 0:
            for j in occurences:
                s = s[:j] + LETTERS_NODIA[i] + s[j + 1:]
    return s

def max_match(s, arr):
    cur = ""
    best_match = ""
    best_points = 1000
    for i in arr:
        no_matches_count = 0
        for c_s, c_i in zip(s, i):
            if c_s != c_i:
                no_matches_count += 1
        if no_matches_count < best_points:
            best_points = no_matches_count
            best_match = i
    return best_match

def dictionize(pred_res):
    train_dict = Dictionary()
    new_res = ""
    cur = ""
    train_dict = Dictionary()
    for i in pred_res:
        if (i.isalpha()):
            cur += i
        else:
            cur_copy = cur
            if without_dia(cur_copy) in train_dict.variants:
                if len(train_dict.variants[without_dia(cur_copy)]) == 1:
                    new_res += train_dict.variants[without_dia(cur_copy)][0]
                else:
                    s = set(train_dict.variants[without_dia(cur_copy)])
                    if cur in s:
                        new_res += cur
                    else:
                        new_res += max_match(cur, train_dict.variants[without_dia(cur_copy)])
            else:
                new_res += cur
            cur = ""
            new_res += i

    new_res += cur

    return new_res



def main(args: argparse.Namespace) -> Optional[str]:
    if args.predict is False:
        np.random.seed(args.seed)
        train_main = Dataset()

        test_data = Dataset(name="diacritics-etest.txt", url="https://ufal.mff.cuni.cz/~zabokrtsky/courses/npfl124/data/")
        
        
        dev_data = Dataset(name="diacritics-dtest.txt", url="https://ufal.mff.cuni.cz/~zabokrtsky/courses/npfl124/data/")
        
        final_train = train_main.data # + "\n" + dev_train.data
        final_target = train_main.target #+ "\n" + dev_train.target

        param_grid = {
            'hidden_layer_sizes': [(300,), (800,), (300, 300)],
            'activation': ['logistic', 'relu'],
            'alpha': [0.0001, 0.001, 0.01]
        }

        parameter_combinations = list(product(param_grid['hidden_layer_sizes'], param_grid['activation'], param_grid['alpha']))
        validation_scores = []

        
        model = Diacritization_Model(final_train, final_target, window_size=5)
        print("Preprocessing the data is starting", file=sys.stderr)
        model.preprocess_data()

        print("Model training is starting", file=sys.stderr)
        
        i = 0
        for hidden_layers, activation, alpha in parameter_combinations:
            print("Training model with parameters:", hidden_layers, activation, alpha, file=sys.stderr)

            model.fit(hidden_layer_sizes=hidden_layers, activation=activation, alpha=alpha)
            
            with lzma.open("modelX" + str(i), "wb") as model_file:
                pickle.dump(model.model, model_file)
            print("Model saved to {}".format("modelX" + str(i)), file=sys.stderr)
            
            pred_res = model.predict(dev_data.data)
            accuracy_on_evaluationData = accuracy(pred_res, dev_data.target)
            print("Accuracy on dev data: {:.2f}%".format(100 * accuracy_on_evaluationData), file=sys.stderr)
            validation_scores.append((hidden_layers, activation, alpha, accuracy))
            i += 1


        best_parameters = max(validation_scores, key=lambda x: x[3])
        print("Best parameters based on validation data:", best_parameters, file=sys.stderr)
        
        
        model.fit(hidden_layer_sizes=best_parameters[0], activation=best_parameters[1], alpha=best_parameters[2])
        pred_res = model.predict(test_data.data)
        accuracy_on_evaluationData = accuracy(pred_res, test_data.target)

        print("Accuracy on evaluation data: {:.2f}%".format(100 * accuracy_on_evaluationData), file=sys.stderr)
        
    


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)