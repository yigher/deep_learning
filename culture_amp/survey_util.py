"""survey_util.py script"""
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet as wn
from nltk import pos_tag


class Util(object):
    """Util class"""
    @staticmethod
    def clean_label(label_input):
        """clean_label
        label_input = string
        returns
            label_input = string"""
        label_input = str(label_input.replace(" ", ""))
        label_input = label_input.replace(",", "")
        if label_input.find("_") >= 0:
            label_input = label_input[:label_input.find("_")]
        return label_input

    @staticmethod
    def clean_labels_np(label_inputs):
        """clean_labels_np - cleans the classification labels
        label_inputs = numpy string array
        returns
            numpy string array"""
        f_out = np.vectorize(Util.clean_label)
        return f_out(label_inputs)

    @staticmethod
    def invert_dict(input_dict):
        """method to invert a dict
        input_dict = dict{}
        returns
            new_dict = dict{}"""
        new_dict = {}
        for key, value in input_dict.items():
            new_dict[value] = key
        return new_dict

    @staticmethod
    def enumerate_y_labels(y_labels):
        """return a dict of key value pairs, and np array of dict values
        y_labels = list of y labels
        returns
            y_dict = dict associated with the y labels i.e. {ALI.5 : 1}
            inverse_y_dict = dict associated with the y labels i.e. {1 : ALI.5}
            y_enumerate_labels = numpy array of enumerate labels i.e. [1,1,1,2,1,1,0]"""
        y_array = np.array(y_labels, dtype=np.chararray)
        y_dict = {}
        y_enumerate_labels = np.zeros(y_array.shape)
        y_unique = np.unique(y_array)
        for i in range(len(y_unique)):
            y_dict[y_unique[i]] = i
        for i in range(len(y_array)):
            y_enumerate_labels[i] = y_dict[y_array[i]]
        inverse_y_dict = Util.invert_dict(y_dict)
        return y_dict, inverse_y_dict, y_enumerate_labels

    @staticmethod
    def remove_stop_words(input_txt):
        """method that remove stop words
        input_txt = string
        returns
            string"""
        return ' '.join(
            [word for word in input_txt.split() if word not in stopwords.words("english")]
        )

    @staticmethod
    def stem_words(input_txt):
        """method that stem the words
        input_txt = string
        returns
            string"""
        stemmer = PorterStemmer()
        return stemmer.stem(input_txt)

    @staticmethod
    def get_argmax_from_prob_array(input_array):
        """method that returns the idx with the highest prob.
        input_array = numpy array with numeric values
        returns
            integer index of highest value in array"""
        return np.argmax(input_array, axis=1)

    @staticmethod
    def sort_np_idx_values(input_array):
        """method that returns the array index from smallest to largest
        input_array = numpy numeric array
        returns
            numpy numeric array"""
        return input_array.argsort()[-3:][::-1]

    @staticmethod
    def get_np_dict_value_by_idx(dict_input, np_input, top_n):
        """method that returns the corresponding top n dict value in the array
        dict_input = dict associated with the y labels i.e. {ALI.5 : 1}
        np_input = numpy array of enumerated labels
        top_n = top n number of labels to return
        returns
            list of top n number of labels"""
        np_input_sorted = Util.sort_np_idx_values(np_input)
        return [[dict_input[first], dict_input[second]] for second, first in np_input_sorted[:, -top_n:]]

    @staticmethod
    def get_tuple_doc2vec_similar(list_input, top_n):
        """get tuple from doc2vec similar
        list_input = list of tuples i.e. [(ALi.5, 0.9987)]
        returns
            top n values of list_input"""
        return [(Util.clean_label(a[0]), a[1]) for a in list_input[:top_n]]

    @staticmethod
    def remove_pronouns(input_text):
        """remove_pronouns
        input_text = string
        returns
            string"""
        input_text = nltk.word_tokenize(input_text)
        tokenized_sentence = pos_tag(input_text)
        tokenized_sentence = [s for s in tokenized_sentence if s[1] != 'NN' and s[1] !='PRP' and s[1] !='IN' and s[1] !='CC' and s[1] !='MD']
        return "".join([" "+i[0] for i in tokenized_sentence]).strip()
