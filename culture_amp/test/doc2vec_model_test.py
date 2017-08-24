"""data_parser_test script containing test methods for data parsers"""
import unittest
import sys
import os
import os.path
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
sys.path.append("../")
from survey_data_parser import SurveyDoc2VecDataParser
from doc2vec_model import Doc2VecModel
from survey_util import Util

class TestDoc2VecModel(unittest.TestCase):
    """TestDoc2VecModel"""
    file_directory = "../data/"
    labelled_file = file_directory + "labeled_data" + ".csv"
    ref_questions_file = file_directory + "reference_questions" + ".csv"
    test_questions_file = file_directory + "test_questions" + ".txt"
    x_out = []
    y_out = []

    train_string_idx = 38
    train_string = "I understand what's expected of me to be successful in my role"
    train_string_label = "ALI.5"

    doc2vec_filename = "../model_files/DOC2VEC_MODEL.doc2vec"

    def test_doc2vec_fit(self):
        """test the Doc2VecModel fit"""
        data_parser = SurveyDoc2VecDataParser()
        x_out, y_out = data_parser.load(self.labelled_file)
        self.assertTrue(x_out is not None)
        self.assertTrue(y_out is not None)
        self.assertTrue(x_out[self.train_string_idx] == self.train_string.lower())
        self.assertTrue(y_out[self.train_string_idx] == self.train_string_label)
        labeled_sentences = data_parser.convert()
        model_out = Doc2VecModel()
        model_out.fit(labeled_sentences)
        model_out.save(self.doc2vec_filename)
        self.assertTrue(os.path.exists(self.doc2vec_filename))

    def test_doc2vec_predict(self):
        """test the Doc2VecModel predict"""
        model_out = Doc2VecModel()
        model_out.load(self.doc2vec_filename)
        pred_out = model_out.predict(self.train_string.lower())
        pred_out1 = np.array([i[0] for i in pred_out])
        pred_out1 = Util.clean_labels_np(pred_out1)
        self.assertTrue(pred_out1[0] == self.train_string_label)

    def test_doc2vec_visualise(self):
        """test the Doc2VecModel visualise"""
        model_out = Doc2VecModel()
        model_out.load(self.doc2vec_filename)
        model = model_out.model
        vector_out = []
        y_label = []
        count = 0
        for doc in model.docvecs:
            vector_out.append(doc)
            y_label.append(model.docvecs.index_to_doctag(count))
            count += 1

        vector_out = np.array(vector_out)
        y_label = np.array(y_label)
        y_label = Util.clean_labels_np(y_label)
        x_embedded = TSNE(perplexity=50, n_iter=10000, verbose=1).fit_transform(vector_out)
        tsne_df = pd.DataFrame(dict(x=x_embedded[:, 0], y=x_embedded[:, 1], label=y_label))
        groups = tsne_df.groupby('label')
        plt.figure(num=1, figsize=(20, 20), facecolor="w", edgecolor="k")
        _, ax_out = plt.subplots()
        ax_out.margins(0.05)
        for name, group in groups:
            ax_out.plot(group.x, group.y, marker='o', linestyle='', ms=5, label=name)
            ax_out.legend()
        plt.savefig("TSNE_visualise.png")
        self.assertTrue(os.path.exists("TSNE_visualise.png"))

if __name__ == '__main__':
    unittest.main()
