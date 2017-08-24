"""doc2vec_model.py"""
import sys
import pickle
from gensim.models import Doc2Vec
from sklearn.utils import shuffle
from abstract_model import AbstractModel
from survey_data_parser import SurveyDoc2VecDataParser
from survey_util import Util
import numpy as np
class Doc2VecModel(AbstractModel):
    """Doc2VecModel"""
    def __init__(
            self,
            total_examples=None,
            total_words=None,
            window_size=5,
            epochs=100,
            alpha=0.025,
            min_alpha=0.025,
            word_count=0,
            queue_factor=2,
            report_delay=1.0,
            compute_loss=None,
            dim_size=50,
            worker_num=4,
            min_count=1,
            down_sample=1e-4
    ):
        self.total_examples = total_examples
        self.total_words = total_words
        self.window_size = window_size
        self.epochs = epochs
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.word_count = word_count
        self.queue_factor = queue_factor
        self.report_delay = report_delay
        self.compute_loss = compute_loss
        self.dim_size = dim_size # dimensionality of the feature vectors in output
        self.worker_num = worker_num # use this many worker threads to train the model
        self.min_count = min_count
        self.down_sample = down_sample # threshold for configuring which higher-frequency words are randomly downsampled
        self.model = None

    def fit(self, inputs):
        """fit method. 
        inputs = list of TagDocument
        returns Doc2Vec"""
        if self.model is None:
            self.model = Doc2Vec(
                min_count=self.min_count,
                alpha=self.alpha,
                min_alpha=self.min_alpha,
                dm=1,
                window=self.window_size,
                size=self.dim_size,
                sample=self.down_sample,
                workers=self.worker_num
            )
        self.model.build_vocab(inputs)

        alpha_delta = (self.alpha - self.min_alpha) / self.epochs
        shuffled = list(inputs)
        for i in range(self.epochs):
            # shuffle to introduce some randomness
            shuffled = shuffle(shuffled)
            self.model.train(shuffled, total_examples=len(shuffled), epochs=self.model.iter)
            # reduce learning rate per epoch
            self.model.alpha -= alpha_delta
            self.model.min_alpha = self.model.alpha
            if i % 10 == 0:
                print("iterations:", i)

        return self.model

    def predict(self, inputs):
        """predict
        inputs = array of string
        returns 
            new_vector = numpy array vector representation of the sentence
            pred_out = tuple list of most likely classifications"""
        new_vector = self.model.infer_vector(inputs)
        pred_out = self.model.docvecs.most_similar([new_vector])
        return new_vector, pred_out

    def save(self, file_location):
        """save model method
        file_location = string file location"""
        self.model.save(file_location)

    def load(self, file_location):
        """load model method"""
        self.model = Doc2Vec.load(file_location)
        return self.model

if __name__ == "__main__":
    MODEL = Doc2VecModel(epochs=100, dim_size=200)
    MODEL_OUT = MODEL.load("model_files/DOC2VEC_MODEL.doc2vec")
    TEST_STRING = "I know where to find information to do my job well".lower()
    TEST_STRING = Util.remove_stop_words(TEST_STRING)
    TEST_STRING = Util.stem_words(TEST_STRING)
    NEW_VECTOR = MODEL_OUT.infer_vector(TEST_STRING)
    print("NEW_VECTOR: ", NEW_VECTOR)
    SAME = MODEL_OUT.docvecs.most_similar([NEW_VECTOR])
    print(SAME)
