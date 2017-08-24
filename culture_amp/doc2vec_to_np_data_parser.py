"""abstract_data_parser script containing abstract base class"""
import numpy as np
from survey_util import Util
from abstract_data_parser import AbstractDataParser

class Doc2VecToNumpyDataParser(AbstractDataParser):
    """Doc2VecToNumpyDataParser"""
    def __init__(self, model):
        self.model = model
        self.x_array = None
        self.y_array = None
        self.y_dict = None
        self.y_inverse_dict = None

    def load(self, input_source):
        """load the data from the input_source
        input_source = file_location
        returns
            x_array = array of training data
            y_array = array of training labels
            y_dict = dict associated with the y labels i.e. {ALI.5 : 1}
            y_inverse_dict = dict associated with the y labels i.e. {1 : ALI.5}"""
        npz = np.load(input_source)
        self.x_array = npz['arr_0']
        self.y_array = npz['arr_1']
        self.y_dict = npz['arr_2']
        self.y_inverse_dict = npz['arr_3']
        return self.x_array, self.y_array, self.y_dict, self.y_inverse_dict

    def convert(self):
        """Convert the input_source
        returns
            x_array = array of training data
            y_array = array of training labels
            y_dict = dict associated with the y labels i.e. {ALI.5 : 1}
            y_inverse_dict = dict associated with the y labels i.e. {1 : ALI.5}"""
        if self.model is None:
            return None
        vector_out = []
        y_label = []
        count = 0
        for doc in self.model.docvecs:
            vector_out.append(doc)
            y_label.append(self.model.docvecs.index_to_doctag(count))
            count += 1
        self.x_array = np.array(vector_out)
        self.y_array = np.array(y_label)
        self.y_array = Util.clean_labels_np(self.y_array)
        self.y_dict, self.y_inverse_dict, self.y_array = Util.enumerate_y_labels(self.y_array)
        return self.x_array, self.y_array, self.y_dict, self.y_inverse_dict

    def save(self, output, data):
        """Save the data object to the output."""
        np.savez(output, self.x_array, self.y_array, self.y_dict, self.y_inverse_dict)
        return True
