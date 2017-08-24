"""survey_data_parser_test script containing test methods for data parsers"""
import unittest
import sys
sys.path.append("../")
from doc2vec_to_np_data_parser import Doc2VecToNumpyDataParser
from doc2vec_model import Doc2VecModel

class TestDataParser(unittest.TestCase):
    """TestDataParser"""
    doc2vec_file = "TestDoc2VecModel" + ".doc2vec"
    y_dict_test = {'ALI.5': 0, 'ENA.3': 1, 'INN.2': 2, 'TEA.2': 3}
    y_reverse_dict_test = {0: 'ALI.5', 1: 'ENA.3', 2: 'INN.2', 3: 'TEA.2'}

    def test_labelled_data_parser(self):
        """test the data parser"""
        doc2vec_model = Doc2VecModel()
        model = doc2vec_model.load(self.doc2vec_file)
        data_parser = Doc2VecToNumpyDataParser(model)
        x_out, y_out, y_dict, y_reverse_dict = data_parser.convert()
        print("x_out shape: ", x_out.shape)
        print("y_out shape: ", y_out.shape)
        print("y_dict: ", y_dict)
        print("y_reverse_dict: ", y_reverse_dict)
        self.assertTrue(x_out is not None)
        self.assertTrue(y_out is not None)
        self.assertTrue(y_dict is not None)
        self.assertTrue(y_reverse_dict is not None)
        self.assertTrue(y_dict == self.y_dict_test)
        self.assertTrue(y_reverse_dict == self.y_reverse_dict_test)

if __name__ == '__main__':
    unittest.main()
