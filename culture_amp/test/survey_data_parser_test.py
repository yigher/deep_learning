"""survey_data_parser_test script containing test methods for data parsers"""
import unittest
import sys
sys.path.append("../")
from survey_data_parser import SurveyDoc2VecDataParser

class TestDataParser(unittest.TestCase):
    """TestDataParser"""
    file_directory = "../data/"
    labelled_file = file_directory + "labeled_data" + ".csv"
    ref_questions_file = file_directory + "reference_questions" + ".csv"
    test_questions_file = file_directory + "test_questions" + ".txt"
    x_out = []
    y_out = []

    train_string_idx = 38
    ref_string = "We are encouraged to be innovative even though some of our initiatives may not succeed"
    ref_string_label = "INN.2"

    ref_string_idx = 3
    train_string = "I understand what's expected of me to be successful in my role"
    train_string_label = "ALI.5"

    test_string_idx = 8
    test_string = "We hold ourselves and our team members accountable for results"

    def test_labelled_data_parser(self):
        """test the data parser"""
        data_parser = SurveyDoc2VecDataParser()
        x_out, y_out = data_parser.load(self.labelled_file)
        self.assertTrue(x_out is not None)
        self.assertTrue(y_out is not None)
        self.assertTrue(x_out[self.train_string_idx] == self.train_string)
        self.assertTrue(y_out[self.train_string_idx] == self.train_string_label)

    def test_ref_questions_data_parser(self):
        """test the ref questions data parser"""
        data_parser = SurveyDoc2VecDataParser()
        x_out, y_out = data_parser.load(self.ref_questions_file)
        self.assertTrue(x_out is not None)
        self.assertTrue(y_out is not None)
        self.assertTrue(x_out[self.ref_string_idx] == self.ref_string)
        self.assertTrue(y_out[self.ref_string_idx] == self.ref_string_label)

    def test_questions_data_parser(self):
        """test the test questions data parser"""
        data_parser = SurveyDoc2VecDataParser()
        x_out, _ = data_parser.load(self.test_questions_file)
        self.assertTrue(x_out is not None)
        self.assertTrue(x_out[self.test_string_idx] == self.test_string)
  
if __name__ == '__main__':
    unittest.main()
