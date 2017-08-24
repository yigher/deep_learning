"""survey_data_parser script containing survey data parser"""
import csv
from abstract_data_parser import AbstractDataParser
from survey_util import Util
from gensim.models.doc2vec import TaggedDocument

class SurveyDoc2VecDataParser(AbstractDataParser):
    """AbstractDataParser for training/testing data"""
    def __init__(
            self,
            remove_stopwords_flag=False,
            word_stemming_flag=False):
        self.x_out = []
        self.y_out = []
        self.labeled_sentences = []
        self.remove_stopwords_flag = remove_stopwords_flag
        self.word_stemming_flag = word_stemming_flag

    def load(self, input_source):
        """load the data from the input_source
        input_source = file_location
        returns
            x_out = array of training data
            y_out = array of training labels"""
        with open(input_source, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile, quotechar='|', delimiter='|')
            row_count = 0
            col_count = None
            for row in reader:
                # obtain the expected column count
                if col_count is None:
                    col_count = len(row)
                # skip header
                if row_count == 0 and col_count > 1:
                    row_count += 1
                    continue
                # if there is text in the 1st and 2nd column, append to x_out and y_out
                # else, ignore the line
                if len(row) == col_count:
                    self.x_out.append(row[0].replace("\"", "").lower())
                    if col_count > 1:
                        self.y_out.append(Util.clean_label(row[1]))
                row_count += 1
        return self.x_out, self.y_out

    def convert(self):
        """Convert the input_source
        returns
            labeled_sentences = list(TaggedDocument)"""
        labeled_sentences = []
        if len(self.y_out) <= 0:
            return labeled_sentences
        if len(self.x_out) != len(self.y_out):
            raise Exception("sentence and labelled lists are of not equal length")
        for i in range(len(self.x_out)):
            input_text = self.x_out[i]
            if self.remove_stopwords_flag:
                # remove stop words
                input_text = Util.remove_stop_words(input_text)
                input_text = Util.remove_pronouns(input_text)
            # stem words
            if self.word_stemming_flag:
                # stem words
                input_text = Util.stem_words(input_text)
            sentence_out = input_text.split()

            self.labeled_sentences.append(
                TaggedDocument(words=sentence_out, tags=[self.y_out[i]+"_%s" % i])
            )
        return self.labeled_sentences

    def save(self, output, data):
        """Save the data object to the output."""
        return

if __name__ == "__main__":
    print('Subclass:', issubclass(SurveyDoc2VecDataParser, AbstractDataParser))
    print('Instance:', isinstance(SurveyDoc2VecDataParser(), AbstractDataParser))
    FILE_DIR = "data/"
    LABELLED_FILE = FILE_DIR + "labeled_data" + ".csv"
    FILE_PARSER = SurveyDoc2VecDataParser()
    X, Y = FILE_PARSER.load(LABELLED_FILE)
    print("X: ", len(X), " Y: ", len(Y))
    LABELED_SENTENCES = FILE_PARSER.convert()
    print(LABELED_SENTENCES)
