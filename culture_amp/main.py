"""program main"""
import warnings
warnings.filterwarnings("ignore")
import sys, os
import numpy as np
from survey_data_parser import SurveyDoc2VecDataParser
from doc2vec_model import Doc2VecModel
from doc2vec_to_np_data_parser import Doc2VecToNumpyDataParser
from sgd_model import SGDModel
from gbc_model import GBClassModel
from sklearn.cross_validation import train_test_split
from survey_util import Util

def main_menu():
    """exec_menu"""
    os.system('cls')
    print("Welcome,\n")
    print("Please choose the menu you want to start:")
    print("1. Vectorise training data and train a new model")
    print("2. Test the model against a test file (default data/test_question.txt)")
    print("0. Quit")
    print("Or enter some text to return the most similar reference questions")
    choice = input(" >>")
    exec_menu(choice)

MODEL_DIR = "model_files/"
DOC2VEC_MODEL_FILE = MODEL_DIR + "/" + "DOC2VEC_MODEL.doc2vec"
DISCRIMINATOR_MODEL_FILE = "discriminator.bin"
SGD_FILE = MODEL_DIR+"/" + "sgd_" + DISCRIMINATOR_MODEL_FILE
GBC_FILE = MODEL_DIR+"/" + "gbc_" + DISCRIMINATOR_MODEL_FILE
N_TOP_CLASSIFICATIONS = 2
# REMOVE_PRONOUNS = True
STOP_WORDS_FLAG = True
STEMMING_FLAG = False

def train_action():
    """train_action method that triggers the model training"""
    print("Select the model. (Doc2Vec by default): ")
    print("1. Stochastic Gradient Descent")
    print("2. Gradient Boosting Classification")
    choice = input(" >>")
    model_name = None
    model_out = None
    if choice == '1':
        model_out = SGDModel()
        model_name = "sgd"
    elif choice == '2':
        model_out = GBClassModel()
        model_name = "gbc"

    print("Enter a filename (or hit enter to use data/labeled_data.csv):")
    choice = input(" >>")
    train_file = "data/labeled_data.csv"
    if choice != '':
        train_file = choice
    if os.path.exists(train_file) is False:
        print("File does not exist. Going back to the main menu")
        print("Hit the Enter key to continue")
        input(">>")
        MENU_ACTIONS['main_menu']()

    if os.path.exists(SGD_FILE):
        os.remove(SGD_FILE)
    if os.path.exists(GBC_FILE):
        os.remove(GBC_FILE)

    print("Building Doc2Vec vector representations of training data")
    data_parser1 = SurveyDoc2VecDataParser(remove_stopwords_flag=STOP_WORDS_FLAG, word_stemming_flag=STEMMING_FLAG)
    data_parser1.load(train_file)
    print("Converting training data to LabeledSentences")
    labeled_sentences = data_parser1.convert()
    print("Fitting Doc2Vec model")
    docvec_model_out = Doc2VecModel(epochs=1000, dim_size=400, window_size=10)
    docvec_model_out.fit(labeled_sentences)
    print("Saving Doc2Vec model")
    docvec_model_out.save(DOC2VEC_MODEL_FILE)
    if model_out is not None:
        print("Preparing for SGDClassifier model")
        doc2vec_model = docvec_model_out.model
        print("Converting from Doc2Vec vectors to numpy array")
        data_parser2 = Doc2VecToNumpyDataParser(doc2vec_model)
        x_out, y_out, y_dict, y_rev_dict = data_parser2.convert()
        print("x_out shape: ", x_out.shape)
        print("y_out shape: ", y_out.shape)
        print("y_dict: ", y_dict)
        print("y_reverse_dict: ", y_rev_dict)
        x_train, x_test, y_train, y_test = train_test_split(x_out, y_out, test_size=0.1)
        sgd_model = SGDModel()
        print("Fitting SGDClassifier model")
        sgd_model.fit([x_train, y_train])
        print("Testing against validation data")
        y_pred = sgd_model.predict(x_test)
        y_pred_max = Util.get_argmax_from_prob_array(y_pred)
        print("correctly predicted: ", (np.sum(y_pred_max == y_test)/y_test.shape[0]))
        top_classification = Util.get_np_dict_value_by_idx(y_rev_dict, y_pred, 2)
        print("top classification: ", top_classification)
        print("Saving model")
        sgd_model.save(MODEL_DIR+"/"+ model_name + "_" + DISCRIMINATOR_MODEL_FILE)
    print("Training complete. Hit the Enter key to continue")
    choice = input(" >>  ")
    MENU_ACTIONS['main_menu']()

def predict_sentence(sentence):
    """predict sentence"""
    model_out = None
    docvec_model_out = Doc2VecModel()
    # print("Loading Doc2Vec model")
    docvec_model_out.load(DOC2VEC_MODEL_FILE)
    # print("Generate vectors from sentence")
    x_test_sen = sentence
    if STOP_WORDS_FLAG:
        x_test_sen = Util.remove_stop_words(x_test_sen)
        x_test_sen = Util.remove_pronouns(x_test_sen)
    if STEMMING_FLAG:
        x_test_sen = Util.stem_words(x_test_sen)
    x_test, y_test = docvec_model_out.predict(x_test_sen.lower())
    # print("Loading SGDClassifier model")
    data_parser = Doc2VecToNumpyDataParser(docvec_model_out.model)
    _, _, _, y_reverse_dict = data_parser.convert()
    print("Sentence: ", sentence)

    if os.path.exists(SGD_FILE):
        model_out = SGDModel()
        model_out.load(SGD_FILE)
    elif os.path.exists(GBC_FILE):
        model_out = GBClassModel()
        model_out.load(GBC_FILE)

    if model_out is not None:       
        y_out = model_out.predict(x_test)
        top_classification = Util.get_np_dict_value_by_idx(y_reverse_dict, y_out, N_TOP_CLASSIFICATIONS)
        print("Top Classifications: ", top_classification)
        print("y_out: ", type(y_out))
        y_out_sorted = np.sort(y_out[0])[::-1]
        print("Probability Score: ", y_out_sorted[:2], "\n")
    else:
        print("Probability Score: ", Util.get_tuple_doc2vec_similar(y_test, N_TOP_CLASSIFICATIONS), "\n")
    return

def predict_test_file_action():
    """train_action"""
    print("Enter a filename (or hit enter to use data/test_questions.txt):")
    choice = input(" >>  ")
    test_file = "data/test_questions.txt"
    if choice != '':
        test_file = choice
    if os.path.exists(test_file) is False:
        print("File does not exist. Going back to the main menu")
        print("Hit the Enter key to continue")
        input(">>")
        MENU_ACTIONS['main_menu']()
    data_parser1 = SurveyDoc2VecDataParser()
    print("Loading test file")
    x_out, _ = data_parser1.load(test_file)
    for sentence in x_out:
        predict_sentence(sentence)
    print("Hit the Enter key to continue")
    input(">>")
    MENU_ACTIONS['main_menu']()

# Back to main menu
def back():
    """back"""
    MENU_ACTIONS['main_menu']()

# Exit program
def exit_program():
    """exit"""
    sys.exit()

# Menu definition
MENU_ACTIONS = {
    'main_menu': main_menu,
    '1': train_action,
    '2': predict_test_file_action,
    '0': exit_program
}

# Execute menu
def exec_menu(choice):
    """exec_menu"""
    os.system('cls')
    choice = choice.lower()
    if choice == '':
        MENU_ACTIONS['main_menu']()
    elif choice.isdigit():
        try:
            MENU_ACTIONS[choice]()
        except KeyError:
            print("Invalid selection, please try again.\n")
            MENU_ACTIONS['main_menu']()
    else:
        predict_sentence(choice)
        print("Hit the Enter key to continue")
        input(">>")
        MENU_ACTIONS['main_menu']()
    return

# Main Program
if __name__ == "__main__":
    main_menu()
