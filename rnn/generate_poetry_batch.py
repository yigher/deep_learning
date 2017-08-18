from sklearn.utils import shuffle
from util import init_weight, get_robert_frost
import numpy as np

def generate_poetry_batches(epochs=500):
    """generate_poetry_batches"""
    sentences, word2idx = get_robert_frost()
    idx2word = {v:k for k, v in word2idx.items()}
    # total number of sentences
    n_sentences = len(sentences)
    # total number of words in corpus
    n_total = sum((len(sentence) + 1) for sentence in sentences)
    for i in range(epochs):
        for j in range(n_sentences):
            if np.random.random() < 0.1:
                print("generating an END to START")
                input_sequence = [0] + sentences[j]
                # [1] is the end token
                output_sequence = sentences[j] + [1]
            else:
                # input sequence is from the start to 2nd last word of X
                # so that the last word can be predicted
                input_sequence = [0] + sentences[j][:-1]
                output_sequence = sentences[j]
            input_seq_sentence = ''
            output_seq_sentence = ''

            for word_idx in input_sequence:
                input_seq_sentence += idx2word[word_idx] + " "
            for word_idx in output_sequence:
                output_seq_sentence += idx2word[word_idx] + " "

            print("input_seq_sentence: ", input_seq_sentence)
            print("output_seq_sentence: ", output_seq_sentence)
            print("n_total: ", n_total)
            # has to be calculated manually
            n_total += len(output_sequence)
            keypressed = input('Press q to quit: ')
            if keypressed == 'q':
                break
        keypressed = input('Press q to quit: ')
        if keypressed == 'q':
            break

if __name__ == "__main__":
    generate_poetry_batches()
