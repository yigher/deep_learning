import os
import sys
import json
import numpy as np
import matplotlib
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
file_name = os.path.basename(sys.argv[0])

def main(we_file='word_embeddings.npy', w2i_file='corpus_word2idx.json', Model=PCA):
    We = np.load(we_file)
    V, D = We.shape
    with open(w2i_file) as f:
        word2idx = json.load(f)
    idx2word = {v:k for k, v in word2idx.items()}

    model = Model()
    Z = model.fit_transform(We)
    plt.scatter(Z[:, 0], Z[:, 1])
    for i in range(V):
        plt.annotate(s=idx2word[i], xy=(Z[i, 0], Z[i, 1]))
    plt.show()

if __name__ == '__main__':
    main(we_file='gru_nonorm_part1_word_embeddings.npy', w2i_file='gru_nonorm_part1_wikipedia_word2idx.json', Model=TSNE)
