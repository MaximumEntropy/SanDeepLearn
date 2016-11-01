"""RNN LM on Zaremba's PTB dataset. https://github.com/wojzaremba/lstm."""
import theano
import theano.tensor as T
import sys
import argparse
import numpy as np
import logging
import os

sys.path.append('/u/subramas/Research/SanDeepLearn/')

from recurrent import FastLSTM, LSTM, GRU
from optimizer import Optimizer
from layer import FullyConnectedLayer, EmbeddingLayer

theano.config.floatX = 'float32'

parser = argparse.ArgumentParser()
parser.add_argument(
    "-train",
    "--train_sentences",
    help="path to ptb train",
    required=True
)
parser.add_argument(
    "-dev",
    "--dev_sentences",
    help="path to ptb dev",
    required=True
)
parser.add_argument(
    "-test",
    "--test_sentences",
    help="path to ptb test",
    required=True
)
parser.add_argument(
    "-batch_size",
    "--batch_size",
    help="batch size",
    required=True
)
parser.add_argument(
    "-cell",
    "--cell_type",
    help="cell type - FastLSTM/LSTM/GRU",
    required=True
)
parser.add_argument(
    "-h",
    "--hidden_dim",
    help="hidden dimension",
    required=True
)
parser.add_argument(
    "-e",
    "--emb_dim",
    help="embedding dimension",
    required=True
)
parser.add_argument(
    "-d",
    "--depth",
    help="num rnn layers",
    required=True
)
parser.add_argument(
    "-exp",
    "--experiment_name",
    help="name of the experiment",
    required=True
)
parser.add_argument(
    "-seed",
    "--seed",
    help="seed for pseudo random number generator",
    default=1337
)

args = parser.parse_args()
data_path_train = args.train_sentences
data_path_dev = args.dev_sentences
data_path_test = args.test_sentences
np.random.seed(seed=int(args.seed))  # set seed for an experiment
experiment_name = args.experiment_name
batch_size = int(args.batch_size)
hidden_dim = int(args.hidden_dim)
embedding_dim = int(args.embedding_dim)
depth = int(args.depth)

if not os.path.exists('log/'):
    os.mkdir('log/')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='log/%s' % (experiment_name),
    filemode='w'
)

# define a new Handler to log to console as well
console = logging.StreamHandler()
# optional, set the logging level
console.setLevel(logging.INFO)
# set a format which is the same for console use
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)


def get_minibatch(lines, index, batch_size):
    """Prepare minibatch."""
    lines = [
        ['<s>'] + line[:40] + ['</s>']
        for line in lines[index: index + batch_size]
    ]

    lens = [len(line) for line in lines]
    max_len = max(lens)

    input_lines = np.array([
        [word2ind[w] for w in line[:-1]] +
        [word2ind['<pad>']] * (max_len - len(line))
        for line in lines
    ]).astype(np.int32)

    output_lines = np.array([
        [word2ind[w] for w in line[1:]] +
        [word2ind['<pad>']] * (max_len - len(line))
        for line in lines
    ]).astype(np.int32)

    mask = np.array(
        [
            ([1] * (l - 1)) + ([0] * (max_len - l))
            for l in lens
        ]
    ).astype(np.float32)

    return input_lines, output_lines, mask


cell_hash = {
    'FastLSTM': FastLSTM,
    'GRU': GRU,
    'LSTM': LSTM
}

rnn_cell = cell_hash[args.cell_type]

train_lines = [line.strip().split() for line in open(data_path_train, 'r')]
dev_lines = [line.strip().split() for line in open(data_path_dev, 'r')]
test_lines = [line.strip().split() for line in open(data_path_test, 'r')]

word2ind = {word: ind for ind, word in enumerate(line) for line in train_lines}
ind2word = {ind: word for ind, word in enumerate(line) for line in train_lines}

word2ind['<s>'] = len(word2ind)
word2ind['</s>'] = len(word2ind) + 1
ind2word[len(word2ind)] = '<s>'
ind2word[len(word2ind) + 1] = '</s>'

inp_t = np.random.randint(low=1, high=100, size=(5, 10)).astype(np.int32)
op_t = np.random.randint(low=1, high=100, size=(5, 10)).astype(np.int32)
mask_t = np.float32(np.random.rand(5, 9).astype(np.float32) > 0.5)

x = T.imatrix('X')
y = T.imatrix('Y')
mask = T.fmatrix('Mask')

embedding_layer = EmbeddingLayer(
    input_dim=len(word2ind),
    output_dim=embedding_dim,
    name='embedding_lookup'
)

rnn_layers = [
    rnn_cell(
        input_dim=embedding_dim,
        output_dim=hidden_dim,
        name='rnn_0'
    )
]

rnn_layers += [
    rnn_cell(
        input_dim=hidden_dim,
        output_dim=hidden_dim,
        name='src_rnn_forward_%d' % (i + 1)
    ) for i in xrange(depth - 1)
]

rnn_h_to_vocab = FullyConnectedLayer(
    input_dim=hidden_dim,
    output_dim=len(word2ind),
    batch_normalization=False,
    activation='softmax',
    name='lstm_h_to_vocab'
)

params = embedding_layer.params + rnn_h_to_vocab.params

for rnn in rnn_layers:
    params += rnn.params

logging.info('Model parameters ...')
logging.info('RNN Cell : %s ' % (args.cell_type))
logging.info('Embedding dim : %d ' % (embedding_dim))
logging.info('RNN Hidden Dim : %d ' % (hidden_dim))
logging.info('Batch size : %s ' % (batch_size))
logging.info('Depth : %s ' % (depth))

embeddings = embedding_layer.fprop(x)

rnn_hidden_states = embeddings
for rnn in rnn_layers:
    rnn.fprop(rnn_hidden_states)
    rnn_hidden_states = rnn.h.dimshuffle(1, 0, 2)

proj_layer_input = rnn_hidden_states.reshape(
    (x.shape[0] * x.shape[1], hidden_dim)
)
proj_output_rep = rnn_h_to_vocab.fprop(proj_layer_input)
final_output = proj_output_rep.reshape(
    (x.shape[0], x.shape[1], len(word2ind))
)
# Clip final out to avoid log problem in cross-entropy
final_output = T.clip(final_output, 1e-5, 1-1e-5)

# Compute cost
cost = - (T.log(final_output[
    T.arange(
        embeddings.shape[0]).dimshuffle(0, 'x').repeat(
            embeddings.shape[1],
            axis=1
        ).flatten(),
    T.arange(embeddings.shape[1]).dimshuffle('x', 0).repeat(
        embeddings.shape[0],
        axis=0
    ).flatten(),
    y.flatten()
]) * mask.flatten()).sum() / T.neq(mask, 0).sum()

logging.info('Computation Graph Node Shapes ...')
logging.info('embedding dim : %s ' % (
    embeddings.eval({x: inp_t}).shape,)
)
logging.info('encoder forward dim : %s' % (
    rnn_layers[-1].h.dimshuffle(1, 0, 2).eval({x: inp_t}).shape,)
)
logging.info('rnn_hidden_states : %s' % (rnn_hidden_states.eval(
    {x: inp_t}).shape,)
)
logging.info('proj_layer_input : %s' % (proj_layer_input.eval(
    {x: inp_t}).shape,)
)
logging.info('final_output : %s' % (final_output.eval(
    {x: inp_t}).shape,)
)
logging.info('cost : %.3f' % (cost.eval(
    {
        x: inp_t,
        y: op_t,
        mask: mask_t,
    }
)))
