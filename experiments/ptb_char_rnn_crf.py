"""RNN LM on Zaremba's PTB dataset. https://github.com/wojzaremba/lstm."""
from __future__ import absolute_import
import theano
import theano.tensor as T
import sys
import argparse
import numpy as np
import logging
import os

sys.path.append('/u/subramas/Research/SanDeepLearn/')

from recurrent import FastLSTM, LSTM, GRU, FastGRU, MiLSTM, LNFastLSTM
from optimizers import Optimizer
from layer import FullyConnectedLayer, EmbeddingLayer
from crf import CRF

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
    "-o",
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
parser.add_argument(
    "-drp"
    "--dropout_rate",
    help="dropout rate",
    default=0.0
)

args = parser.parse_args()
data_path_train = args.train_sentences
data_path_dev = args.dev_sentences
data_path_test = args.test_sentences
np.random.seed(seed=int(args.seed))  # set seed for an experiment
experiment_name = args.experiment_name
batch_size = int(args.batch_size)
hidden_dim = int(args.hidden_dim)
embedding_dim = int(args.emb_dim)
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


def generate_samples(
    batch_input
):
    """Generate random samples."""
    decoded_batch = f_eval(
        batch_input
    )
    logging.info('Src : %s ' % (''.join([ind2word[x] if x in ind2word else '' for x in batch_input])))
    logging.info('Sample : %s ' % (''.join([ind2word[x] if x in ind2word else '' for x in decoded_batch])))


def get_perplexity(dataset='train'):
    """Compute perplexity on train/dev/test."""
    if dataset == 'dev':
        dataset = dev_lines
    elif dataset == 'test':
        dataset = test_lines

    perplexities = []

    for j in xrange(len(dataset)):
        inp, op = get_minibatch(
            dataset,
            j
        )

        decoded_batch_ce = f_ce(
            inp,
            op
        )
        perplexities.append(decoded_batch_ce)

    return np.exp(np.mean(perplexities))


def get_minibatch(lines, index):
    """Prepare minibatch."""
    line = list(lines[index]) + ['</s>']

    input_line = np.array(
        [word2ind[w] if w in word2ind else word2ind['<unk>'] for w in line[:-1]]
    ).astype(np.int32)

    output_line = np.array(
        [word2ind[w] if w in word2ind else word2ind['<unk>'] for w in line[1:]]
    ).astype(np.int32)

    return input_line, output_line


cell_hash = {
    'FastLSTM': FastLSTM,
    'GRU': GRU,
    'LSTM': LSTM,
    'FastGRU': FastGRU,
    'MiLSTM': MiLSTM,
    'LNFastLSTM': LNFastLSTM
}

rnn_cell = cell_hash[args.cell_type]

train_lines = [line.strip() for line in open(data_path_train, 'r')]
dev_lines = [line.strip() for line in open(data_path_dev, 'r')]
test_lines = [line.strip() for line in open(data_path_test, 'r')]

word2ind = {'<s>': 0, '</s>': 1, '<pad>': 2}
ind2word = {0: '<s>', 1: '</s>', 2: '<pad>'}
ind = 3

for line in train_lines:
    for word in line:
        if word not in word2ind:
            word2ind[word] = ind
            ind2word[ind] = word
            ind += 1

logging.info('Found %d words in vocabulary' % (len(word2ind)))

inp_t = np.random.randint(low=1, high=20, size=(5,)).astype(np.int32)
op_t = np.random.randint(low=1, high=20, size=(5)).astype(np.int32)

x = T.ivector()
y = T.ivector()

embedding_layer = EmbeddingLayer(
    input_dim=len(word2ind),
    output_dim=embedding_dim,
    name='embedding_lookup'
)

rnn_layers = [
    rnn_cell(
        input_dim=embedding_dim,
        output_dim=hidden_dim,
        name='rnn_0',
        batch_input=False
    )
]

rnn_layers += [
    rnn_cell(
        input_dim=hidden_dim,
        output_dim=hidden_dim,
        name='src_rnn_forward_%d' % (i + 1),
        batch_input=False
    ) for i in xrange(depth - 1)
]

rnn_h_to_vocab = FullyConnectedLayer(
    input_dim=hidden_dim,
    output_dim=len(word2ind),
    batch_normalization=False,
    activation='linear',
    name='lstm_h_to_vocab'
)

crf = CRF(len(word2ind))

params = embedding_layer.params + rnn_h_to_vocab.params
params += crf.params

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
    rnn_hidden_states = rnn.h

final_output = rnn_h_to_vocab.fprop(rnn_hidden_states)
final_output_softmax = T.nnet.softmax(T.clip(final_output, 1e-5, 1-1e-5))
ce = T.nnet.categorical_crossentropy(
    final_output_softmax,
    y
).mean()
'''
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
'''
cost = crf.fprop(final_output, y)
logging.info('Computation Graph Node Shapes ...')
logging.info('embedding dim : %s ' % (
    embeddings.eval({x: inp_t}).shape,)
)
logging.info('encoder forward dim : %s' % (
    rnn_layers[-1].h.eval({x: inp_t}).shape,)
)
logging.info('rnn_hidden_states : %s' % (rnn_hidden_states.eval(
    {x: inp_t}).shape,)
)
logging.info('final_output : %s' % (final_output.eval(
    {x: inp_t}).shape,)
)
logging.info('cost : %.3f' % (cost.eval(
    {
        x: inp_t,
        y: op_t,
    }
)))

logging.info('Compiling updates ...')
updates = Optimizer(clip=5.0).rmsprop(
    cost=cost,
    params=params,
)

logging.info('Compiling train function ...')
f_train = theano.function(
    inputs=[x, y],
    outputs=cost,
    updates=updates
)

logging.info('Compiling eval function ...')
f_eval = theano.function(
    inputs=[x],
    outputs=crf.fprop(input=final_output, ground_truth=None, viterbi=True, return_best_sequence=True, mode='eval'),
)

logging.info('Compiling cross entropy function ...')
f_ce = theano.function(
    inputs=[x, y],
    outputs=ce
)

num_epochs = 100
logging.info('Training network ...')
for i in range(num_epochs):
    costs = []
    np.random.shuffle(train_lines)
    for j in xrange(0, len(train_lines)):
        inp, op = get_minibatch(
            train_lines,
            j,
        )
        entropy = f_train(
            inp,
            op,
        )
        costs.append(entropy)
        logging.info('Epoch : %d Minibatch : %d Loss : %.3f' % (
            i,
            j,
            entropy
        ))
        if j % 10 == 0:
            generate_samples(inp)

    logging.info('Epoch : %d Average perplexity on Train is %.5f ' % (
        i,
        np.exp(np.mean(costs))
    ))
    dev_perplexities = get_perplexity(dataset='dev')
    logging.info('Epoch : %d Average perplexity on Dev is %.5f ' % (
        i,
        np.mean(dev_perplexities)
    ))
    test_perplexities = get_perplexity(dataset='test')
    logging.info('Epoch : %d Average perplexity on Test is %.5f ' % (
        i,
        np.mean(test_perplexities)
    ))

    print 'Epoch Loss : %.3f ' % (np.mean(costs))
