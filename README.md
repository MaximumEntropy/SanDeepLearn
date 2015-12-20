# SanDeepLearn

# MNIST Multilayer Perceptron

```
train, dev, test = pickle.load(gzip.open('data/mnist.pkl.gz', 'rb'))

train_x, train_y = train[0], train[1].astype(np.int32)
dev_x, dev_y = dev[0], dev[1].astype(np.int32)
test_x, test_y = test[0], test[1].astype(np.int32)

train_yy = np.zeros((train_y.shape[0], 10)).astype(np.int32)
dev_yy = np.zeros((dev_y.shape[0], 10)).astype(np.int32)
test_yy = np.zeros((test_y.shape[0], 10)).astype(np.int32)

for ind, val in enumerate(train_y):
	train_yy[ind][val] = 1

network = SequentialNetwork()
network.add(FullyConnectedLayer(train_x.shape[1], 500, activation='sigmoid'))
network.add(FullyConnectedLayer(500, 10, activation='sigmoid'))
network.add(SoftMaxLayer(hierarchical=False))

network.compile(loss='categorical_crossentropy')

network.train(train_x, train_yy, nb_epochs=500, valid_x=dev_x, valid_y=dev_y, test_x=test_x, test_y=test_y)
```