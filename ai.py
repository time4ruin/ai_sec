import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import csv

class RNN:
    def __init__(self):
        self.model = tf.keras.Sequential(self._build_layers())
        self.model.compile(optimizer='Adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.summary()

    def _build_layers(self):
        layers = [
            tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(None, 5)),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(5, activation='softmax')
        ]
        return layers

    def fit(self, x, t, val_x, val_t, n_epoch, n_batch):
        return self.model.fit(x, t, validation_data=(val_x, val_t), epochs=n_epoch, batch_size=n_batch)

    def evaluate(self, x, t):
        y = self.model(x, training=False)
        return tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(t, y))

    def save(self):
        self.model.save('saved_model/my_model')

    def load(self):
        self.model = tf.keras.models.load_model('saved_model/my_model')

def read_csv(attack_name):
    for i in range(1, 1001):
        f = open('./result/' + attack_name + str(i) + '.csv', 'r', encoding='utf-8')
        rdr = csv.reader(f)
        for line in rdr:
            if line[0] == "Core":
                tmp = []
                Instructions = np.array([])
                Cycles = np.array([])
                L1MISS = np.array([])
                L1HIT = np.array([])
                L3MISS = np.array([])
            else :
                if len(tmp) == 0:
                    tmp = np.array([float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5])])
                    tmp = tmp.reshape((1,) + tmp.shape)
                else:
                    tmp2 = np.array([float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5])])
                    tmp2 = tmp2.reshape((1,) + tmp2.shape)
                    tmp = np.concatenate((tmp, tmp2), axis=0)
        # tmp = np.stack([Instructions, Cycles, L1MISS, L1HIT, L3MISS], axis=0)
        tmp = tmp.reshape((1,) + tmp.shape)
        f.close()
        if i == 1:
            X = tmp
        else:
            X = np.concatenate([X, tmp], axis=0)

    return X

def read_all():
    attacks = ['no', 'FR', 'FF', 'PP', 'Spectre']
    X = np.array([])
    t = np.array([])
    for i in range(0, len(attacks)):
        tmp = read_csv(attacks[i])
        tmp2 = np.full((1000), i)
        if X.size == 0:
            X = tmp
            t = tmp2
        else:
            X = np.concatenate([X, tmp], axis=0)
            t = np.concatenate([t, tmp2], axis=0)
        # print(X.shape, t.shape)
    return X[:,:49,:], t


def main():
    X, t = read_all()
    # print(X, t)
    tr_X = np.concatenate([X[:800], X[1000:1800], X[2000:2800], X[3000:3800], X[4000:4800]], axis=0)
    tr_t = np.concatenate([t[:800], t[1000:1800], t[2000:2800], t[3000:3800], t[4000:4800]], axis=0)
    val_X = np.concatenate([X[800:900], X[1800:1900], X[2800:2900], X[3800:3900], X[4800:4900]], axis=0)
    val_t = np.concatenate([t[800:900], t[1800:1900], t[2800:2900], t[3800:3900], t[4800:4900]], axis=0)

    te_X = np.concatenate([X[900:1000], X[1900:2000], X[2900:3000], X[3900:4000], X[4900:5000]], axis=0)
    te_t = np.concatenate([t[900:1000], t[1900:2000], t[2900:3000], t[3900:4000], t[4900:5000]], axis=0)
    # print(tr_X.shape, tr_t.shape, te_X.shape, te_t.shape)
    # print(tr_X, tr_t)
    
    scalers = {}
    for i in range(tr_X.shape[2]):
        scalers[i] = StandardScaler()
        tr_X[:, :, i] = scalers[i].fit_transform(tr_X[:, :, i])
        val_X[:, :, i] = scalers[i].transform(val_X[:, :, i])
        te_X[:, :, i] = scalers[i].transform(te_X[:, :, i])

    rnn = RNN()

    hist = rnn.fit(tr_X, tr_t, val_X, val_t, 10, 100)
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(hist.history['accuracy'], c='b', label='train acc')
    ax[0].plot(hist.history['val_accuracy'], c='r', label='val acc')
    ax[1].plot(hist.history['loss'], c='r', label='loss')
    ax[0].legend()
    ax[1].legend()
    plt.savefig('ai.png')
    fig.show()
    plt.show()

    # rnn.load()
    print("Evaluation Result(0-5s data)")
    print(rnn.evaluate(te_X[:,:49,:], te_t))
    print("Evaluation Result(1-5s data)")
    print(rnn.evaluate(te_X[:,9:49,:], te_t))
    print("Evaluation Result(2-5s data)")
    print(rnn.evaluate(te_X[:,19:49,:], te_t))
    print("Evaluation Result(0-4.5s data)")
    print(rnn.evaluate(te_X[:,:44,:], te_t))
    print("Evaluation Result(1-4.5s data)")
    print(rnn.evaluate(te_X[:,9:44,:], te_t))
    print("Evaluation Result(2-4.5s data)")
    print(rnn.evaluate(te_X[:,19:44,:], te_t))
    print("Evaluation Result(0-4s data)")
    print(rnn.evaluate(te_X[:,:39,:], te_t))
    print("Evaluation Result(1-4s data)")
    print(rnn.evaluate(te_X[:,9:39,:], te_t))
    print("Evaluation Result(2-4s data)")
    print(rnn.evaluate(te_X[:,19:39,:], te_t))
    # rnn.save()
    

if __name__ == '__main__':
    main()