import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class SeqModel:


    def __init__(self):
        pass

    def create_model(self):
        input = tf.keras.Input(shape=(1,))
        output = tf.keras.layers.Dense(1)(input)

        model = tf.keras.Model(input,output)
        print(model.summary())
        """
       Total params: 2
       Trainable params: 2
       Non-trainable params: 0
       """

    def execute(self):
        (x, y) = self.make_random_data()
        x_train, y_train = x[:150], y[:150]
        x_test, y_test = x[:150], y[:150]
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(units=1, input_dim=1))
        self.model.summary()

        self.model.compile(optimizer = 'sgd', loss = 'mse')

        history = self.model.fit(x_train, y_train, epochs = 50, validation_split = 0.3)
        epochs = np.arange(1, 50+1)
        plt.plot(epochs, history.history['loss'], label = 'Training loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    @staticmethod
    def make_random_data():
        x = np.random.uniform(low = -2, high=2, size = 200)
        y = []
        for t in x:
            r = np.random.normal(loc = 0.0,
                                 scale=(0.5 + t*t/3),
                                 size = None)
            y.append(r)

        return x, 1.726*x - 0.84 + np.array(y)