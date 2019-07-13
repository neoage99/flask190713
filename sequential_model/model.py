import tensorflow as tf
import numpy as np

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

    def compile_train(self):
        self.model.compile(optimizer = 'sgd', loss = 'mse')
        (x, y) = self.make_random_data()
        history = self.model.fit(x, y, epochs = 500, validation_split = 0.3)
        return [x,y,history]

    @staticmethod
    def make_random_data():
        x = np.random.uniform(low=-2, high=2, size=200)
        y = []
        for t in x:
            r = np.random.normal(loc=0.0,
                                 scale=(0.5 + t*t/3),
                                 size= None)
            y.append(r)
        history = x, 1.726*x - 0.84 + np.array(y)
        arr = [x, y, history]
        return arr