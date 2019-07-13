import tensorflow as tf
class Add:
    def __init__(self):
        a = tf.constant(1)
        b = tf.constant(2)
        c = a + b
        print('텐서플로2 덧셈: a + b = %d'%(c))