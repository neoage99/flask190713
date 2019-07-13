import tensorflow_datasets as tfds

class Imdb:
    def __init__(self):
        pass

    def create_model(self):
        train_valication_split = tfds.Split.TRAIN.subsplit([6,4])

        (train_data, validation_data), test_data = tfds.load(
            name="imdb_reviews",
            split=(train_valication_split, tfds.Split.TEST),
            as_supervised=True
        )

        train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))

        print('10개의 샘플 %s'% (train_examples_batch))
        print('10개의 라벨 %s'% (train_labels_batch))
        