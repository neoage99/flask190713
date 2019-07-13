from sequential_model.model import SeqModel
import matplotlib.pyplot as plt

if __name__ == '__main__':
    seq = SeqModel()
    seq.create_model()
    arr = seq.compile_train()

    epochs = 500
    plt.plot(epochs, arr[2].history['loss'], label='Training Loss')
    plt.plot(epochs, arr[2].history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
