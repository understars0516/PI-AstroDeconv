import tensorflow as tf


class StopTrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, target_loss):
        super(StopTrainingCallback, self).__init__()
        self.target_loss = target_loss

    def on_epoch_end(self, epoch, logs=None):
        if logs.get('loss') <= self.target_loss:
            print(f"\nStopping training as loss {logs.get('loss')} reached target {self.target_loss}")
            self.model.stop_training = True
