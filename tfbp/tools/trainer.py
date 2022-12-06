import os
import tensorflow as tf

class SingleTrainer():
    def __init__(self, dataloader, model, loss, optimizer, scheduler,
                epochs,batch_size, global_step):
        self.dataloader = dataloader
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.batch_size = batch_size
        self.global_step = global_step
        self.train_acc_metric = tf.keras.metrics.SparseCategoricalCrossentropy()

    @tf.function
    def train_step(self, input_data):
        x, y = input_data
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss_value = self.loss(y, logits)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.train_acc_metric.update_state(y, logits)
        return loss_value

    def run(self):
        for epoch in range(self.epochs):
            print("\nStart of epoch %d"%(epoch,))
            for step, input_data in enumerate(self.dataloader):
                loss = self.train_step(input_data)
            train_acc = self.train_acc_metric.result()
            print("Training acc over epoch: %.4f" % (float(train_acc),))
            self.train_acc_metric.reset_states()

    def save(self, output_path, model_name=None):
        os.makedirs(output_path, exist_ok=True)
        if model_name:
            save_path = os.path.join(output_path, model_name+'.h5py')
        else:
            save_path = os.path.join(output_path, 'cls_model.h5py')
        self.model.save(save_path,)

class DistriTrainer():
    def __init__(self,dataloader, model, loss, optimizer, scheduler, strategy,
                epochs, batch_size, global_step):
        self.dataloader = dataloader
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.strategy = strategy
        self.epochs = epochs
        self.batch_size = batch_size
        self.global_step = global_step

    # @tf.function
    def train_step(self, input_data):
        def step_fn(inputs):
            x, y = input_data
            with tf.GradientTape() as tape:
                logits = self.model(x, training=True)
                loss_value = tf.reduce_sum(self.loss(y, logits)) * (1.0 / global_batch_size)
            grads = tape.gradient(loss_value,self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
            return self.loss(y, logits)

        per_example_losses = mirrored_strategy.run(
            step_fn, args=(input_data,))
        mean_loss = mirrored_strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_example_losses, axis=0)
        return mean_loss

    def run(self):
        for epoch in range(self.epochs):
            print("\nStart of epoch %d"%(epoch,))
            for step, input_data in enumerate(dataloader):
                loss = self.train_step(input_data)
            train_acc = self.train_acc_metric.result()
            self.train_acc_metric.reset_states()