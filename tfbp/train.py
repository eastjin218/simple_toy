import argparse, os
import pprint
import tensorflow as tf

from tools.data_loader import DataLoader
from tools.trainer import SingleTrainer, DistriTrainer
from tools.models import ClsModel, PretrainedClsModel
from tools.losses import CustomLoss
# from tools.optimizers import adam_opti

def define_argparser(is_continue=False):
    p = argparse.ArgumentParser()

    p.add_argument(
        '--load_fn',
        default=None,
        help='./models/cls_model.h5py'
    )
    p.add_argument(
        '--input_path',
        help='./cls-tfrecords'
    )
    p.add_argument(
        '--output_path',
        help='./models'
    )
    p.add_argument(
        '--model_fn',
        default=None
    )
    p.add_argument(
        '--is_distribute',
        action='store_true'
    )
    p.add_argument(
        '--custom_loss',
        action='store_true'
    )
    p.add_argument(
        '--custom_opti',
        action='store_true'
    )
    p.add_argument(
        '--batch_size',
        default=8
    )
    p.add_argument(
        '--epochs',
        default=3
    )
    p.add_argument(
        '--input_dim',
        default=224
    )
    p.add_argument(
        '--output_dim',
        default=6
    )
    config = p.parse_args()
    return config

def get_dataloader(input_path, batch_size, is_distribute=False, strategy=None):
    data_loader = DataLoader(input_path, batch_size)
    loader = data_loader.cls_tfrecord()
    if is_distribute:
        distri_loader = strategy.experimental_distribute_dataset(loader)
        return distri_loader
    return loader

def get_model(config):
    if config.load_fn:
        model = tf.keras.models.load_model(config.load_fn)
        print(f'load model done!! {os.path.basename(config.load_fn)}')
    else:
        model = PretrainedClsModel(input_dim=config.input_dim, output_dim=config.output_dim)
        raw_input = (config.input_dim, config.input_dim, 3)
        model.build_graph(raw_input).summary()
    return model

def get_loss(config, strategy=None):
    if strategy:
        if config.custom_loss:
            loss = CustomLoss(strategy=strategy)
        else:
            loss = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    else:
        if config.custom_loss:
            loss = CustomLoss(strategy=strategy)
        else:
            loss = tf.keras.losses.SparseCategoricalCrossentropy()
    return loss

def get_optimizer(config):
    if config.custom_opti:
        # opti = CostomOpti()
        pass
    else:
        opti = tf.keras.optimizers.Adam()
    return opti

# def get_scheduler():
#     pass

def get_trainer(dataloader, model, loss, optimizer, strategy, 
                epochs, batch_size, global_step, scheduler=None,):
    if strategy:
        trainer = DistriTrainer(dataloader, model, loss, optimizer, scheduler,
                                strategy, epochs, batch_size, global_step)
    else:
        trainer = SingleTrainer(dataloader, model, loss, optimizer, scheduler,
                                epochs, batch_size, global_step)
    return trainer

def main(config):
    def print_config(config):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))
    print_config(config)
    strategy = None
    global_step=0
    if config.is_distribute:
        strategy = tf.distribute.MirroredStrategy()
    loader = get_dataloader(config.input_path,
                            config.batch_size,
                            config.is_distribute,
                            strategy,
                            ) # img[22,224,3]float32 , label[]int64
    # print(next(iter(loader))) # dataloader test code
    model = get_model(config)
    loss = get_loss(config, strategy)
    opti = get_optimizer(config)
    trainer = get_trainer(
        dataloader = loader,
        model = model,
        loss = loss,
        optimizer = opti,
        strategy=strategy,
        epochs = config.epochs,
        batch_size=config.batch_size,
        global_step=global_step,
    )
    trainer.run()
    trainer.save(config.output_path, config.model_fn)
    # count = 0
    # for i in loader:
    #     print(i)
    #     count +=1
    # print(count)
        
if __name__=="__main__":
    # single > python train.py --input_path ./dataset/cls-tfrecords --output_path ./models/
    # single continue > python train.py --input_path ./dataset/cls-tfrecords --load_fn ./models/cls_model.h5py --output_path ./models
    config = define_argparser()
    main(config)