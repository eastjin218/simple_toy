import argparse
import pprint
import tensorflow as tf

from tools.data_loader import DataLoader
from tools.trainer import SingleTrainer
from tools.models import ClsModel
from tools.losses import CustomLoss
# from tools.optimizers import adam_opti

def define_argparser(is_continue=False):
    p = argparse.ArgumentParser()

    if is_continue:
        p.add_argument(
            '--load_fn',
            required=True,
        )
    
    # p.add_argument(
    #     '--model_fn',
    #     required=not is_continue,
    # )
    p.add_argument(
        '--input_path',
        help='./cls-tfrecords'
    )
    p.add_argument(
        '--output_path'
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
        '--batch_size',
        default=8
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
        distri_loader = strategey.experimental_distribute_dataset(loader)
        return distri_loader
    return loader

def get_model(config):
    model = ClsModel(input_dim=config.input_dim, output_dim=config.output_dim)
    raw_input = (config.input_dim, config.input_dim, 3)
    model.build_graph(raw_input).summary()
    return model

def get_loss(config, strategy=None):
    if config.custom_loss:
        loss = CustomLoss(strategy=strategy)
    elif strategy:
        with
    else:
        loss = tf.keras.losses.CategoricalCrossentropy()
    return loss

# def get_optimizer(config):
#     if config.custom_opti:
#         opti = CostomOpti()
#     else:
#         opti = tf.keras.optimizers.Adam()
#     return opti

# def get_scheduler():
#     pass

def main(config):
    def print_config(config):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))
    print_config(config)
    strategy = None
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
    # opti = get_optimizer(config)
    trainer = SingleTrainer(
        dataset = loader,
        model = model,
        loss = loss,
        optimizer = opti,
        strategey = strategy
    )
    trainer.run()

    a = next(iter(loader))
    for i in a:
        model(i)
        break
if __name__=="__main__":
    config = define_argparser()
    main(config)