import argparse
import pprint
import tensorflow as tf

from tools.data_loader import DataLoader
# from tools.trainer import SingleTrainer
# from tools.models import cls_model
# from tools.losses import mse_loss
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
        '--input_path'
    )
    p.add_argument(
        '--output_path'
    )
    config = p.parse_args()
    return config

def get_dataloader(input_path, is_distribute=False):
    data_loader = DataLoader(input_path)
    loader = data_loader.test_tfrecord()
    # if is_distribute:
    #     loader = data_loader.distri_loader()
    # else:
    #     loader = data_loader.single_loader()
    return loader

# def get_model():
#     model = cls_model()
#     return model

# def get_loss():
#     loss = mse_loss()
#     return loss

# def get_optimizer():
#     opti = adam_opti()
#     return opti

# def get_scheduler():
#     pass

def main(config):
    def print_config(config):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))
    print_config(config)

    loader = get_dataloader(config.input_path)
    print(loader)
if __name__=="__main__":
    config = define_argparser()
    main(config)