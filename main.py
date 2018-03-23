"""Entry point."""
import os

import torch

import data
import config
import utils
import trainer
import importlib as imp
# logger = utils.get_logger()

def main(args):
    """main: Entry point."""
    utils.prepare_dirs(args)

    if args.num_gpu > 0:
        torch.cuda.manual_seed(args.random_seed)

    if args.network_type == 'rnn':
        pass
    elif args.network_type == 'cnn':
        dataset = data.image.Image(args, args.data_path)
    else:
        raise NotImplementedError(f"{args.dataset} is not supported")
    trnr = trainer.Trainer(args, dataset)

    if args.mode == 'train':
        utils.save_args(args)
        trnr.train()
    elif args.mode == 'derive':
        assert args.load_path != "", ("`--load_path` should be given in "
                                      "`derive` mode")
        trnr.derive()

    else:
        if not args.load_path:
            raise Exception("[!] You should specify `load_path` to load a"
                            "pretrained model")
        trnr.test()


if __name__ == "__main__":
    args, unparsed = config.get_args()
    main(args)
