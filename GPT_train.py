import pdb
import os
import sys
import time
import importlib
import argparse

import numpy as np

import torch
# from torch import nn, optim

from transformer.minGPT import Lit_minGPT
from callbacks.GPT_callbacks import ImageLogger

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl

# from exp_utils import create_exp_dir
# from utils import uniform_initializer, xavier_normal_initializer, calc_iwnll, calc_mi, calc_au, sample_sentences, visualize_latent, reconstruct


def init_config():
    parser = argparse.ArgumentParser(description='GPT transformer for VQVAE_spec')

    # model hyperparameters
    parser.add_argument('--dataset', type=str, required=True, help='dataset to use')
    parser.add_argument('--experiment', type=str, required=True, default="yahoo", help='experiment name')

    # select mode
    parser.add_argument('--train', type=int, default=False, help='start training process')
    parser.add_argument('--resume', type=str, default=None, help='resume_from the checkpoint')
    parser.add_argument('--workers', type=int, default=1, help='number of workers for data',)
    parser.add_argument('--eval', type=int, default=False, help='evaluate model')
    parser.add_argument('--test', type=int, default=False, help='test model')
    parser.add_argument('--logging_frequency', type=int, default=200, help='number of steps for text logging')
    # parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--test_interpolation', type=int, default=False, help='test anc visualise an interpolation between 2 sounds')
    
    
    # decoding
    # parser.add_argument('--reconstruct_from', type=str, default='', help="the model checkpoint path")
    # parser.add_argument('--reconstruct_to', type=str, default="decoding.txt", help="save file")
    # parser.add_argument('--decoding_strategy', type=str, choices=["greedy", "beam", "sample"], default="greedy")
    parser.add_argument('--reconstruct_spec', type=str, default='', help="model ckpt for mel-spectrograms reconstuction")
    parser.add_argument('--vocoder', type=str, default='', help="model ckpt for vocoder for audio reconstuction")


    args = parser.parse_args()

    # set args.cuda
    args.cuda = torch.cuda.is_available()

    args.seed = 783435
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    # load config file into args
    config_file = "config.config_GPT_%s" % args.dataset
    params = importlib.import_module(config_file).params
    args = argparse.Namespace(**vars(args), **params)
    
    return args


def main(args):

    if args.cuda:
        print('using cuda')

    #device = torch.device("cuda" if args.cuda else "cpu")
    device = "cuda" if args.cuda else "cpu"
    args.device = device
    print(args)

    ############################### model initialisation ######################                                              
    
    
    gpt = Lit_minGPT(args)



    ##################################### CALLBACKS and TRAINER ###############
    
    img_logger = ImageLogger(args = args)
    
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step", log_momentum=False)
    logger = TensorBoardLogger(save_dir = "lightning_logs/"+ args.experiment  + "-" + args.dataset, name = 'TensorBoardLoggs') #, flush_secs = 10) 
    checkpoint_callback = ModelCheckpoint(save_top_k = 1,
                                          monitor="val/loss",
                                          mode="min",
                                          save_last= True,
                                          dirpath= "lightning_logs/" + args.experiment + "-" + args.dataset + "/checkpoints/version_" + str(logger.version),
                                          filename= args.dataset + "-model-{epoch:02d}-{loss:.2f}",
                                          )
                                          
    early_stopping = EarlyStopping('loss', patience = 10)
    
    # bar = LitProgressBar()
    
    trainer = pl.Trainer(default_root_dir= "lightning_logs", 
                        accelerator="cuda" if args.cuda else "cpu" , #'gpu',
                        max_epochs= args.epochs,
                        callbacks=[img_logger, checkpoint_callback, lr_monitor], #callbeck_of_my_drm, TtLogg], # early_stopping], #RichProgressBar(leave=True)],
                        logger = logger,
                        num_sanity_val_steps=0,
                        # devices=-1,
                        # gradient_clip_val=clip_grad,
                        # limit_train_batches = 10,
                        # limit_val_batches= 10,
                        #  limit_test_batches= 2,
                        #  log_every_n_steps=2,
                        #  fast_dev_run = True,
                        )
    
     ############################## training ##############################
    
    
    if args.train:
    
        trainer.fit(gpt, 
                    ckpt_path = args.resume,
                    )



    #################################  evaluation ###########################
    

    if args.eval == 1:
    
    #   if args.resume !="":
    #     gpt = trainer.resume_from_checkpoint(args.resume)
    #     # gpt = VAE.load_from_checkpoint(ckpt_pathargs.resume, args= args)
      
      trainer.validate(model = gpt, ckpt_path = args.resume)
    

    
     ########################## TESTING ############################
    
    if args.test== 1:
           
    
        vocab = gpt.vocab
        vocab_size = len(vocab)
    
        trainer.test(gpt, ckpt_path = args.resume)
        torch.cuda.empty_cache()
    
        gpt.to(device)

    
    
    ####################################### inference #####################################
    
    # if args.reconstruct_from != "":
    #     print('begin decoding')
    #     gpt.load_from_checkpoint(args.reconstruct_from, args = args)
    #     gpt.eval().to(args.device)
    #     save_dir = "samples/"
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #     path = ".".join(args.reconstruct_from.split("/")[-1].split(".")[:-1]) + \
    #             "_{}".format(args.decoding_strategy)
    #     vocab = gpt.vocab
    #     vocab_size = len(vocab)
    





if __name__ == '__main__':
    args = init_config()
    main(args)



