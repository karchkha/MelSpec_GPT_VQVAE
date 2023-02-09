import os
import sys
import time
import importlib
import argparse

import numpy as np

import torch
# from torch import nn, optim

# from data import MonoTextData
from transformer.Lit_GPT_VAE import GPT_VAE
from callbacks.GPT_VAE_callbacks import TextLogger, callbeck_of_my_dreams #, LitProgressBar

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl

# from exp_utils import create_exp_dir
from utils import uniform_initializer, xavier_normal_initializer, calc_iwnll, calc_mi, calc_au, sample_sentences, visualize_latent, reconstruct

# clip_grad = 5.0

ns=2


def init_config():
    parser = argparse.ArgumentParser(description='VAE mode collapse study')

    # model hyperparameters
    parser.add_argument('--dataset', type=str, required=True, help='dataset to use')
    parser.add_argument('--experiment', type=str, required=True, default="yahoo", help='experiment name')

    # optimization parameters
    parser.add_argument('--momentum', type=float, default=0, help='sgd momentum')
    parser.add_argument('--opt', type=str, choices=["sgd", "adam"], default="sgd", help='sgd momentum')
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--nsamples', type=int, default=1, help='number of iw samples for training')
    parser.add_argument('--iw_train_nsamples', type=int, default=-1)
    parser.add_argument('--iw_train_ns', type=int, default=1, help='number of iw samples for training in each batch')
    parser.add_argument('--iw_nsamples', type=int, default=500,
                         help='number of samples to compute importance weighted estimate')

    # select mode
    parser.add_argument('--train', type=int, default=False, help='start training process')
    parser.add_argument('--resume', type=str, default=None, help='resume_from the checkpoint')
    parser.add_argument('--workers', type=int, default=1, help='number of workers for data',)
    parser.add_argument('--eval', type=int, default=False, help='evaluate model')
    parser.add_argument('--test', type=int, default=False, help='test model')
    parser.add_argument('--logging_frequency', type=int, default=500, help='number of steps for text logging')
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--test_interpolation', type=int, default=False, help='test anc visualise an interpolation between 2 sounds')
    
    
    

    # decoding
    parser.add_argument('--reconstruct_from', type=str, default='', help="the model checkpoint path")
    parser.add_argument('--reconstruct_to', type=str, default="decoding.txt", help="save file")
    parser.add_argument('--decoding_strategy', type=str, choices=["greedy", "beam", "sample"], default="greedy")
    parser.add_argument('--reconstruct_spec', type=str, default='', help="model ckpt for mel-spectrograms reconstuction")
    parser.add_argument('--vocoder', type=str, default='', help="model ckpt for vocoder for audio reconstuction")

    # annealing paramters
    parser.add_argument('--warm_up', type=int, default=10, help="number of annealing epochs")
    parser.add_argument('--kl_start', type=float, default=1.0, help="starting KL weight")

    # inference parameters
    parser.add_argument('--seed', type=int, default=783435, metavar='S', help='random seed')

    # output directory

    parser.add_argument("--save_latent", type=int, default=0)

    # new
    parser.add_argument("--fix_var", type=float, default=-1)                          ### ???
    parser.add_argument("--freeze_epoch", type=int, default=-1)                       ### ???
    # parser.add_argument("--reset_dec", action="store_true", default=False)            ### ???
    parser.add_argument("--beta", type=float, default=1.0)                            #### this is deciding AE or VAE. beta = 0 means AE!
    
    
    parser.add_argument("--fb", type=int, default=0, help="0: no fb; 1: fb; 2: max(target_kl, kl) for each dimension")	
    parser.add_argument("--target_kl", type=float, default=-1, help="target kl of the free bits trick")

    args = parser.parse_args()

    # set args.cuda
    args.cuda = torch.cuda.is_available()

    # set seeds
    # seed_set = [783435, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    # args.seed = seed_set[args.taskid]
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    # load config file into args
    config_file = "config.config_GPT_VAE_%s" % args.dataset
    params = importlib.import_module(config_file).params
    args = argparse.Namespace(**vars(args), **params)
    
    # set args.label
    if 'label' in params:
        args.label = params['label']
    else:
        args.label = False

    return args


def main(args):

    if args.cuda:
        print('using cuda')

    #device = torch.device("cuda" if args.cuda else "cpu")
    device = "cuda" if args.cuda else "cpu"
    args.device = device
    print(args)

    ############################### model initialisation ######################                                              
    
    
    vae = GPT_VAE(args)


                ########## loading pre-trained encoder #############
    if args.load_path != "" and args.resume is None:
        # pl_module_ckpt = GPT_VAE.load_from_checkpoint(args.load_path, args = args)
        checkpoint = torch.load(args.load_path, map_location=torch.device('cpu'))
        encoder_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if "encoder" in k}
        try:
          vae.load_state_dict(encoder_state_dict, strict=False)
          print("loaded encoder from:", args.load_path)
        except RuntimeError as e:
          print(f'Error while loading the state dict: {e}')
        
        # vae.encoder = pl_module_ckpt.encoder
        del checkpoint, encoder_state_dict

    
    ##################################### CALLBACKS and TRAINER ###############
    
    callbeck_of_my_drm = callbeck_of_my_dreams()
    TtLogg = TextLogger(args)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step", log_momentum=False)
    logger = TensorBoardLogger(save_dir = "lightning_logs/"+ args.experiment  + "-" + args.dataset, name = 'TensorBoardLoggs') 
    checkpoint_callback = ModelCheckpoint(
                                        save_top_k = 0,
                                        monitor="loss",
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
                        callbacks=[checkpoint_callback, lr_monitor, callbeck_of_my_drm, TtLogg], # early_stopping], #RichProgressBar(leave=True)],
                        logger = logger,
                        num_sanity_val_steps=0,
                        devices=-1,
                        # gradient_clip_val=clip_grad,
                        # limit_train_batches = 2,
                        # limit_val_batches= 2,
                        #  limit_test_batches= 2,
                        #  log_every_n_steps=2,
                        #  fast_dev_run = True,
                        )
    
     ############################## training ##############################
    
    
    if args.train:
    
        trainer.fit(vae, 
                    ckpt_path = args.resume,
                    )



    #################################  evaluation ###########################
    

    if args.eval == 1:
    
    #   if args.resume !="":
    #     vae = trainer.resume_from_checkpoint(args.resume)
    #     # vae = VAE.load_from_checkpoint(ckpt_pathargs.resume, args= args)
      
      trainer.validate(model = vae, ckpt_path = args.resume)
    
    
    
     ########################## TESTING ############################
    
    if args.test== 1:
           
    
        vocab = vae.vocab
        vocab_size = len(vocab)
    
        test_data = MonoTextData(args.test_data, label=args.label, vocab=vocab)
    
    
        test_data_batch = test_data.create_data_batch(batch_size=1,
                                                      device=device,
                                                      batch_first=True)
    

        trainer.test(vae, ckpt_path = args.resume)
        torch.cuda.empty_cache()
    
        vae.to(device)
        with torch.no_grad():
            # calc_iwnll(vae, test_data_batch, args)
    
            nll, ppl = calc_iwnll(vae, test_data_batch, args)
            print('iw nll: %.4f, iw ppl: %.4f' % (nll, ppl))    
    
    
    ####################################### inference #####################################
    
    if args.reconstruct_from != "":
        print('begin decoding')
        vae.load_from_checkpoint(args.reconstruct_from, args = args)
        vae.eval().to(args.device)
        save_dir = "samples/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        path = ".".join(args.reconstruct_from.split("/")[-1].split(".")[:-1]) + \
                "_{}".format(args.decoding_strategy)
    
    
        vocab = vae.vocab
        vocab_size = len(vocab)
    
        test_data = MonoTextData(args.test_data, label=args.label, vocab=vocab)
    
    
        test_data_batch = test_data.create_data_batch(batch_size=args.batch_size,
                                                      device=device,
                                                      batch_first=True)
    
        
        with torch.no_grad():
            if args.decode_input != "":
                decode_data = MonoTextData(args.decode_input, vocab=vae.vocab)
    
                reconstruct(vae, decode_data, vocab, args.decoding_strategy, os.path.join(save_dir, path + ".rec")) #, args.device)
                print("saved output in", path)
            else:
                z = vae.sample_from_prior(100)
                
                # print(z)
                sample_from_prior(vae, z, args.decoding_strategy,
                    os.path.join(save_dir, path + ".sample"))

            if args.reconstruct_to != "":
                
                # test(vae, test_data_batch, "TEST", args)
                reconstruct(vae, test_data_batch, vocab, args.decoding_strategy, args.reconstruct_to)



if __name__ == '__main__':
    args = init_config()
    main(args)



