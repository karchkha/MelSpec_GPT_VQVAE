import pdb

import math
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import optim
from torch.utils.data import DataLoader, Dataset
from modules import GaussianLSTMEncoder, LSTMDecoder

from transformer import GPTEncoder, GPTDecoder

# from data import MonoTextData
from datasets.vas import DataModule, VocabEntry


from modules.utils import log_sum_exp
import gc



class GPT_VAE(pl.LightningModule):
    """VAE with normal prior"""
    def __init__(self, args):
        super(GPT_VAE, self).__init__()
        self.args = args
        self.len_train_data = 0
        # self.vocab = VocabEntry()
        # self.vocab_size = len(self.vocab)
        self.len_val_data = 0
        
        self.datamodule_loader()
        self.train_data = self.train_dataloader()
        self.val_data = self.val_dataloader()
        self.test_data = self.test_dataloader()

        print('Train data: %d samples' % self.len_train_data)
        print('finish reading datasets, vocab size is %d' % self.args.vocab_size)

        self.encoder = GPTEncoder(args, n_unmasked = self.args.block_size, last_linear = self.args.n_embd*2)
        self.decoder = GPTDecoder(args, embd_pdrop=args.embd_pdrop, resid_pdrop=args.resid_pdrop, attn_pdrop=args.attn_pdrop, block_size = self.args.block_size+1)

        loc = torch.zeros(self.args.n_embd, device=args.device)
        scale = torch.ones(self.args.n_embd, device=args.device)

        self.prior = torch.distributions.normal.Normal(loc, scale)

        
        #### helper variables
        
        # self.lr = self.args.lr  
        
        self.ns=2
        
        self.test_loss = 0.0
        self.ppl = 0.0
        self.nll = 0.0
        self.kl_loss = 0.0  
        self.rec_loss = 0.0
        

        # These variables we should save and load with checkpoints I think
        self.best_loss = 1e4
        self.pre_mi = 0
        self.kl_weight = self.args.kl_start
        
        # kl-weight control 
        if self.args.warm_up > 0:
            self.anneal_rate = (1.0 - self.args.kl_start) / (self.args.warm_up * (self.len_train_data / self.args.batch_size)) # used for Kl_weight control
        else:
            self.anneal_rate = 0
            
        self.dim_target_kl = self.args.target_kl / float(self.args.n_embd)
        
        # # optimisers set up
        # if self.args.opt == "sgd":
        #     self.optimizer = optim.SGD(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr, momentum=self.args.momentum)
            
        # elif self.args.opt == "adam":
        #     self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=1e-6)
        #     self.lr = 1e-6
        # else:
        #     raise ValueError("optimizer not supported")
        
        self.optimizer = self.GPT_configure_optimizers()
        
        self.forward_shuffle_idx, self.backward_shuffle_idx  = self.make_idx (5, 53) ### for VQ code reading in coeerect order 

    def encode(self, x, nsamples=1):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]
        """
        return self.encoder.encode(x, nsamples)

    def encode_stats(self, x):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the mean of latent z with shape [batch, nz]
            Tensor2: the logvar of latent z with shape [batch, nz]
        """

        return self.encoder.encode_stats(x)

    def decode(self, z, strategy, top_k = None, temperature = None):
        """generate samples from z given strategy

        Args:
            z: [batch, nsamples, nz]
            strategy: "beam" or "greedy" or "sample"
            K: the beam width parameter

        Returns: List1
            List1: a list of decoded word sequence
        """
        is_taining = self.training
        if is_taining:
          self.eval()
        
        z_start_indices = torch.empty((z.size(0),0), dtype=torch.int64).to(self.args.device)

        if strategy == "beam":
            
            # pdb.set_trace()
            index_sampled, att_dec = self.decoder.sample(z_start_indices, z,
                                                steps=self.args.block_size,
                                                temperature=temperature if temperature is not None else 1.0,
                                                sample=True,
                                                top_k=top_k if top_k is not None else 100)

        elif strategy == "greedy" or strategy == "sample":
            index_sampled, att_dec = self.decoder.sample(z_start_indices, z,
                                                steps=self.args.block_size,
                                                sample=False,
                                                )

        if is_taining:
          self.train()

        return index_sampled, att_dec
            

    def sample_from_inference(self, x, nsamples=1):
        """perform sampling from inference net
        Returns: Tensor
            Tensor: samples from infernece nets with
                shape (batch_size, nsamples, nz)
        """
        z, _, _ = self.encoder.sample(x, nsamples)

        return z


    def reconstruct(self, x, decoding_strategy="greedy", K=None):
        """reconstruct from input x

        Args:
            x: (batch, *)
            decoding_strategy: "beam" or "greedy" or "sample"
            K: the beam width parameter

        Returns: List1
            List1: a list of decoded word sequence
        """
        # z = self.sample_from_inference(x) #.squeeze(1)
        z, _, att_enc = self.encoder.sample(x, nsamples=1)
        
        rec, att_dec = self.decode(z, decoding_strategy, K)

        return rec, (att_enc, att_dec)


    def loss(self, x, kl_weight, nsamples=1):
        """
        Args:
            x: if the data is constant-length, x is the data tensor with
                shape (batch, *). Otherwise x is a tuple that contains
                the data tensor and length list

        Returns: Tensor1, Tensor2, Tensor3
            Tensor1: total loss [batch]
            Tensor2: reconstruction loss shape [batch]
            Tensor3: KL loss shape [batch]
        """

        z, KL = self.encode(x, nsamples)

        # (batch)
        reconstruct_err = self.decoder.reconstruct_error(x, z).mean(dim=1)

        
        return reconstruct_err + kl_weight * KL, reconstruct_err, KL


################################ VQ code reading in corrct order helper functions #############################

    def make_idx(self, H, W):
        idx = np.arange(H * W).reshape(H, W)
        idx = idx.T
        idx = torch.tensor(idx.ravel())
        return idx, torch.argsort(idx)
    
    
    def code_reader(self, x, reverse=False):
      B, L = x.shape
      L_idx = len(self.forward_shuffle_idx)
      if L > L_idx:
          # an ugly patch for "infinite" sampling because self.*_shuffle_idx are shorter
          # otherwise even uglier patch in other places. 'if' is triggered only on sampling.
          assert L % L_idx == 0 and L / L_idx == int(L / L_idx), f'L: {L}, L_idx: {L_idx}'
          W_scale = L // L_idx
          print(f'Permuter is making a guess on the temp scale: {W_scale}. Ignore on "infinite" sampling')
          idx, rev_idx = self.make_idx(self.H, self.W * W_scale)
          if not reverse:
              return x[:, idx]
          else:
              return x[:, rev_idx]
      else:
          if not reverse:
              return x[:, self.forward_shuffle_idx]
          else:
              return x[:, self.backward_shuffle_idx]    

################################################ GET INPUT ####################################################  

    def get_input(self, batch):
        x = batch['codes'].to(memory_format=torch.contiguous_format)

        #### 10 sec verion ###################
        x = x.permute(0, 2, 1)
        x = torch.flatten(x, start_dim=1)
        
        x = x.to(self.args.device) # just in case it it's not there yet

        # print(x[0], "\n")

        return x



################################################## TRAINING STEP ####################################################  

    def training_step(self, batch, batch_idx):
        
        x = self.get_input(batch)
        batch_data = x
        # print(batch_data)
        # print(batch_data.shape)
        
        if self.args.beta == 0:
            self.kl_weight = self.args.beta
        else:
            self.kl_weight = min(1.0, self.kl_weight + self.anneal_rate)
        
        
        #this is used for loss normalisation
        batch_size, sent_len = batch_data.size()
        report_num_words = (sent_len - 1) * batch_size
        report_num_sents = batch_size
        
        if self.args.beta == 0:
            if self.args.iw_train_nsamples < 0:
                loss, loss_rc, loss_kl = self.loss(batch_data, self.kl_weight, nsamples=self.args.nsamples)
            else:
                loss, loss_rc, loss_kl = self.loss_iw(batch_data, self.kl_weight, nsamples=self.args.iw_train_nsamples, ns=self.ns)
        else:
            if self.args.fb == 0:	
                loss, loss_rc, loss_kl = self.loss(batch_data, self.kl_weight, nsamples=self.args.nsamples)	
            elif self.args.fb == 1:	
                loss, loss_rc, loss_kl = self.loss(batch_data, self.kl_weight, nsamples=self.args.nsamples)	
                kl_mask = (loss_kl > self.args.target_kl).float()	
                loss = loss_rc + kl_mask * self.kl_weight * loss_kl 	
            elif self.args.fb == 2:	
                mu, logvar, _ = self.encoder(batch_data)	
                z = self.encoder.reparameterize(mu, logvar, self.args.nsamples)	
                loss_kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)	
                kl_mask = (loss_kl > self.dim_target_kl).float()	
                

                fake_loss_kl = (kl_mask * loss_kl).sum(dim=1)	
                loss_rc = self.decoder.reconstruct_error(batch_data, z).mean(dim=1)	
                loss = loss_rc + self.kl_weight * fake_loss_kl
                # pdb.set_trace()

            elif self.args.fb == 3:	
                loss, loss_rc, loss_kl = self.loss(batch_data, self.kl_weight, nsamples=self.args.nsamples)	
                kl_mask = (loss_kl.mean() > self.args.target_kl).float()	
                loss = loss_rc + kl_mask * self.kl_weight * loss_kl             
            
            
        loss = loss.mean(dim=-1)

        loss_rc = loss_rc.sum()
        loss_kl = loss_kl.sum()

        report_rec_loss = loss_rc.item() / report_num_sents
        report_kl_loss = loss_kl.item() / report_num_sents
        
        if self.args.beta == 0:
            report_loss = loss.item() * batch_size
        else:
            report_loss = loss_rc.item() + loss_kl.item()
        
        train_loss = report_loss / report_num_sents
        
        self.log("train/loss", train_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/loss_rc", report_rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/loss_kl", report_kl_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/kl_weight",  self.kl_weight, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
        return loss
        
        
        
################################################# VALIDATION STEP ####################################################  

    def validation_step(self, batch, batch_idx):

        x = self.get_input(batch)
        batch_data = x
        
        #this is used for loss normalisation
        batch_size, sent_len = batch_data.size()
        report_num_words = (sent_len - 1) * batch_size
        report_num_sents = batch_size
        
        if self.args.beta == 0:
            if self.args.iw_train_nsamples < 0:
                loss, loss_rc, loss_kl = self.loss(batch_data, self.kl_weight, nsamples=self.args.nsamples)
            else:
                loss, loss_rc, loss_kl = self.loss_iw(batch_data, self.kl_weight, nsamples=self.args.iw_train_nsamples, ns=self.ns)
        else:
            loss, loss_rc, loss_kl = self.loss(batch_data, 1.0, nsamples=self.args.nsamples)

        loss_rc = loss_rc.sum()
        loss_kl = loss_kl.sum()
        loss = loss.sum()
        
        report_rec_loss = loss_rc.item() / report_num_sents
        report_kl_loss = loss_kl.item() / report_num_sents
        
        if self.args.beta == 0:
            report_loss = loss.item() / report_num_sents
        else:
            if self.args.warm_up == 0 and self.args.kl_start < 1e-6:
                loss = loss_rc.item()
                report_loss = loss / report_num_sents	
            else:	
                report_loss = loss.item() / report_num_sents	

        
        self.log("loss", report_loss, prog_bar=False, logger=False, on_step=True, on_epoch=True) # this one is for chekpointing and earlistopping
        self.log("val/loss", report_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("val/loss_rc", report_rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("val/loss_kl", report_kl_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
        return {"val_loss": loss, "val_loss_rc": loss_rc, "val_loss_kl": loss_kl, "report_num_words": report_num_words, "report_num_sents": report_num_sents }
        
    def validation_epoch_end(self, validation_step_outputs):
        report_kl_loss = report_rec_loss = report_loss = 0
        report_num_words = report_num_sents = 0
        self.test_loss = 0.0
        self.nll = 0.0
        self.kl_loss = 0.0  
        self.rec_loss = 0.0
        self.ppl = 0.0

        for out in validation_step_outputs:
            report_rec_loss += out["val_loss_rc"]
            report_kl_loss += out["val_loss_kl"]
            report_loss += out["val_loss"]
            report_num_words += out["report_num_words"]
            report_num_sents += out["report_num_sents"]

        self.test_loss = report_loss / report_num_sents
        self.nll = (report_kl_loss + report_rec_loss) / report_num_sents
        self.kl_loss = report_kl_loss / report_num_sents
        self.rec_loss = report_rec_loss / report_num_sents
        self.ppl = torch.exp(self.nll * report_num_sents / report_num_words) #np.exp
        
        print("i'm in validation_epoch_endin model")

#################################  maybe this helps with memory problem #######################    
    
    # def on_train_epoch_end(self):
    #     gc.collect()


########################################### Mutual info and Active units calculation #################################################

    def calc_mi(self, test_data_batch):
        # calc_mi_v3

        mi = 0
        num_examples = 0
    
        mu_batch_list, logvar_batch_list = [], []
        neg_entropy = 0.
        for batch_data in test_data_batch:
            
            # print(batch_data["file_path_"])
            
            x = self.get_input(batch_data)
            batch_data = x
            
            mu, logvar, _ = self.encoder.forward(batch_data.to(self.args.device))
            x_batch, nz = mu.size()
            ##print(x_batch, end=' ')
            num_examples += x_batch
    
            # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)
            neg_entropy += (-0.5 * nz * math.log(2 * math.pi)- 0.5 * (1 + logvar).sum(-1)).sum().item()
            mu_batch_list += [mu.cpu()]
            logvar_batch_list += [logvar.cpu()]
    
        neg_entropy = neg_entropy / num_examples
        ##print()
    
        num_examples = 0
        log_qz = 0.
        for i in range(len(mu_batch_list)):
            ###############
            # get z_samples
            ###############
            mu, logvar = mu_batch_list[i].cuda(), logvar_batch_list[i].cuda()
            
            # [z_batch, 1, nz]
            if hasattr(self.encoder, 'reparameterize'):
                z_samples = self.encoder.reparameterize(mu, logvar, 1)
            else:
                z_samples = self.encoder.gaussian_enc.reparameterize(mu, logvar, 1)
            z_samples = z_samples.view(-1, 1, nz)
            num_examples += z_samples.size(0)
    
            ###############
            # compute density
            ###############
            # [1, x_batch, nz]
            #mu, logvar = mu_batch_list[i].cuda(), logvar_batch_list[i].cuda()
            #indices = list(np.random.choice(np.arange(len(mu_batch_list)), 10)) + [i]
            indices = np.arange(len(mu_batch_list))
            mu = torch.cat([mu_batch_list[_] for _ in indices], dim=0).cuda()
            logvar = torch.cat([logvar_batch_list[_] for _ in indices], dim=0).cuda()
            x_batch, nz = mu.size()
    
            mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
            var = logvar.exp()
    
            # (z_batch, x_batch, nz)
            dev = z_samples - mu
    
            # (z_batch, x_batch)
            log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
                0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))
    
            # log q(z): aggregate posterior
            # [z_batch]
            log_qz += (log_sum_exp(log_density, dim=1) - math.log(x_batch)).sum(-1)
    
        log_qz /= num_examples
        mi = neg_entropy - log_qz

        ########################## cleaning data 
        
        # mu = mu.cpu()
        # logvar = logvar.cpu()
        # x = x.cpu()
        # batch_data = batch_data.cpu()
        # z_samples = z_samples.cpu()
        # dev = dev.cpu()
        # log_density = log_density.cpu()
        
        # del mu_batch_list, logvar_batch_list
        # del mu, logvar, x, batch_data, z_samples, dev, log_density
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
    
        return mi
    
    
    def calc_au(self, test_data_batch, delta=0.01):
        """compute the number of active units
        """
        cnt = 0
        for batch_data in test_data_batch:
            x = self.get_input(batch_data)
            batch_data = x
            mean, _, _ = self.encode_stats(batch_data.to(self.args.device))
            if cnt == 0:
                means_sum = mean.sum(dim=0, keepdim=True)
            else:
                means_sum = means_sum + mean.sum(dim=0, keepdim=True)
            cnt += mean.size(0)
    
        # (1, nz)
        mean_mean = means_sum / cnt
    
        cnt = 0
        for batch_data in test_data_batch:
            x = self.get_input(batch_data)
            batch_data = x
            mean, _, _ = self.encode_stats(batch_data.to(self.args.device))
            if cnt == 0:
                var_sum = ((mean - mean_mean) ** 2).sum(dim=0)
            else:
                var_sum = var_sum + ((mean - mean_mean) ** 2).sum(dim=0)
            cnt += mean.size(0)
    
        # (nz)
        au_var = var_sum / (cnt - 1)
        
        
        ########################## cleaning data 
        
        
        # batch_data = batch_data.cpu()
        # del means_sum, mean_mean, var_sum, mean, batch_data
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        
        return (au_var >= delta).sum().item(), au_var


########################################### Test step #################################################

    def test_step(self, batch, batch_idx):

        x = self.get_input(batch)
        batch_data = x
        
        #this is used for loss normalisation
        batch_size, sent_len = batch_data.size()
        report_num_words = (sent_len - 1) * batch_size
        report_num_sents = batch_size
        
        if self.args.beta == 0:
            if self.args.iw_train_nsamples < 0:
                loss, loss_rc, loss_kl = self.loss(batch_data, self.kl_weight, nsamples=self.args.nsamples)
            else:
                loss, loss_rc, loss_kl = self.loss_iw(batch_data, self.kl_weight, nsamples=self.args.iw_train_nsamples, ns=self.ns)
        else:
            loss, loss_rc, loss_kl = self.loss(batch_data, 1.0, nsamples=self.args.nsamples)
            
            
        loss_rc = loss_rc.sum()
        loss_kl = loss_kl.sum()
        loss = loss.sum()
        
        report_rec_loss = loss_rc.item() / report_num_sents
        report_kl_loss = loss_kl.item() / report_num_sents
        
        if self.args.beta == 0:
            report_loss = loss.item() / report_num_sents
        else:
            if self.args.warm_up == 0 and self.args.kl_start < 1e-6:
                loss = loss_rc.item()
                report_loss = loss / report_num_sents	
            else:	
                report_loss = loss.item() / report_num_sents
        
        self.log("test/loss", report_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("test/loss_rc", report_rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("test/loss_kl", report_kl_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
        return {"test_loss": loss, "test_loss_rc": loss_rc, "test_loss_kl": loss_kl, "report_num_words": report_num_words, "report_num_sents": report_num_sents }
        
    def test_epoch_end(self, test_step_outputs):
        report_kl_loss = report_rec_loss = report_loss = 0
        report_num_words = report_num_sents = 0
        test_loss = 0.0
        nll = 0.0
        kl_loss = 0.0  
        rec_loss = 0.0
        ppl = 0.0

        for out in test_step_outputs:
            report_rec_loss += out["test_loss_rc"]
            report_kl_loss += out["test_loss_kl"]
            report_loss += out["test_loss"]
            report_num_words += out["report_num_words"]
            report_num_sents += out["report_num_sents"]

        test_loss = report_loss / report_num_sents
        nll = (report_kl_loss + report_rec_loss) / report_num_sents
        kl_loss = report_kl_loss / report_num_sents
        rec_loss = report_rec_loss / report_num_sents
        ppl = torch.exp(nll * report_num_sents / report_num_words) #np.exp
        
        
        with torch.no_grad():
            print("\rcalculating cur_mi", end="",flush=True)
            cur_mi = self.calc_mi(self.test_data)
            print("\rCalculating active units", end="",flush=True)    
            au, au_var = self.calc_au(self.test_data)
        
        print('\rEpoch: %d - Test_loss: %.4f, kl: %.4f, recon: %.4f, nll: %.4f, ppl: %.4f, active_units: %d, mutual_info: %.4f' % (self.current_epoch, 
                                                                                                                    test_loss,
                                                                                                                    kl_loss, 
                                                                                                                    rec_loss, 
                                                                                                                    nll, 
                                                                                                                    ppl, 
                                                                                                                    au, 
                                                                                                                    cur_mi))
                                                                                                                    
                                                                                                                    

    def sample_from_prior(self, nsamples):
        """sampling from prior distribution

        Returns: Tensor
            Tensor: samples from prior with shape (nsamples, nz)
        """
        return self.prior.sample((nsamples,)).unsqueeze(0)                                                                                                                  


# ##############################################################################################################

#     def loss_iw(self, x, kl_weight, nsamples=50, ns=10):
#         """
#         Args:
#             x: if the data is constant-length, x is the data tensor with
#                 shape (batch, *). Otherwise x is a tuple that contains
#                 the data tensor and length list

#         Returns: Tensor1, Tensor2, Tensor3
#             Tensor1: total loss [batch]
#             Tensor2: reconstruction loss shape [batch]
#             Tensor3: KL loss shape [batch]
#         """

#         mu, logvar, _ = self.encoder.forward(x)

#         ##################
#         # compute KL
#         ##################
#         KL = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)

#         tmp = []
#         reconstruct_err_sum = 0

#         #import pdb

#         for _ in range(int(nsamples / ns)):

#             # (batch, nsamples, nz)
#             z = self.encoder.reparameterize(mu, logvar, ns)

#             ##################
#             # compute qzx
#             ##################
#             nz = z.size(2)

#             # (batch_size, 1, nz)
#             _mu, _logvar = mu.unsqueeze(1), logvar.unsqueeze(1)
#             var = _logvar.exp()

#             # (batch_size, nsamples, nz)
#             dev = z - _mu

#             # (batch_size, nsamples)
#             log_qzx = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
#                 0.5 * (nz * math.log(2 * math.pi) + _logvar.sum(-1))

#             ##################
#             # compute qzx
#             ##################
#             log_pz = (-0.5 * math.log(2*math.pi) - z**2 / 2).sum(dim=-1)


#             ##################
#             # compute reconstruction loss
#             ##################
#             # (batch)
#             reconstruct_err = self.decoder.reconstruct_error(x, z)
#             reconstruct_err_sum += reconstruct_err.cpu().detach().sum(dim=1)

#             #pdb.set_trace()

#             tmp.append(reconstruct_err + kl_weight * (log_qzx - log_pz))

#         nll_iw = log_sum_exp(torch.cat(tmp, dim=-1), dim=-1) - math.log(nsamples)

#         return nll_iw, reconstruct_err_sum / nsamples, KL


#     def nll_iw(self, x, nsamples, ns=100):
#         """compute the importance weighting estimate of the log-likelihood
#         Args:
#             x: if the data is constant-length, x is the data tensor with
#                 shape (batch, *). Otherwise x is a tuple that contains
#                 the data tensor and length list
#             nsamples: Int
#                 the number of samples required to estimate marginal data likelihood
#         Returns: Tensor1
#             Tensor1: the estimate of log p(x), shape [batch]
#         """

#         # compute iw every ns samples to address the memory issue
#         # nsamples = 500, ns = 100
#         # nsamples = 500, ns = 10

#         # TODO: note that x is forwarded twice in self.encoder.sample(x, ns) and self.eval_inference_dist(x, z, param)
#         #.      this problem is to be solved in order to speed up

#         tmp = []
#         for _ in range(int(nsamples / ns)):
#             # [batch, ns, nz]
#             # param is the parameters required to evaluate q(z|x)
#             z, param, _ = self.encoder.sample(x, ns)

#             # [batch, ns]
#             log_comp_ll = self.eval_complete_ll(x, z)
#             log_infer_ll = self.eval_inference_dist(x, z, param)

#             tmp.append(log_comp_ll - log_infer_ll)

#         ll_iw = log_sum_exp(torch.cat(tmp, dim=-1), dim=-1) - math.log(nsamples)

#         return -ll_iw

#     def KL(self, x):
#         _, KL = self.encode(x, 1)

#         return KL

#     def eval_prior_dist(self, zrange):
#         """perform grid search to calculate the true posterior
#         Args:
#             zrange: tensor
#                 different z points that will be evaluated, with
#                 shape (k^2, nz), where k=(zmax - zmin)/space
#         """

#         # (k^2)
#         return self.prior.log_prob(zrange).sum(dim=-1)

#     def eval_complete_ll(self, x, z):
#         """compute log p(z,x)
#         Args:
#             x: Tensor
#                 input with shape [batch, seq_len]
#             z: Tensor
#                 evaluation points with shape [batch, nsamples, nz]
#         Returns: Tensor1
#             Tensor1: log p(z,x) Tensor with shape [batch, nsamples]
#         """

#         # [batch, nsamples]
#         log_prior = self.eval_prior_dist(z)
#         log_gen = self.eval_cond_ll(x, z)

#         return log_prior + log_gen

#     def eval_cond_ll(self, x, z):
#         """compute log p(x|z)
#         """

#         return self.decoder.log_probability(x, z)

#     def eval_log_model_posterior(self, x, grid_z):
#         """perform grid search to calculate the true posterior
#          this function computes p(z|x)
#         Args:
#             grid_z: tensor
#                 different z points that will be evaluated, with
#                 shape (k^2, nz), where k=(zmax - zmin)/pace

#         Returns: Tensor
#             Tensor: the log posterior distribution log p(z|x) with
#                     shape [batch_size, K^2]
#         """
#         try:
#             batch_size = x.size(0)
#         except:
#             batch_size = x[0].size(0)

#         # (batch_size, k^2, nz)
#         grid_z = grid_z.unsqueeze(0).expand(batch_size, *grid_z.size()).contiguous()

#         # (batch_size, k^2)
#         log_comp = self.eval_complete_ll(x, grid_z)

#         # normalize to posterior
#         log_posterior = log_comp - log_sum_exp(log_comp, dim=1, keepdim=True)

#         return log_posterior




#     def sample_from_posterior(self, x, nsamples):
#         """perform MH sampling from model posterior
#         Returns: Tensor
#             Tensor: samples from model posterior with
#                 shape (batch_size, nsamples, nz)
#         """

#         # use the samples from inference net as initial points
#         # for MCMC sampling. [batch_size, nsamples, nz]
#         cur = self.encoder.sample_from_inference(x, 1)
#         cur_ll = self.eval_complete_ll(x, cur)
#         total_iter = self.args.mh_burn_in + nsamples * self.args.mh_thin
#         samples = []
#         for iter_ in range(total_iter):
#             next = torch.normal(mean=cur,
#                 std=cur.new_full(size=cur.size(), fill_value=self.args.mh_std))
#             # [batch_size, 1]
#             next_ll = self.eval_complete_ll(x, next)
#             ratio = next_ll - cur_ll

#             accept_prob = torch.min(ratio.exp(), ratio.new_ones(ratio.size()))

#             uniform_t = accept_prob.new_empty(accept_prob.size()).uniform_()

#             # [batch_size, 1]
#             mask = (uniform_t < accept_prob).float()

#             mask_ = mask.unsqueeze(2)

#             cur = mask_ * next + (1 - mask_) * cur
#             cur_ll = mask * next_ll + (1 - mask) * cur_ll

#             if iter_ >= self.args.mh_burn_in and (iter_ - self.args.mh_burn_in) % self.args.mh_thin == 0:
#                 samples.append(cur.unsqueeze(1))


#         return torch.cat(samples, dim=1)
        


#     def calc_model_posterior_mean(self, x, grid_z):
#         """compute the mean value of model posterior, i.e. E_{z ~ p(z|x)}[z]
#         Args:
#             grid_z: different z points that will be evaluated, with
#                     shape (k^2, nz), where k=(zmax - zmin)/pace
#             x: [batch, *]

#         Returns: Tensor1
#             Tensor1: the mean value tensor with shape [batch, nz]

#         """

#         # [batch, K^2]
#         log_posterior = self.eval_log_model_posterior(x, grid_z)
#         posterior = log_posterior.exp()

#         # [batch, nz]
#         return torch.mul(posterior.unsqueeze(2), grid_z.unsqueeze(0)).sum(1)

#     def calc_infer_mean(self, x):
#         """
#         Returns: Tensor1
#             Tensor1: the mean of inference distribution, with shape [batch, nz]
#         """

#         mean, logvar_ = self.encoder.forward(x)

#         return mean



#     def eval_inference_dist(self, x, z, param=None):
#         """
#         Returns: Tensor
#             Tensor: the posterior density tensor with
#                 shape (batch_size, nsamples)
#         """
#         return self.encoder.eval_inference_dist(x, z, param)

#     def calc_mi_q(self, x):
#         """Approximate the mutual information between x and z
#         under distribution q(z|x)

#         Args:
#             x: [batch_size, *]. The sampled data to estimate mutual info
#         """

#         return self.encoder.calc_mi(x)
        
    ############################### OPTIMIZER ###########################################  
        
    
    
    def configure_optimizers(self):
        
        optimizer = self.optimizer 

        return optimizer
        
        
    def GPT_configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )

        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.Conv1d, torch.nn.LSTM, torch.nn.GRU)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif ('weight' in pn or 'bias' in pn) and isinstance(m, (torch.nn.LSTM, torch.nn.GRU)):
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('encoder.transformer.pos_emb')
        no_decay.add('decoder.transformer.pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.args.learning_rate, betas=(0.9, 0.95))
        return optimizer
        
        
#     ########################################### lr control ################################       
        
        
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def set_lr(self, lr):
        self.optimizer.param_groups[0]['lr'] = lr
        self.optimizer.param_groups[1]['lr'] = lr
        
        
#     ########################################### SAVE AND LOAD SOME VARIABLES ################################    
        
    def on_save_checkpoint(self, checkpoint):
        # 99% of use cases you don't need to implement this method
        checkpoint['kl_weight'] = self.kl_weight
        checkpoint['best_loss'] = self.best_loss
        checkpoint['pre_mi'] = self.pre_mi
        # checkpoint['lr'] = self.lr
         

    def on_load_checkpoint(self, checkpoint):
        self.kl_weight = checkpoint['kl_weight']
        self.best_loss = checkpoint['best_loss']
        self.pre_mi = checkpoint['pre_mi']
        # self.lr = checkpoint['lr']
        # self.set_lr(self.lr)
        

    ############################## DATA loaders #################################################
    
    def datamodule_loader(self):
        
        mel_num =  80
        spec_len = 860
        spec_crop_len = 848
        random_crop = False
        
        self.data = DataModule(batch_size =self.args.batch_size, 
                          spec_dir_path = self.args.spec_dir_path,
                          mel_num = mel_num, 
                          spec_len = spec_len, 
                          spec_crop_len =spec_crop_len, 
                          random_crop =random_crop,
                          num_workers=self.args.workers)
        self.data.setup()

    def train_dataloader(self):
        
        train_data = self.data.train_dataset
        self.len_train_data = len(train_data)
        loader = self.data.train_dataloader() 
        
        return loader
        
    def val_dataloader(self):
        
        val_data = self.data.val_dataset 
        self.len_val_data = len(val_data)    
        loader =  self.data.val_dataloader()
        
        return loader
        
    def test_dataloader(self):
        pass
        # test_data = MonoTextData(self.args.test_data, label=self.args.label, vocab=self.vocab)
    

        # test_data_batch =test_data.create_data_batch(batch_size=self.args.batch_size,
        #                                         device= "cpu",
        #                                         batch_first=True)
                                                
                                                
        # print()
        # loader =   self.data.test_dataloader() # DataLoader(test_data_batch, batch_size=None, num_workers=self.args.workers) #, pin_memory=True) #, pin_memory=True, pin_memory_device = self.args.device)
        
        # return loader