"""
taken from: https://github.com/karpathy/minGPT/
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import pdb
import math
import logging
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
sys.path.insert(0, '.')  # nopep8
import pytorch_lightning as pl

logger = logging.getLogger(__name__)

from datasets.datamodule import DataModule

from vqvae.big_model_attn_gan import LitVQVAE


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)




class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        mask = torch.tril(torch.ones(config.block_size,
                                     config.block_size))
        if hasattr(config, "n_unmasked"):
            mask[:config.n_unmasked, :config.n_unmasked] = 1
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = self.attn_drop(att) @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))

        return y, att


class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        # x = x + self.attn(self.ln1(x))

        # x is a tuple (x, attention)
        x, _ = x
        res = x
        x = self.ln1(x)
        x, att = self.attn(x)
        x = res + x

        x = x + self.mlp(self.ln2(x))

        return x, att

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """
    def __init__(self, args,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0, last_linear = None, block_size = None):
        super().__init__()
        config = GPTConfig(vocab_size=args.vocab_size, block_size=args.block_size,
                          embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                          n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                          n_unmasked=n_unmasked, last_linear = last_linear)
                          
        if block_size is not None:  # for manually controling block size in instances if nessesary
            config.block_size = block_size
        
        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        ### defining output shape. should be vocab size if decoder and shoud be 2*latent size for encoder (i take 1024)
        if config.last_linear is not None:
            output_size = last_linear
        else: 
            output_size = config.vocab_size
            
        self.head = nn.Linear(config.n_embd, output_size, bias=False)
        
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, embeddings=None, targets=None):
        # forward the GPT model
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector

        # pdb.set_trace()

        if embeddings is not None:  # prepend explicit embeddings
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)

        # returns only last layer attention
        # giving tuple (x, None) just because Sequential takes a single input but outputs two (x, atttention).
        # att is (B, H, T, T)
        x, att = self.blocks((x, None))
        x = self.ln_f(x)
        
        logits = self.head(x)
        # if hasattr(self, "head"):
        #     logits = self.head(x)
        # else:
        #     logits = x

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, att



class GPTClass(GPT):

    def __init__(self, args):
        super().__init__(args, embd_pdrop=args.embd_pdrop, resid_pdrop=args.resid_pdrop, attn_pdrop=args.attn_pdrop,n_unmasked=args.n_unmasked, last_linear=args.last_linear, block_size = args.block_size)
        self.embedder = nn.Embedding(args.class_size, args.n_embd)

    def forward(self, idx, token):
        token = self.embedder(token)
        # calling forward from super
        return super().forward(idx, embeddings=token)



class Lit_minGPT(pl.LightningModule):
    def __init__(self, 
                 args,
                 ckpt_path=None, 
                 ignore_keys=[],
                 first_stage_key="image",
                 cond_stage_key="depth",
                 downsample_cond_size=-1,
                 pkeep=1.0):

        super().__init__()
        self.args = args
        self.transformer = GPTClass(args)     
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.downsample_cond_size = downsample_cond_size
        self.pkeep = pkeep
        
        self.datamodule_loader()
        
        self.forward_shuffle_idx, self.backward_shuffle_idx  = self.make_idx (5, 53) ### for VQ code reading in coeerect order 
        
        if self.args.reconstruct_spec!="":
            self.first_stage_model = LitVQVAE(num_embeddings = 128 , embedding_dim = 256)
            self.first_stage_model.load_state_dict(torch.load(self.args.reconstruct_spec))
            self.first_stage_model.eval().to(self.args.device)
        



    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")


    def forward(self, x, c=None):
        # one step to produce the logits
        z_indices = x
        
        # if self.training and self.pkeep < 1.0:
        #     mask = torch.bernoulli(self.pkeep * torch.ones(z_indices.shape, device=z_indices.device))
        #     mask = mask.round().to(dtype=torch.int64)
        #     r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
        #     a_indices = mask*z_indices+(1-mask)*r_indices
        # else:
        #     a_indices = z_indices

        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = z_indices

        # in the case we do not want to encode condition anyhow (e.g. inputs are features)

        # make the prediction
        logits, _, _ = self.transformer(z_indices[:, :-1], c)
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        cond_size = c.size(-1)
        
        logits = logits[:, cond_size-1:]

        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, sample=False, top_k=None,
               callback=lambda k: None):

        block_size = self.transformer.get_block_size()
        assert not self.transformer.training
        if self.pkeep <= 0.0:
            raise NotImplementedError('Implement for GPTFeatsCLass')
            raise NotImplementedError('Implement for GPTFeats')
            raise NotImplementedError('Implement for GPTClass')
            raise NotImplementedError('also the model outputs attention')
            # one pass suffices since input is pure noise anyway
            assert len(x.shape)==2
            # noise_shape = (x.shape[0], steps-1)
            # noise = torch.randint(self.transformer.config.vocab_size, noise_shape).to(x)
            noise = c.clone()[:,x.shape[1]-c.shape[1]:-1]
            x = torch.cat((x,noise),dim=1)
            logits, _ = self.transformer(x)
            # take all logits for now and scale by temp
            logits = logits / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                shape = probs.shape
                probs = probs.reshape(shape[0]*shape[1],shape[2])
                ix = torch.multinomial(probs, num_samples=1)
                probs = probs.reshape(shape[0],shape[1],shape[2])
                ix = ix.reshape(shape[0],shape[1])
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # cut off conditioning
            x = ix[:, c.shape[1]-1:]
        else:

            for k in range(steps):
                callback(k)
                if isinstance(self.transformer, (GPTClass)):
                    # if assert is removed, you need to make sure that the combined len is not longer block_s
                    cond_size = c.size(-1)
                    assert x.size(1) + cond_size <= block_size

                    x_cond = x
                    c_cond = c
                    logits, _, att = self.transformer(x_cond, c_cond)
                else:
                    assert x.size(1) <= block_size  # make sure model can see conditioning
                    x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
                    logits, _, att = self.transformer(x_cond)
                # pluck the logits at the final step and scale by temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop probabilities to only the top k options
                if top_k is not None:
                    logits = self.top_k_logits(logits, top_k)
                # apply softmax to convert to probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution or take the most likely
                if sample:
                    ix = torch.multinomial(probs, num_samples=1)
                else:
                    _, ix = torch.topk(probs, k=1, dim=-1)
                # append to the sequence and continue
                x = torch.cat((x, ix), dim=1)

        return x, att.detach().cpu()



####################################### Getting data and train/val/test steps ######################

    def get_input(self, key, batch):
        if isinstance(key, str):
            # if batch[key] is 1D; else the batch[key] is 2D
            if key in ['feature', 'target']:
                x = self.cond_stage_model.get_input(batch, key)
            else:
                x = batch[key]
                if len(x.shape) == 3:
                    x = x[..., None]
                x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
            if x.dtype == torch.double:
                x = x.float()
        elif isinstance(key, ListConfig):
            x = self.cond_stage_model.get_input(batch, key)
            for k, v in x.items():
                if v.dtype == torch.double:
                    x[k] = v.float()
        return x



    def get_x(self, batch):
        x = batch['codes']
    
        x = x.permute(0, 2, 1)#.to(memory_format=torch.contiguous_format)
        x = torch.flatten(x, start_dim=1)
        x = x.to(self.args.device) # just in case it it's not there yet
    
        return x

    def get_c(self, batch):
        c = batch["target"].unsqueeze(1)
    
        c = c.to(self.args.device) # just in case it it's not there yet
    
        return c
        
    def get_xc(self, batch, N=None):
        x = self.get_x(batch)
        c = self.get_c(batch)
        
        if N is not None:
            x = x[:N]
            c = c[:N]

        return x, c        

    def shared_step(self, batch, batch_idx):
        x, c = self.get_xc(batch)
        logits, target = self(x, c)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss        

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
    

    ############################## DATA loaders #################################################
    
    def datamodule_loader(self):
        
        mel_num =  80
        spec_len = 860
        spec_crop_len = 848
        random_crop = False
        
        self.data = DataModule(batch_size =self.args.batch_size, 
                          spec_dir_path = self.args.spec_dir_path,  #### TODO this points to spec and code finds other data relative to that need to do it proper 
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


############################################### LOGGING ???? ################################
        
        
    ####TODO
    ### need to take this out of the model into callbacks!!!

    @torch.no_grad()
    def decode_to_img(self, index, zshape, stage='first'):
        if stage == 'first':
            index = self.code_reader(index, reverse=True)
        elif stage == 'cond':
            print('in cond stage in decode_to_img which is unexpected ')
            index = self.cond_stage_permuter(index, reverse=True)
        else:
            raise NotImplementedError

        bhwc = (zshape[0], zshape[2], zshape[3], zshape[1])
        quant_z = self.first_stage_model._vq_vae.get_codebook_entry(index.reshape(-1), shape=bhwc)
        x = self.first_stage_model.decode(quant_z)
        return x

    @torch.no_grad()
    def log_images(self, batch, temperature=None, top_k=None, callback=None, lr_interface=False, **kwargs):
        log = dict()

        N = 1 # to make logging faster, because i log only 1 audio and picture anyways!!
        if lr_interface:
            x, c = self.get_xc(batch, N, diffuse=False, upsample_factor=8)
        else:
            x, c = self.get_xc(batch, N)
        x = x.to(device=self.device)
        # c = c.to(device=self.device)
        if isinstance(c, dict):
            c = {k: v.to(self.device) for k, v in c.items()}
        else:
            c = c.to(self.device)

        quant_z = torch.zeros((x.size(0), 256, 5, 53)) 
        
        z_indices = x
        
        quant_c = c   # output can be features or a single class or a featcls dict
        c_indices = c
        
        # print(c_indices)
        
        # create a "half"" sample
        z_start_indices = z_indices[:, :z_indices.shape[1]//2]
        index_sample_half, att_half = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1]-z_start_indices.shape[1],
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample = self.decode_to_img(index_sample_half, quant_z.shape)

        # sample
        z_start_indices = z_indices[:, :0]
        index_sample_nopix, att_nopix = self.sample(z_start_indices, c_indices,
                                              steps=z_indices.shape[1],
                                              temperature=temperature if temperature is not None else 1.0,
                                              sample=True,
                                              top_k=top_k if top_k is not None else 100,
                                              callback=callback if callback is not None else lambda k: None)
        x_sample_nopix = self.decode_to_img(index_sample_nopix, quant_z.shape)

        # det sample
        z_start_indices = z_indices[:, :0]
        index_sample_det, att_det = self.sample(z_start_indices, c_indices,
                                            steps=z_indices.shape[1],
                                            sample=False,
                                            callback=callback if callback is not None else lambda k: None)
        x_sample_det = self.decode_to_img(index_sample_det, quant_z.shape)

        # reconstruction
        x_rec = self.decode_to_img(z_indices, quant_z.shape)
        
        
        
        log["codes"] = x
        log["codes_half"] = index_sample_half
        log["codes_nopix"] = index_sample_nopix
        log["codes_det"] = index_sample_det
        
        # print(log["codes"])

        log["inputs"] = self.get_input("image", batch)   #.flip(dims=(1, 2)) ### getting original spectrogram form database
        log["reconstructions"] = x_rec

        if isinstance(self.cond_stage_key, str):   ### not using this bassically
            cond_is_not_image = self.cond_stage_key != "image"

        if cond_is_not_image:
            cond_rec = quant_c
            log["conditioning_rec"] = cond_rec
            log["conditioning"] = c

        log["samples_half"] = x_sample
        log["samples_nopix"] = x_sample_nopix
        log["samples_det"] = x_sample_det
        log["att_half"] = att_half
        log["att_nopix"] = att_nopix
        log["att_det"] = att_det
        return log



############################################### optimisers #########################################

    def configure_optimizers(self):
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
        for mn, m in self.transformer.named_modules():
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
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
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


