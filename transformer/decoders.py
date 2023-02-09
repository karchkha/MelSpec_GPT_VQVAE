import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from .minGPT import GPT

class GPTDecoder(nn.Module):
    """GPT decoder with constant-length data and conditioning first element"""
    def __init__(self, args,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0, last_linear = None, block_size = None):
        super().__init__()

        self.args = args
        self.transformer = GPT(self.args, embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop, n_unmasked=n_unmasked, \
        last_linear = last_linear, block_size = block_size)
        
        vocab_mask = torch.ones(args.vocab_size)
        self.loss = nn.CrossEntropyLoss(weight=vocab_mask, reduction='none')

    def forward(self, x, c=None):
        # one step to produce the logits
        z_indices = x
        
        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = z_indices

        # make the prediction
        logits, _, _ = self.transformer(z_indices[:, :-1], c)
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        cond_size = c.size(-2)
        
        logits = logits[:, cond_size-1:]

        return logits, target

    def reconstruct_error(self, x, z):
        """Cross Entropy as in the language case
        Args:
            x: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        Returns:
            loss: (batch_size, n_sample). Loss
            across different sentence and z
        """

        batch_size, seq_len = x.size()
        n_sample = z.size(1)

        output_logits, tgt = self(x, z)

        # pdb.set_trace()
        if n_sample == 1:
            tgt = tgt.contiguous().view(-1)
        else:
            # (batch_size * n_sample * seq_len)
            tgt = tgt.unsqueeze(1).expand(batch_size, n_sample, seq_len) \
                     .contiguous().view(-1)

        # (batch_size * n_sample * seq_len)
        loss = self.loss(output_logits.view(-1, output_logits.size(2)),
                         tgt)

        # (batch_size, n_sample)
        return loss.view(batch_size, n_sample, -1).sum(-1)   
    
  
    def log_probability(self, x, z):
        """Cross Entropy in the language case
        Args:
            x: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        Returns:
            log_p: (batch_size, n_sample).
                log_p(x|z) across different x and z
        """

        return -self.reconstruct_error(x, z)
        
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

        for k in range(steps):
            callback(k)

            # if assert is removed, you need to make sure that the combined len is not longer block_s
            cond_size = c.size(-2)
            assert x.size(1) + cond_size <= block_size

            x_cond = x
            c_cond = c
            
            logits, _, att = self.transformer(x_cond, c_cond)

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

