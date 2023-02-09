from modules.Lit_vae import VAE
import torch
from utils import calc_iwnll, calc_mi, calc_au
# from tqdm import tqdm
# import sys
from vqvae.big_model_attn_gan import LitVQVAE
import torchvision
import yaml
from vocoder.modules import Generator
import librosa
from pathlib import Path
import os
import soundfile as sf
import wave
import numpy as np

from utils import visualize_latent

from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from PIL import Image
import yaml
from pathlib import Path


####################### CALLBACKS ################################


class ImageLogger(Callback):
  def __init__(self, batch_frequency=200, max_images=1, clamp=True, increase_log_steps=False,
                for_specs=False, vocoder_cfg=None, spec_dir_name=None, sample_rate=None, args=None):
      super().__init__()
      self.args = args
      
      if self.args.logging_frequency > 0:
          self.batch_freq = self.args.logging_frequency
      else:
          self.batch_freq = batch_frequency
      
      self.max_images = max_images
    #   self.logger_log_images = pl.loggers.TensorBoardLogger

    #   self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
    #   if not increase_log_steps:
    #       self.log_steps = [self.batch_freq]
          
    #   print(self.log_steps)
      
      self.clamp = clamp

      self.spec_dir_name = spec_dir_name
      self.sample_rate = sample_rate
      print('We will not save audio for conditioning and conditioning_rec')

      if self.sample_rate == None:
        try:
          self.sample_rate = self.args.sample_rate
        except ValueError:
          print("you'll need a sample_rate and it's None now")

      if self.args.vocoder!="":
          self.for_specs = True
          self.vocoder = self.load_vocoder(self.args.vocoder, eval_mode=True)['model'].to(self.args.device)
      
  def load_vocoder(self, ckpt_vocoder: str, eval_mode: bool):
      ckpt_vocoder = Path(ckpt_vocoder)
      vocoder_sd = torch.load(ckpt_vocoder / 'best_netG.pt', map_location='cpu')
  
      with open(ckpt_vocoder / 'args.yml', 'r') as f:
          args = yaml.load(f, Loader=yaml.UnsafeLoader)
  
      vocoder = Generator(args.n_mel_channels, args.ngf, args.n_residual_layers)
      vocoder.load_state_dict(vocoder_sd)
  
      if eval_mode:
          vocoder.eval()
  
      return {'model': vocoder}

  def _visualize_attention(self, attention, scale_by_prior=True):
      if scale_by_prior:
          B, H, T, T = attention.shape
          # attention weight is 1/T: if we have a seq with length 3 the weights are 1/3, 1/3, and 1/3
          # making T by T matrix with zeros in the upper triangular part
          attention_uniform_prior = 1 / torch.arange(1, T+1).view(1, T, 1).repeat(B, 1, T)
          attention_uniform_prior = attention_uniform_prior.tril().view(B, 1, T, T).to(attention.device)
          attention = attention - attention_uniform_prior

      attention_agg = attention.sum(dim=1, keepdims=True)
      return attention_agg

  def _log_rec_audio(self, specs, tag, global_step, pl_module=None, save_rec_path=None):

      # specs are (B, 1, F, T)
      for i, spec in enumerate(specs):
          spec = spec.data.squeeze(0).cpu().numpy()
          # audios are in [-1, 1], making them in [0, 1]
          spec = (spec + 1) / 2
          # wave = self.vocoder.vocode(spec, global_step)
          with torch.no_grad():
            wave = self.vocoder(torch.from_numpy(spec).unsqueeze(0).to(self.args.device)).squeeze().cpu().numpy()
          wave = torch.from_numpy(wave).unsqueeze(1)
          if pl_module is not None:
              pl_module.logger.experiment.add_audio(f'{tag}_{i}', wave, pl_module.global_step, self.sample_rate)
          # in case we would like to save it on disk
          # if save_rec_path is not None:
          #     try:
          #         librosa.output.write_wav(save_rec_path, wave.squeeze(0).numpy(), self.sample_rate)
          #     except AttributeError:
          #         soundfile.write(save_rec_path, wave.squeeze(0).numpy(), self.sample_rate, 'FLOAT')
  
  @rank_zero_only
  def log_everything(self, pl_module, images, batch, batch_idx, split):

    cond_stage_model = "ClassOnlyStage" # in the future when we use some temporal features we can use other names from below

    for k in images:
      tag = f'{split}/{k}'
      if cond_stage_model in ['ClassOnlyStage', 'FeatsClassStage'] and k in ['conditioning', 'conditioning_rec']:
          # saving the classes for the current batch
          limit = min(len(batch["label"]), self.max_images)
          pl_module.logger.experiment.add_text(tag, '; '.join(batch['label'][:limit]), global_step = pl_module.global_step)
          # breaking here because we don't want to call add_image
          if cond_stage_model == 'FeatsClassStage':
              grid = torchvision.utils.make_grid(images[k]['feature'].unsqueeze(1).permute(0, 1, 3, 2), nrow=1, normalize=True)
          else:
              continue
    
      elif k in ["codes", "codes_half", "codes_nopix", "codes_det"]:
          # print(images[k])
          pl_module.logger.experiment.add_text(tag, str(images[k].tolist()), global_step = pl_module.global_step )
          continue
    
      elif k in ['att_nopix', 'att_half', 'att_det']:
          B, H, T, T = images[k].shape
          grid = torchvision.utils.make_grid(self._visualize_attention(images[k]), nrow=H, normalize=True)
      elif cond_stage_model in ['RawFeatsStage', 'VQModel1d', 'FeatClusterStage'] and k in ['conditioning', 'conditioning_rec']:
          grid = torchvision.utils.make_grid(images[k].unsqueeze(1).permute(0, 1, 3, 2), nrow=1, normalize=True)
      else:
          if self.for_specs:
              # flipping values along frequency dim, otherwise mels are upside-down (1, F, T)
              grid = torchvision.utils.make_grid(images[k].flip(dims=(2,)), nrow=1)
              # also reconstruct waveform given the spec and inv_transform
              if k not in ['conditioning', 'conditioning_rec', 'att_nopix', 'att_half', 'att_det']:
                  self._log_rec_audio(images[k], tag, pl_module.global_step, pl_module=pl_module)
          else:
              grid = torchvision.utils.make_grid(images[k])
          # attention is already in [0, 1] therefore ignoring this line
          grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
      pl_module.logger.experiment.add_image(tag, grid, global_step=pl_module.global_step)
    # pl_module.logger.experiment.flush()

  # ###@rank_zero_only
  # def log_local(self, pl_module, split, images, batch, batch_idx):
  #     root = os.path.join(pl_module.logger.save_dir, 'images', split)

  #     cond_stage_model = "ClassOnlyStage" # in the future when we use some temporal features we can use other names from below
      
  #     # for key in images:
  #     #     print(key)
  #     #     print(images[key].shape)

  #     for k in images:
  #         if cond_stage_model in ['ClassOnlyStage', 'FeatsClassStage'] and k in ['conditioning', 'conditioning_rec']:
  #             filename = '{}_gs-{:06}_e-{:03}_b-{:06}.txt'.format(
  #                 k,
  #                 pl_module.global_step,
  #                 pl_module.current_epoch,
  #                 batch_idx)
  #             path = os.path.join(root, filename)
  #             os.makedirs(os.path.split(path)[0], exist_ok=True)
  #             # saving the classes for the current batch
  #             with open(path, 'w') as file:
  #                 file.write('\n'.join(batch['label']))
  #             # next loop iteration here because we don't want to call add_image
  #             if cond_stage_model == 'FeatsClassStage':
  #                 grid = torchvision.utils.make_grid(images[k]['feature'].unsqueeze(1).permute(0, 1, 3, 2), nrow=1, normalize=True)
  #             else:
  #                 continue
  #         elif k in ['att_nopix', 'att_half', 'att_det']:  # GPT CLass
  #             B, H, T, T = images[k].shape
  #             grid = torchvision.utils.make_grid(self._visualize_attention(images[k]), nrow=H, normalize=True)
  #         elif cond_stage_model in ['RawFeatsStage', 'VQModel1d', 'FeatClusterStage'] and k in ['conditioning', 'conditioning_rec']:
  #             grid = torchvision.utils.make_grid(images[k].unsqueeze(1).permute(0, 1, 3, 2), nrow=1, normalize=True)
  #         else:
  #             if self.for_specs:
  #                 print(k)
  #                 # flipping values along frequency dim, otherwise mels are upside-down (1, F, T)
  #                 grid = torchvision.utils.make_grid(images[k].flip(dims=(2,)), nrow=1)
  #             else:
  #                 grid = torchvision.utils.make_grid(images[k], nrow=4)
  #             # attention is already in [0, 1] therefore ignoring this line
  #             grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w

  #         grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
  #         grid = grid.numpy()
  #         grid = (grid*255).astype(np.uint8)
  #         filename = '{}_gs-{:06}_e-{:03}_b-{:06}.png'.format(
  #             k,
  #             pl_module.global_step,
  #             pl_module.current_epoch,
  #             batch_idx)
  #         path = os.path.join(root, filename)
  #         os.makedirs(os.path.split(path)[0], exist_ok=True)
  #         Image.fromarray(grid).save(path)

  #         # also save audio on the disk
  #         if self.for_specs:
  #             tag = f'{split}/{k}'
  #             filename = filename.replace('.png', '.wav')
  #             path = os.path.join(root, filename)
  #             if k not in ['conditioning', 'conditioning_rec', 'att_nopix', 'att_half', 'att_det']:
  #                 self._log_rec_audio(images[k], tag, pl_module.global_step, save_rec_path=path)

  def log_img(self, pl_module, batch, batch_idx, split='train'):
      if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
              hasattr(pl_module, 'log_images') and
              callable(pl_module.log_images) and
              self.max_images > 0 and
              pl_module.first_stage_key != 'feature'):
          logger = type(pl_module.logger)

          is_train = pl_module.training
          if is_train:
              pl_module.eval()

          with torch.no_grad():
              images = pl_module.log_images(batch, split=split)
              
          for k in images:
              if isinstance(images[k], dict):
                  N = min(images[k]['feature'].shape[0], self.max_images)
                  images[k]['feature'] = images[k]['feature'][:N]
                  if isinstance(images[k]['feature'], torch.Tensor):
                      images[k]['feature'] = images[k]['feature'].detach().cpu()
                      if self.clamp and k not in ["codes", "codes_half", "codes_nopix", "codes_det"]:
                          images[k]['feature'] = torch.clamp(images[k]['feature'], -1., 1.)
              else:
                  N = min(images[k].shape[0], self.max_images)
                  images[k] = images[k][:N]
                  if isinstance(images[k], torch.Tensor):
                      images[k] = images[k].detach().cpu()
                      if self.clamp and k not in ["codes", "codes_half", "codes_nopix", "codes_det"]:
                          images[k] = torch.clamp(images[k], -1., 1.)

          # self.log_local(pl_module, split, images, batch, batch_idx)
          # logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
          # print(logger_log_images)
          # logger_log_images(pl_module, images, batch, pl_module.global_step, split)

          self.log_everything(pl_module, images, batch, pl_module.global_step, split)
        #   pl_module.logger.experiment.flush()

          if is_train:
              pl_module.train()

  def check_frequency(self, batch_idx):
      if (batch_idx % self.batch_freq) == 0: # or (batch_idx in self.log_steps):
        #   try:
        #       self.log_steps.pop(0)
        #   except IndexError:
        #       pass
          return True
      return False

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx): #, dataloader_idx):
      self.log_img(pl_module, batch, batch_idx, split='train')

  def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
      self.log_img(pl_module, batch, batch_idx, split='val')
        
        