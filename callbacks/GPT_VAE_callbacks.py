import pdb

from pytorch_lightning.callbacks import Callback, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
# from data import MonoTextData

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

from utils import visualize_latent

####################### CALLBACKS ################################



#########################TEXT LOGGER for VAE ######################################        
        
class TextLogger(Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if self.args.reconstruct_spec!="":
            self.vqvae_model = LitVQVAE(num_embeddings = self.args.vocab_size , embedding_dim = 256)
            self.vqvae_model.load_state_dict(torch.load(self.args.reconstruct_spec))
            self.vqvae_model.eval().to(self.args.device)
        
        if self.args.vocoder!="":
            self.melgan = self.load_vocoder(self.args.vocoder, eval_mode=True)['model'].to(self.args.device)
        
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

    def check_frequency(self, batch_idx):
        if batch_idx % self.args.logging_frequency == 0:
            return True
        return False
        
    def pad(self, list, size, padding):
        return list + [padding] * abs((len(list)-size))
    
        
    def batch_to_sentence(self, data):
        sentence = ""

        batch_size, sent_size = data.size()
        batch_size = 1 #### limit to only 1 sentance
        
        decoded_batch = [[] for _ in range(batch_size)]
        
        for i in range(batch_size):
            for j in range (1, sent_size-1):
                decoded_batch[i].append(data[i, j].item())
        
        for sent in decoded_batch:
            line = " ".join(str(sent)) + "  \n"
            sentence += line
        
        return sentence
        

    def spec_to_audio_to_st(self, x, sample_rate, vocoder=None):
        # audios are in [-1, 1], making them in [0, 1]
        spec = (x.data.squeeze(0) + 1) / 2
    
        out = {}
        if vocoder:
            # (L,) <- wave: (1, 1, L).squeeze() <- spec: (1, F, T)
            wave_from_vocoder = vocoder(spec).squeeze().cpu().detach().numpy()
            out['vocoder'] = wave_from_vocoder
        return out
        
    def get_input(self, batch):   # for getting original spectrograms from database
        x = batch['image']
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()
            

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.check_frequency(batch_idx):
            self.log_everything(pl_module, batch, batch_idx, split='train')

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.check_frequency(batch_idx) or batch_idx == pl_module.len_val_data-1:        
            self.log_everything(pl_module, batch, batch_idx, split='val')
            
            
            
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
            

    def log_everything(self, pl_module, batch, batch_idx, split='train'):

        logger = type(pl_module.logger)
        
        self.vqvae_model.to(pl_module.device)
        self.melgan.to(pl_module.device)
        
        
        ###################### original spectrogram ####################################
        original_spec = self.get_input(batch)[0].unsqueeze(0) # limiting to 1-st image
        original_spec_for_image = self.log_images(original_spec)
        # pl_module.logger.experiment.add_image(f'{split}/epoch-{pl_module.current_epoch}/step-{batch_idx}/original', original_spec, global_step=batch_idx)
        pl_module.logger.experiment.add_image(f'{split}/original', original_spec_for_image, global_step=pl_module.global_step)
        
        ################################################# original audio ############################################
        if self.args.vocoder !="":
            orig_file_path = '../AV_Datasets/VAS/'+batch['label'][0]+'/videos/'+batch['file_path_'][0][-19:-8]+'.mp4'
            if os.path.isfile(orig_file_path):
                try:
                    waves, _ = librosa.load(orig_file_path, sr=22050)
                    # waves, _ = sf.read(orig_file_path, samplerate=2250 )
                    # waves = wave.open(orig_file_path,'r')
                    waves = torch.from_numpy(waves).unsqueeze(1)
                    # pl_module.logger.experiment.add_audio(f'{split}/epoch-{pl_module.current_epoch}/step-{batch_idx}/original_audio', waves, batch_idx, 22050)
                    pl_module.logger.experiment.add_audio(f'{split}/original_audio', waves, pl_module.global_step, 22050)
                except:
                    pass
            else:
                waves = self.spec_to_audio_to_st(original_spec, 22050, vocoder=self.melgan)
                waves = torch.from_numpy(waves['vocoder']).unsqueeze(1)
                # pl_module.logger.experiment.add_audio(f'{split}/epoch-{pl_module.current_epoch}/step-{batch_idx}/reconstraction_audio', waves, batch_idx, 22050)
                pl_module.logger.experiment.add_audio(f'{split}/original_audio', waves, pl_module.global_step, 22050)



        ###################### original codebook logging as sentence ####################################
        codes_from_data = pl_module.get_input(batch)
        original = self.batch_to_sentence(codes_from_data)       
        # pl_module.logger.experiment.add_text(f'{split}/epoch-{pl_module.current_epoch}/step-{batch_idx}/original', original, global_step=batch_idx)
        pl_module.logger.experiment.add_text(f'{split}/original', original, global_step=pl_module.global_step)
        
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.no_grad():


            
            
            ############################################## Greedy reconstruct ##############################################################
            ################################################################################################################################

            if self.args.test_interpolation:
                self.audio_interpolation(model = pl_module, batch = batch, batch_idx = batch_idx, split = split, strategy = "greedy")
                
            ############# codebook reconstruction ##########################
            codes, atts = pl_module.reconstruct(codes_from_data, "greedy")
            text_reconstructed = self.batch_to_sentence(codes)
            # pl_module.logger.experiment.add_text(f'{split}/epoch-{pl_module.current_epoch}/step-{batch_idx}/reconstraction', text_reconstructed, global_step=batch_idx)
            pl_module.logger.experiment.add_text(f'{split}/greedy_reconstraction', text_reconstructed, global_step=pl_module.global_step)
            
            ######################### Visualise Attentions ###########################
            att_enc, att_dec = atts
            att_enc = att_enc[0].unsqueeze(0)
            att_dec = att_dec[0].unsqueeze(0)
            
            tag = f'{split}/encoder'
            B, H, T, T = att_enc.shape
            grid = torchvision.utils.make_grid(self._visualize_attention(att_enc), nrow=H, normalize=True)
            pl_module.logger.experiment.add_image(tag, grid, global_step=pl_module.global_step)
            
            tag = f'{split}/decoder'
            grid = torchvision.utils.make_grid(self._visualize_attention(att_dec), nrow=H, normalize=True)
            pl_module.logger.experiment.add_image(tag, grid, global_step=pl_module.global_step)



            
            ############# spectrum reconstruction ##########################
            if self.args.reconstruct_spec !="":
                spec_reconstructions = self.codes_to_spec(codes, self.vqvae_model, pl_module)
                spec_reconstructions_image = self.log_images(spec_reconstructions)
                # pl_module.logger.experiment.add_image(f'{split}/epoch-{pl_module.current_epoch}/step-{batch_idx}/reconstraction', spec_reconstructions_image, global_step=batch_idx)
                pl_module.logger.experiment.add_image(f'{split}/greedy_reconstraction', spec_reconstructions_image, global_step=pl_module.global_step)
                
            ############# audio reconstruction ##########################
            if self.args.vocoder !="":
                waves = self.spec_to_audio_to_st(spec_reconstructions, 22050, vocoder=self.melgan)
                waves = torch.from_numpy(waves['vocoder']).unsqueeze(1)
                # pl_module.logger.experiment.add_audio(f'{split}/epoch-{pl_module.current_epoch}/step-{batch_idx}/reconstraction_audio', waves, batch_idx, 22050)
                pl_module.logger.experiment.add_audio(f'{split}/greedy_reconstraction_audio', waves, pl_module.global_step, 22050)
            

#             ################# Sampling form prior ####################
            
#             z = pl_module.sample_from_prior(1)

#             z = z.to(pl_module.device)  
  
#             ###### codebook sampled ###############
#             codes, _ =  pl_module.decode(z, "greedy")                  
#             sampled_from_prior = self.batch_to_sentence(codes)
#             # pl_module.logger.experiment.add_text(f'{split}/epoch-{pl_module.current_epoch}/step-{batch_idx}/sampled_from_prior', sampled_from_prior, global_step=batch_idx)
#             pl_module.logger.experiment.add_text(f'{split}/greedy_sampled_from_prior', sampled_from_prior, global_step=pl_module.global_step)
            
#             ###### spectrogram sampled ###############
#             if self.args.reconstruct_spec !="":
#                 spec_sampled = self.codes_to_spec(codes, self.vqvae_model, pl_module)
#                 spec_sampled_image = self.log_images(spec_sampled)
#                 # pl_module.logger.experiment.add_image(f'{split}/epoch-{pl_module.current_epoch}/step-{batch_idx}/sample_from_prior', spec_sampled_image, global_step=batch_idx)
#                 pl_module.logger.experiment.add_image(f'{split}/greedy_sample_from_prior', spec_sampled_image, global_step=pl_module.global_step) 

#             if self.args.vocoder !="":
#                 waves = self.spec_to_audio_to_st(spec_sampled, 22050, vocoder=self.melgan)
#                 waves = torch.from_numpy(waves['vocoder']).unsqueeze(1)
#                 # pl_module.logger.experiment.add_audio(f'{split}/epoch-{pl_module.current_epoch}/step-{batch_idx}/sample_from_prior_audio', waves, batch_idx, 22050)
#                 pl_module.logger.experiment.add_audio(f'{split}/greedy_sample_from_prior_audio', waves, pl_module.global_step, 22050)
                

            # ############################################## beam reconstruct ##############################################################
            # ################################################################################################################################

            # TODO: make beam True False arg that decides if this part of the code runs or not!!!
            # TODO: make real beam search, this is some bullshit! :))


            if self.args.test_interpolation:
                self.audio_interpolation(model = pl_module, batch = batch, batch_idx = batch_idx, split = split, strategy = "beam")

            ############# codebook reconstruction ##########################
            codes, atts = pl_module.reconstruct(codes_from_data, "beam")
            text_reconstructed = self.batch_to_sentence(codes)
            # pl_module.logger.experiment.add_text(f'{split}/epoch-{pl_module.current_epoch}/step-{batch_idx}/reconstraction', text_reconstructed, global_step=batch_idx)
            pl_module.logger.experiment.add_text(f'{split}/beam_reconstraction', text_reconstructed, global_step=pl_module.global_step)
            
            
            
            ######################### Visualise Attentions ###########################
            att_enc, att_dec = atts
            att_enc = att_enc[0].unsqueeze(0)
            att_dec = att_dec[0].unsqueeze(0)
            
            tag = f'{split}/encoder-beam'
            B, H, T, T = att_enc.shape
            grid = torchvision.utils.make_grid(self._visualize_attention(att_enc), nrow=H, normalize=True)
            pl_module.logger.experiment.add_image(tag, grid, global_step=pl_module.global_step)
            
            tag = f'{split}/decoder-beam'
            grid = torchvision.utils.make_grid(self._visualize_attention(att_dec), nrow=H, normalize=True)
            pl_module.logger.experiment.add_image(tag, grid, global_step=pl_module.global_step)
            
            
            
            ############# spectrum reconstruction ##########################
            if self.args.reconstruct_spec !="":
                spec_reconstructions = self.codes_to_spec(codes, self.vqvae_model, pl_module)
                spec_reconstructions_image = self.log_images(spec_reconstructions)
                # pl_module.logger.experiment.add_image(f'{split}/epoch-{pl_module.current_epoch}/step-{batch_idx}/reconstraction', spec_reconstructions_image, global_step=batch_idx)
                pl_module.logger.experiment.add_image(f'{split}/beam_reconstraction', spec_reconstructions_image, global_step=pl_module.global_step)
                
            
            if self.args.vocoder !="":
                waves = self.spec_to_audio_to_st(spec_reconstructions, 22050, vocoder=self.melgan)
                waves = torch.from_numpy(waves['vocoder']).unsqueeze(1)
                # pl_module.logger.experiment.add_audio(f'{split}/epoch-{pl_module.current_epoch}/step-{batch_idx}/reconstraction_audio', waves, batch_idx, 22050)
                pl_module.logger.experiment.add_audio(f'{split}/beam_reconstraction_audio', waves, pl_module.global_step, 22050)


#             ################# Sampling form prior ####################
            
#             z = pl_module.sample_from_prior(1)
#             z = z.to(pl_module.device)              
#             ###### codebook sampled ###############
#             codes, _ =  pl_module.decode(z, "beam")                 
#             sampled_from_prior = self.batch_to_sentence(codes)
#             # pl_module.logger.experiment.add_text(f'{split}/epoch-{pl_module.current_epoch}/step-{batch_idx}/sampled_from_prior', sampled_from_prior, global_step=batch_idx)
#             pl_module.logger.experiment.add_text(f'{split}/beam_sampled_from_prior', sampled_from_prior, global_step=pl_module.global_step)
            
#             ###### spectrogram sampled ###############
#             if self.args.reconstruct_spec !="":
#                 spec_sampled = self.codes_to_spec(codes, self.vqvae_model, pl_module)
#                 spec_sampled_image = self.log_images(spec_sampled)
#                 # pl_module.logger.experiment.add_image(f'{split}/epoch-{pl_module.current_epoch}/step-{batch_idx}/sample_from_prior', spec_sampled_image, global_step=batch_idx)
#                 pl_module.logger.experiment.add_image(f'{split}/beam_sample_from_prior', spec_sampled_image, global_step=pl_module.global_step) 

#             if self.args.vocoder !="":
#                 waves = self.spec_to_audio_to_st(spec_sampled, 22050, vocoder=self.melgan)
#                 waves = torch.from_numpy(waves['vocoder']).unsqueeze(1)
#                 # pl_module.logger.experiment.add_audio(f'{split}/epoch-{pl_module.current_epoch}/step-{batch_idx}/sample_from_prior_audio', waves, batch_idx, 22050)
#                 pl_module.logger.experiment.add_audio(f'{split}/beam_sample_from_prior_audio', waves, pl_module.global_step, 22050)

                


        if is_train:
            pl_module.train()
            
    def audio_interpolation(self, model, batch, batch_idx, split, strategy):

        spec = batch['image']

        spec_from = spec[0].unsqueeze(0).unsqueeze(0).to(torch.float32).to(self.args.device)
        spec_to = spec[1].unsqueeze(0).unsqueeze(0).to(torch.float32).to(self.args.device)
            
        ###### get codebook of those data  ###
        

        codebook_from = self.vqvae_model.encode(spec_from)
        _, _, info = self.vqvae_model._vq_vae(codebook_from)
        codebook_from = info[2].squeeze(1).unsqueeze(0)
        codebook_from = model.code_reader(codebook_from)

        codebook_to = self.vqvae_model.encode(spec_to)
        _, _, info = self.vqvae_model._vq_vae(codebook_to)
        codebook_to = info[2].squeeze(1).unsqueeze(0)     
        codebook_to = model.code_reader(codebook_to)
 
        z_from = model.sample_from_inference(codebook_from) #.squeeze(1) 
        z_to = model.sample_from_inference(codebook_to) #.squeeze(1)
        
        n = 0
        # ##### display 1 original audio
        # ##### display 1 original spec 
        spec_sampled_image_orig = self.log_images(spec_from)
        model.logger.experiment.add_image(f'zSpec_interpolation/{split}/epoch-{model.current_epoch}/step-{batch_idx}/{strategy}', spec_sampled_image_orig, global_step=n)         
        if self.args.vocoder !="":
            waves = self.spec_to_audio_to_st(spec_from, 22050, vocoder=self.melgan)
            waves = torch.from_numpy(waves['vocoder']).unsqueeze(1)
            model.logger.experiment.add_audio(f'zAudio_interpolation/{split}/epoch-{model.current_epoch}/step-{batch_idx}/{strategy}', waves, n, 22050)
        
        for v in torch.linspace(0, 1, 5):
            
            n = n+1            
            z = v * z_to + (1 - v) * z_from
            # pdb.set_trace()
            codes, _ = model.decode(z, strategy)    

            if self.args.reconstruct_spec !="":
                spec_sampled = self.codes_to_spec(codes, self.vqvae_model, model)
                spec_sampled_image = self.log_images(spec_sampled)
                model.logger.experiment.add_image(f'zSpec_interpolation/{split}/epoch-{model.current_epoch}/step-{batch_idx}/{strategy}', spec_sampled_image, global_step=n)            
            
            if self.args.vocoder !="":
                waves = self.spec_to_audio_to_st(spec_sampled, 22050, vocoder=self.melgan)
                waves = torch.from_numpy(waves['vocoder']).unsqueeze(1)
                model.logger.experiment.add_audio(f'zAudio_interpolation/{split}/epoch-{model.current_epoch}/step-{batch_idx}/{strategy}', waves, n, 22050)
                
            spec_sampled = []


        # ##### display last original audio
        # ##### display last original spec
        n = n+1
        spec_sampled_image_orig_last = self.log_images(spec_to)
        # print(spec_sampled_image_orig_last.shape)
        model.logger.experiment.add_image(f'zSpec_interpolation/{split}/epoch-{model.current_epoch}/step-{batch_idx}/{strategy}', spec_sampled_image_orig_last, global_step=n)     
        if self.args.vocoder !="":
            waves = self.spec_to_audio_to_st(spec_to, 22050, vocoder=self.melgan)
            waves = torch.from_numpy(waves['vocoder']).unsqueeze(1)
            model.logger.experiment.add_audio(f'zAudio_interpolation/{split}/epoch-{model.current_epoch}/step-{batch_idx}/{strategy}', waves, n, 22050)
            
    def codes_to_spec(self, codes, model, pl_module):
        # TODO I need to make this 53 and 5 as parameters or make them countable from other params. 
        # So if anything changes in settings this will change automatically.

        codes = codes[0].unsqueeze(0)  
        codes = pl_module.code_reader(codes, reverse=True)

        quantize_from_codebooks = model._vq_vae.get_codebook_entry(codes.reshape(-1), (1,5,53,256))
        reconstructions = model.decode(quantize_from_codebooks)

        return reconstructions
        
        
    def log_images(self, spec):
        x = spec
        
        # get to 0-1 for images
        x = (x + 1.0) / 2.0
        x= torch.clamp(x, 0.0, 1.0)

        # make grid
        return x.flip(dims=(2,)).squeeze(1) #torchvision.utils.make_grid(x.flip(dims=(2,)), nrow =1) #, normilize=True)

          

            
            
            


################################ LR CONTROL + mi, au and ppl logging and PRINTING #############################################


class callbeck_of_my_dreams(Callback):
    def __init__(self):
        self.decay_cnt = 0
        self.not_improved = 0
        self.decay_epoch = 5
        self.lr_decay = 0.5

    def on_validation_end(self, trainer, pl_module):

        with torch.no_grad():
            print("\rcalculating mutual_info", end="",flush=True)
            cur_mi = pl_module.calc_mi(pl_module.val_data)
        

            print("\rCalculating active units", end="",flush=True)    
            au, au_var = pl_module.calc_au(pl_module.val_data)
        
        

        print('\rEpoch: %d - loss: %.4f, kl: %.4f, recon: %.4f, nll: %.4f, ppl: %.4f, active_units: %d, mutual_info: %.4f' % (pl_module.current_epoch, 
                                                                                                                            pl_module.test_loss,
                                                                                                                            pl_module.kl_loss, 
                                                                                                                            pl_module.rec_loss, 
                                                                                                                            pl_module.nll, 
                                                                                                                            pl_module.ppl, 
                                                                                                                            au, cur_mi))
        
        
        pl_module.logger.experiment.add_scalar("metrics/mutual_info",  cur_mi, global_step=pl_module.current_epoch)
        pl_module.logger.experiment.add_scalar("metrics/active_units",  au, global_step=pl_module.current_epoch)
        pl_module.logger.experiment.add_scalar("metrics/ppl",  pl_module.ppl, global_step=pl_module.current_epoch)
        pl_module.logger.experiment.add_scalar("metrics/nll",  pl_module.nll, global_step=pl_module.current_epoch)
        pl_module.logger.experiment.add_scalar("metrics/starting_best_loss",  pl_module.best_loss, global_step=pl_module.current_epoch)
        
        
        if trainer.state.fn=="fit":
            
            pl_module.lr = pl_module.get_lr() # to make sure we have everything in sync and read lr form checkpoints correctly
            
                
            if pl_module.test_loss > pl_module.best_loss :

                self.not_improved += 1
                if self.not_improved >= self.decay_epoch and pl_module.current_epoch >=15:
                    
                    # ##############################
                    
                    # print("model did't improve for more than %d epochs so we load the best model ckpt %s" % (self.decay_epoch, 
                    #                                                                                         trainer.checkpoint_callback.best_model_path))
                    
                    # # Here we load model to another variable and then attach parts of it to running pl_module. There must be better way to do this but I didn't find it.
                    # # just loading checkpoint was breaking trainer rutine and I wasn't able to change lr and access optimisers.
                    # pl_module_ckpt = VAE.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, args = pl_module.args).to(pl_module.args.device)
                    # pl_module.encoder = pl_module_ckpt.encoder
                    # pl_module.decoder = pl_module_ckpt.decoder
                    # pl_module.best_loss = pl_module_ckpt.best_loss
                    # pl_module.pre_mi = pl_module_ckpt.pre_mi
                    # pl_module.kl_weight = pl_module_ckpt.kl_weight
                    
                    
                    # ##################################
                    
                    # pl_module.lr = pl_module.lr * self.lr_decay
                    # pl_module.set_lr(pl_module.lr)
                    
                    print("\rEpoch: %d - Best loss was: %.4f not_improved: %d and new lr to: %.4f\n" % (pl_module.current_epoch, 
                                                                                                  pl_module.best_loss,
                                                                                                  self.not_improved, 
                                                                                                  pl_module.lr, 
                                                                                                  ))
                    # self.not_improved = 0

                else:
                    print("\rEpoch: %d - Best loss: %.4f not_improved: %d and lr : %.4f\n" % (pl_module.current_epoch, 
                                                                                                  pl_module.best_loss,
                                                                                                  self.not_improved, 
                                                                                                  pl_module.lr, 
                                                                                                  ))
                
        
            else:
                
                self.not_improved = 0
                print("\rEpoch: %d - Best loss was: %.4f not_improved: %d lr %.4f setting best_loss %.4f\n" % (pl_module.current_epoch, 
                                                                                                pl_module.best_loss,
                                                                                                self.not_improved, 
                                                                                                pl_module.lr,
                                                                                                pl_module.test_loss 
                                                                                                ))
                
                pl_module.best_loss = pl_module.test_loss
                
            if pl_module.args.save_latent > 0 and pl_module.current_epoch <= pl_module.args.save_latent:
                visualize_latent(args, epoch, vae, "cuda", test_data)
                
                
        else:
            pass
            # print("\rCurrunt epoch: %d - Best loss was: %.4f lr %.4f \n" % (pl_module.current_epoch, 
            #                                                                     pl_module.best_loss,
            #                                                                     # pl_module.lr
            #                                                                    ))  #### here put everything you wanna print in time of evaluation and testing