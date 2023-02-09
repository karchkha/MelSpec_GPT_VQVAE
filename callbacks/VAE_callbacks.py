from pytorch_lightning.callbacks import Callback, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
# from data import MonoTextData
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

from utils import visualize_latent

####################### CALLBACKS ################################



#########################TEXT LOGGER for VAE ######################################        
        



class TextLogger(Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if self.args.reconstruct_spec!="":
            self.vqvae_model = LitVQVAE(num_embeddings = 128 , embedding_dim = 256)
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
    
    def reconstruct(self, model, data, strategy, device):
        data = data[0].unsqueeze(0)   #limit to only one sentance
        sentence = ""
        decoded_batch = model.reconstruct(data, strategy)
        for sent in decoded_batch:
            line = " ".join(str(sent)) + "  \n"
            sentence += line
        
        decoded_batch = decoded_batch[0]
        ### this will convert symbols to numbers and just give 0 to everything if thre will be <s> or </s> symbol.
        for i in range(len(decoded_batch)):
            decoded_batch[i]=model.vocab.word2id[decoded_batch[i]]
            if decoded_batch[i]>model.vocab_size - 3:
                for k in range(i,len(decoded_batch)):
                    decoded_batch[k]=0
                    i = i+1
                    
        decoded_batch = self.pad(decoded_batch, 265, 0)  # paddong to 265. TODO make all this automatic
            
        return sentence, decoded_batch
        
    def batch_to_sentence(self, model, data):
        sentence = ""

        batch_size, sent_size = data.size()
        batch_size = 1 #### limit to only 1 sentance
        
        decoded_batch = [[] for _ in range(batch_size)]
        
        for i in range(batch_size):
            for j in range (1, sent_size-1):
              decoded_batch[i].append(model.vocab.id2word(data[i, j].item()))
        
        for sent in decoded_batch:
            line = " ".join(str(sent)) + "  \n"
            sentence += line
        
        return sentence
        
    def sample_from_z(self, model, z, strategy):
        sentence = ""
        decoded_batch = model.decode(z, strategy)

        for sent in decoded_batch:
            line = " ".join(str(sent)) + "  \n"
            sentence += line
        
        decoded_batch = decoded_batch[0]
        ### this will convert symbols to numbers and just give 0 to everything if thre will be start or end symbol.
        for i in range(len(decoded_batch)):
            decoded_batch[i]=model.vocab.word2id[decoded_batch[i]]
            if decoded_batch[i]>model.vocab_size - 3:
                for k in range(i,len(decoded_batch)):
                    decoded_batch[k]=0
                    i = i+1
        decoded_batch = self.pad(decoded_batch, 265, 0) # paddong to 265. TODO make all this automatic
        
        return sentence, decoded_batch
        
    def spec_to_audio_to_st(self, x, sample_rate, vocoder=None):
        # audios are in [-1, 1], making them in [0, 1]
        spec = (x.data.squeeze(0) + 1) / 2
    
        out = {}
        if vocoder:
            # (L,) <- wave: (1, 1, L).squeeze() <- spec: (1, F, T)
            wave_from_vocoder = vocoder(spec).squeeze().cpu().numpy()
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
            

    def log_everything(self, pl_module, batch, batch_idx, split='train'):

        logger = type(pl_module.logger)
        
        ###################### original spectrogram ####################################
        original_spec = self.get_input(batch)[0].unsqueeze(0) # limiting to 1-st image
        original_spec = self.log_images(original_spec)
        pl_module.logger.experiment.add_image(f'{split}/epoch-{pl_module.current_epoch}/step-{batch_idx}/original', original_spec, global_step=batch_idx)

        if self.args.vocoder !="":
            orig_file_path = '../AV_Datasets/VAS/'+batch['label'][0]+'/videos/'+batch['file_path_'][0][-19:-8]+'.mp4'
            if os.path.isfile(orig_file_path):
                try:
                    waves, _ = librosa.load(orig_file_path, sr=22050)
                    # waves, _ = sf.read(orig_file_path, samplerate=2250 )
                    # waves = wave.open(orig_file_path,'r')
                    waves = torch.from_numpy(waves).unsqueeze(1)
                    pl_module.logger.experiment.add_audio(f'{split}/epoch-{pl_module.current_epoch}/step-{batch_idx}/original_audio', waves, batch_idx, 22050)
                except:
                    pass

        ###################### original codebook logging as sentence ####################################
        codes_from_data = pl_module.get_input(batch)
        original = self.batch_to_sentence(pl_module, codes_from_data)       
        pl_module.logger.experiment.add_text(f'{split}/epoch-{pl_module.current_epoch}/step-{batch_idx}/original', original, global_step=batch_idx)
        
        
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.no_grad():

            if self.args.test_interpolation:
                self.audio_interpolation(model = pl_module, batch = batch, batch_idx = batch_idx, split = split)
            
            ############# codebook reconstruction ##########################
            text_reconstructed, codes = self.reconstruct(pl_module, codes_from_data, self.args.decoding_strategy, self.args.device)
            pl_module.logger.experiment.add_text(f'{split}/epoch-{pl_module.current_epoch}/step-{batch_idx}/reconstraction', text_reconstructed, global_step=batch_idx)
            
            # print(codes)
            
            
            ############# spectrum reconstruction ##########################
            if self.args.reconstruct_spec !="":
                spec_reconstructions = self.codes_to_spec(codes, self.vqvae_model)
                spec_reconstructions_image = self.log_images(spec_reconstructions)
                pl_module.logger.experiment.add_image(f'{split}/epoch-{pl_module.current_epoch}/step-{batch_idx}/reconstraction', spec_reconstructions_image, global_step=batch_idx)
            
            
            if self.args.vocoder !="":
                waves = self.spec_to_audio_to_st(spec_reconstructions, 22050, vocoder=self.melgan)
                waves = torch.from_numpy(waves['vocoder']).unsqueeze(1)
                pl_module.logger.experiment.add_audio(f'{split}/epoch-{pl_module.current_epoch}/step-{batch_idx}/reconstraction_audio', waves, batch_idx, 22050)
            

            ################# Sampling form prior ####################
            
            z = pl_module.sample_from_prior(1)
            
            ###### codebook sampled ###############
            sampled_from_prior, codes = self.sample_from_z(pl_module, z, self.args.decoding_strategy)
            pl_module.logger.experiment.add_text(f'{split}/epoch-{pl_module.current_epoch}/step-{batch_idx}/sampled_from_prior', sampled_from_prior, global_step=batch_idx)
            
            ###### spectrogram sampled ###############
            if self.args.reconstruct_spec !="":
                spec_sampled = self.codes_to_spec(codes, self.vqvae_model)
                spec_sampled_image = self.log_images(spec_sampled)
                pl_module.logger.experiment.add_image(f'{split}/epoch-{pl_module.current_epoch}/step-{batch_idx}/sample_from_prior', spec_sampled_image, global_step=batch_idx)            

            if self.args.vocoder !="":
                waves = self.spec_to_audio_to_st(spec_sampled, 22050, vocoder=self.melgan)
                waves = torch.from_numpy(waves['vocoder']).unsqueeze(1)
                pl_module.logger.experiment.add_audio(f'{split}/epoch-{pl_module.current_epoch}/step-{batch_idx}/sample_from_prior_audio', waves, batch_idx, 22050)
                
                


        if is_train:
            pl_module.train()
            
    def audio_interpolation(self, model, batch, batch_idx, split):

        spec = batch['image']
        # from_middle = spec.shape[2]//2
        # spec_from = spec[0,:,from_middle:from_middle+160].unsqueeze(0).unsqueeze(0).to(self.args.device)
        # spec_to = spec[1,:,from_middle:from_middle+160].unsqueeze(0).unsqueeze(0).to(self.args.device)

        spec_from = spec[0,:,:160].unsqueeze(0).unsqueeze(0).to(self.args.device)
        spec_to = spec[1,:,:160].unsqueeze(0).unsqueeze(0).to(self.args.device)
            
        ###### get codebook of those data  ###
        

        # codebook_from = self.vqvae_model.encode(spec_from)
        # _, _, info = self.vqvae_model._vq_vae(codebook_from)
        # codebook_from = info[2].squeeze(1).unsqueeze(0)
        

        # codebook_to = self.vqvae_model.encode(spec_to)
        # _, _, info = self.vqvae_model._vq_vae(codebook_to)
        # codebook_to = info[2].squeeze(1).unsqueeze(0)     
 
 
        ### reading codes from database directly because specs are not converted well when converting only parts of it :))
        codes_from_data = model.get_input(batch)
        codebook_from = codes_from_data[0].unsqueeze(0)
        codebook_to = codes_from_data [5].unsqueeze(0)        
        
        
        ############ attach start and end symbols ############
        
        # starts = torch.full((1,1),model.vocab.word2id['<s>'], dtype=torch.int64).to(self.args.device)
        # ends = torch.full((1,1),model.vocab.word2id['</s>'], dtype=torch.int64).to(self.args.device)
        
        # codebook_from = torch.cat((starts,codebook_from),1)    
        # codebook_from = torch.cat((codebook_from,ends),1).to(memory_format=torch.contiguous_format)   

        # codebook_to = torch.cat((starts,codebook_to),1)    
        # codebook_to = torch.cat((codebook_to,ends),1).to(memory_format=torch.contiguous_format)     
        
        ###### get z of those data  ######
        
        # z_from, _ = model.encode(codebook_from)                     ############################????????????????????????????????????????????????#@@@@@@@@@@@@@@
        # z_to, _ = model.encode(codebook_to)                       ########## might have to use: self.sample_from_inference(x).squeeze(1) @########################
        
        z_from = model.sample_from_inference(codebook_from).squeeze(1) 
        # print(z_from)
        z_to = model.sample_from_inference(codebook_to).squeeze(1)
        
        n = 0
        # ##### display 1 original audio
        # ##### display 1 original spec 
        spec_sampled_image_orig = self.log_images(spec_from)
        model.logger.experiment.add_image(f'{split}/epoch-{model.current_epoch}/step-{batch_idx}/spec_interpolation', spec_sampled_image_orig, global_step=n)         
        if self.args.vocoder !="":
            waves = self.spec_to_audio_to_st(spec_from, 22050, vocoder=self.melgan)
            waves = torch.from_numpy(waves['vocoder']).unsqueeze(1)
            model.logger.experiment.add_audio(f'{split}/epoch-{model.current_epoch}/step-{batch_idx}/audio_interpolation', waves, n, 22050)
        
        for v in torch.linspace(0, 1, 10):
            
            n = n+1            
            z = v * z_to + (1 - v) * z_from
            # print(z)
            codes = model.decode(z, self.args.decoding_strategy)     #      _, codes = self.sample_from_z(model, z, self.args.decoding_strategy) 

            codes = codes[0]
            ### this will convert symbols to numbers and just give 0 to everything if thre will be <s> or </s> symbol.
            for i in range(len(codes)):
                codes[i]=model.vocab.word2id[codes[i]]
                if codes[i]>model.vocab_size - 3:
                    for k in range(i,len(codes)):
                        codes[k]=0
                        i = i+1
            
            codes = self.pad(codes, 265, 0)
            
            # print(codes)

            if self.args.reconstruct_spec !="":
                spec_sampled = self.codes_to_spec(codes, self.vqvae_model)
                spec_sampled_image = self.log_images(spec_sampled)
                model.logger.experiment.add_image(f'{split}/epoch-{model.current_epoch}/step-{batch_idx}/spec_interpolation', spec_sampled_image, global_step=n)            
            
            if self.args.vocoder !="":
                waves = self.spec_to_audio_to_st(spec_sampled, 22050, vocoder=self.melgan)
                waves = torch.from_numpy(waves['vocoder']).unsqueeze(1)
                model.logger.experiment.add_audio(f'{split}/epoch-{model.current_epoch}/step-{batch_idx}/audio_interpolation', waves, n, 22050)
                
            spec_sampled = []


        # ##### display last original audio
        # ##### display last original spec
        n = n+1
        spec_sampled_image_orig_last = self.log_images(spec_to)
        # print(spec_sampled_image_orig_last.shape)
        model.logger.experiment.add_image(f'{split}/epoch-{model.current_epoch}/step-{batch_idx}/spec_interpolation', spec_sampled_image_orig_last, global_step=n)     
        if self.args.vocoder !="":
            waves = self.spec_to_audio_to_st(spec_to, 22050, vocoder=self.melgan)
            waves = torch.from_numpy(waves['vocoder']).unsqueeze(1)
            model.logger.experiment.add_audio(f'{split}/epoch-{model.current_epoch}/step-{batch_idx}/audio_interpolation', waves, n, 22050)
            
    def codes_to_spec(self, codes, model):
        # TODO I need to make this 53 and 5 as parameters or make them countable from other params. 
        # So if anything changes in settings this will change automatically.
        # print(codes)
        codes = torch.tensor(codes).squeeze(0).to(self.args.device)
        codes = codes.view(53, 5)
        codes = codes.permute(1,0)
        codes = codes[:,:10]
        # print(codes)
        codes = torch.flatten(codes)
        # print(codes)

        quantize_from_codebooks = model._vq_vae.get_codebook_entry(codes, (1,5,10,256))
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
                    
                    ##############################
                    
                    print("model did't improve for more than %d epochs so we load the best model ckpt %s" % (self.decay_epoch, 
                                                                                                            trainer.checkpoint_callback.best_model_path))
                    
                    # Here we load model to another variable and then attach parts of it to running pl_module. There must be better way to do this but I didn't find it.
                    # just loading checkpoint was breaking trainer rutine and I wasn't able to change lr and access optimisers.
                    pl_module_ckpt = VAE.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, args = pl_module.args).to(pl_module.args.device)
                    pl_module.encoder = pl_module_ckpt.encoder
                    pl_module.decoder = pl_module_ckpt.decoder
                    pl_module.best_loss = pl_module_ckpt.best_loss
                    pl_module.pre_mi = pl_module_ckpt.pre_mi
                    pl_module.kl_weight = pl_module_ckpt.kl_weight
                    
                    
                    ##################################
                    
                    pl_module.lr = pl_module.lr * self.lr_decay
                    pl_module.set_lr(pl_module.lr)
                    
                    print("\rEpoch: %d - Best loss was: %.4f not_improved: %d and new lr to: %.4f\n" % (pl_module.current_epoch, 
                                                                                                  pl_module.best_loss,
                                                                                                  self.not_improved, 
                                                                                                  pl_module.lr, 
                                                                                                  ))
                    self.not_improved = 0
                    # pl_module.best_loss = pl_module.test_loss  # Best loss will be taken from checkpoint!
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
            
            print("\rCurrunt epoch: %d - Best loss was: %.4f lr %.4f \n" % (pl_module.current_epoch, 
                                                                                pl_module.best_loss,
                                                                                pl_module.lr
                                                                                ))  #### here put everything you wanna print in time of evaluation and testing
        
        

        
        

        
        