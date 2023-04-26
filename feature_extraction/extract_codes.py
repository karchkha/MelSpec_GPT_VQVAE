import sys
sys.path.append('..' )

import albumentations
import os
from vqvae.big_model_attn_gan import LitVQVAE
import torch
import argparse
# import os
from glob import glob
import numpy as np

class Crop(object):

    def __init__(self, cropped_shape=None, random_crop=False):
        self.cropped_shape = cropped_shape
        if cropped_shape is not None:
            mel_num, spec_len = cropped_shape
            if random_crop:
                self.cropper = albumentations.RandomCrop
            else:
                self.cropper = albumentations.CenterCrop
            self.preprocessor = albumentations.Compose([self.cropper(mel_num, spec_len)])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __call__(self, item):
        item = self.preprocessor(image=item)['image']
        return item

def get_codes(mel_path, device, spec_crop_len, model, transforms, folder_name='codes_10s'):
    # print("\r",mel_path, end="", flush = True)
    save_dir = os.path.dirname(os.path.dirname(mel_path))
    audio_name = os.path.basename(mel_path).split('.')[0]
    isFile = os.path.isfile(os.path.join(save_dir + "/" + folder_name, audio_name + '_code.npy'))
    
    if not isFile:    
        try:
            print("\rworking on",mel_path, end="", flush = True)
            mel = np.load(mel_path).astype(np.float32) #this might need to be changed later with VQVAE 1024 (???)
            # mel = mel[:,:spec_crop_len]
            mel = transforms(mel)
            mel = 2 * mel - 1
            
            mel_tensor = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(device)
            os.makedirs(save_dir+ "/" + folder_name, exist_ok=True)
            
            y = model.encode(mel_tensor)
            loss, quantize, info = model._vq_vae(y)
            codes = info[2].reshape( quantize.shape[2], quantize.shape[3])
            
            y_np=codes.cpu().detach().numpy()  
            
            
            # np.save(P.join(save_dir, audio_name + '_mel.npy'), mel_spec)
            np.save(os.path.join(save_dir + "/" + folder_name, audio_name + '_code.npy'), y_np)
        except:
            print(mel_path, "is damaged")
    else:
        print("\rfile exists:",mel_path, end="", flush = True)


if __name__ == '__main__':
    paser = argparse.ArgumentParser()
    
    # paser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
    paser.add_argument("-i", "--input_dir", default="data/vas/features")
    paser.add_argument("-m", "--model_dir", default="lightning_logs/2021-06-06T19-42-53_vas_codebook.pt")
    paser.add_argument("-emb_dim", "--embedding_dim", default=256)
    paser.add_argument("-n_e", "--num_embeddings", type=int, default=128)
    paser.add_argument("-crop", "--spec_crop_len", default=848)
    args = paser.parse_args()
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec_crop_len = args.spec_crop_len

    input_dir = "../" + args.input_dir
    model_dir = "../" + args.model_dir
    embedding_dim = args.embedding_dim
    num_embeddings = args.num_embeddings
    
    
    model = LitVQVAE(num_embeddings, embedding_dim )
    model.load_state_dict(torch.load(model_dir))
    model.eval().to(device)
    transforms = Crop([80, spec_crop_len], False)
    
    folders = os.listdir(input_dir)

    if "vggsound" in input_dir:
        print("In VGGSound")

        for folder in folders:
            if folder == "melspec_10s_22050hz":
                mel_dir = input_dir + "/" + folder

                mel_paths = glob(os.path.join(mel_dir, "*.npy"))

                mel_paths.sort()

                for mel_path in mel_paths:
                    get_codes(mel_path, device, spec_crop_len, model, transforms)   
                    # break    #### comment this out if this is running and not damaging you files
            else:
                pass
    
    elif "vas" in input_dir:
        print("In VAS")

        for folder in folders:
            mel_dir = input_dir + "/" + folder + "/" + "melspec_10s_22050hz"

            mel_paths = glob(os.path.join(mel_dir, "*.npy"))

            mel_paths.sort()

            for mel_path in mel_paths:
                get_codes(mel_path, device, spec_crop_len, model, transforms)
                break #### comment this out if this is running and not damaging you files