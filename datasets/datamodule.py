import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset 
from .vas import VASSpecs
from .vggsound import VGGSoundSpecs




class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, spec_dir_path, num_workers=None,  mel_num = None, spec_len = None, spec_crop_len = None, random_crop = None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else batch_size*2

        self.spec_dir_path = spec_dir_path
        self.mel_num = mel_num
        self.spec_len = spec_len
        self.spec_crop_len = spec_crop_len
        self.random_crop = random_crop

    def setup(self, stage=None):

        if "vggsound" in self.spec_dir_path:
            print("In VGGSound")

            self.train_dataset = VGGSoundSpecs( 'train', 
                            spec_dir_path = self.spec_dir_path, 
                            mel_num = self.mel_num, 
                            spec_len = self.spec_len, 
                            spec_crop_len = self.spec_crop_len, 
                            random_crop = self.random_crop)
            self.val_dataset = VGGSoundSpecs( 'valid', 
                            spec_dir_path = self.spec_dir_path, 
                            mel_num = self.mel_num, 
                            spec_len = self.spec_len, 
                            spec_crop_len = self.spec_crop_len, 
                            random_crop = self.random_crop)
            self.test_dataset = VGGSoundSpecs( 'test', 
                             spec_dir_path = self.spec_dir_path, 
                             mel_num = self.mel_num, 
                             spec_len = self.spec_len, 
                             spec_crop_len = self.spec_crop_len, 
                             random_crop = self.random_crop)

        elif "vas" in self.spec_dir_path:
            print("In VAS")
            self.train_dataset = VASSpecs( 'train', 
                            spec_dir_path = self.spec_dir_path, 
                            mel_num = self.mel_num, 
                            spec_len = self.spec_len, 
                            spec_crop_len = self.spec_crop_len, 
                            random_crop = self.random_crop)
            self.val_dataset = VASSpecs( 'valid', 
                            spec_dir_path = self.spec_dir_path, 
                            mel_num = self.mel_num, 
                            spec_len = self.spec_len, 
                            spec_crop_len = self.spec_crop_len, 
                            random_crop = self.random_crop)
            # self.test_dataset = VASSpecs( 'test', 
            #                  spec_dir_path = self.spec_dir_path, 
            #                  mel_num = self.mel_num, 
            #                  spec_len = self.spec_len, 
            #                  spec_crop_len = self.spec_crop_len, 
            #                  random_crop = self.random_crop)



    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=self.worker_init_fn,
                          drop_last = True, shuffle=True)
        
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=self.worker_init_fn, drop_last = True)

    def val_dataloader_shuffled(self): ### I added this just for an experiment
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=self.worker_init_fn, drop_last = True, shuffle=True)
        
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=self.worker_init_fn, drop_last = True)

    @staticmethod
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)
        