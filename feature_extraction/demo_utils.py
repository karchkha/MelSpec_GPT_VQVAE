'''
The code is partially borrowed from:
https://github.com/v-iashin/video_features/blob/861efaa4ed67/utils/utils.py
and
https://github.com/PeihaoChen/regnet/blob/199609/extract_audio_and_video.py
'''
import os
import shutil
import subprocess
from glob import glob
from pathlib import Path
from typing import Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


from feature_extraction.extract_mel_spectrogram import get_spectrogram

def which_ffmpeg() -> str:
    '''Determines the path to ffmpeg library

    Returns:
        str -- path to the library
    '''
    result = subprocess.run(['which', 'ffmpeg'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ffmpeg_path = result.stdout.decode('utf-8').replace('\n', '')
    return ffmpeg_path

def which_ffprobe() -> str:
    '''Determines the path to ffprobe library

    Returns:
        str -- path to the library
    '''
    result = subprocess.run(['which', 'ffprobe'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ffprobe_path = result.stdout.decode('utf-8').replace('\n', '')
    return ffprobe_path


def check_video_for_audio(path):
    assert which_ffprobe() != '', 'Is ffmpeg installed? Check if the conda environment is activated.'
    cmd = f'{which_ffprobe()} -loglevel error -show_entries stream=codec_type -of default=nw=1 {path}'
    result = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    result = result.stdout.decode('utf-8')
    print(result)
    return 'codec_type=audio' in result



def extract_melspectrogram(in_path: str, sr: int, duration: int = 10, tmp_path: str = './tmp') -> np.ndarray:
    '''Extract Melspectrogram similar to RegNet.'''
    assert which_ffmpeg() != '', 'Is ffmpeg installed? Check if the conda environment is activated.'
    # assert in_path.endswith('.mp4'), 'The file does not end with .mp4. Comment this if expected'
    # create tmp dir if doesn't exist
    os.makedirs(tmp_path, exist_ok=True)

    # Extract audio from a video if needed
    if in_path.endswith('.wav'):
        audio_raw = in_path
    else:
        audio_raw = os.path.join(tmp_path, f'{Path(in_path).stem}.wav')
        cmd = f'{which_ffmpeg()} -i {in_path} -hide_banner -loglevel panic -f wav -vn -y {audio_raw}'
        subprocess.call(cmd.split())

    # Extract audio from a video
    audio_new = os.path.join(tmp_path, f'{Path(in_path).stem}_{sr}hz.wav')
    cmd = f'{which_ffmpeg()} -i {audio_raw} -hide_banner -loglevel panic -ac 1 -ab 16k -ar {sr} -y {audio_new}'
    subprocess.call(cmd.split())

    length = int(duration * sr)
    audio_zero_pad, spec = get_spectrogram(audio_new, save_dir=None, length=length, save_results=False)

    # specvqgan expects inputs to be in [-1, 1] but spectrograms are in [0, 1]
    spec = 2 * spec - 1

    return spec


def show_grid(imgs):
    print('Rendering the Plot with Frames Used in Conditioning')
    figsize = ((imgs.shape[1] // 228 + 1) * 5, (imgs.shape[2] // 228 + 1) * 5)
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=figsize)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    return fig






if __name__ == '__main__':
    # if empty, it wasn't found
    print(which_ffmpeg())
