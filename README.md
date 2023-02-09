# MelSpec_GPT_VQVAE

Audio Generation model working with GPT3 and VQVAE compressed representation of MelSpectrograms


# Introduction

Welcome to the MelSpec_GPT_VQVAE repository! 

The project investigates the application of Transformer-based GPT3 models as a generative method for audio generation. The audio signal is represented as 128 distinct tokens, which are compressed versions of mel-spectrograms. The study builds upon the Veqtor quantized encoder/decoder architecture described my other repository: https://github.com/karchkha/MelSpec_VQVAE, which in turn is partially taken from the paper "Taming Visually Guided Sound Generation" (https://arxiv.org/abs/2110.08791) and its corresponding repository (https://github.com/v-iashin/SpecVQGAN).

The repository consists of two parts/directions:

The first part focuses on the evaluation of GPT3 as a generative model when conditioned on audio class categories, such as "dog barking," "baby crying," "sneezing," and "fireworks," among others. This part of the project serves as a preliminary examination of the feasibility of audio generation, with the potential for future studies on time-aligned/controlled audio and music generation utilizing these models.

The second part of the project involves the utilization of a Variational Autoencoder (VAE), with GPT3 modules in both the encoder and decoder sides. The encoder consists of an unmasked transformer that compresses the audio token sequence into a 1024-dimensional latent representation, which is then combined with a variance and a noise term to serve as the conditioning input for the decoder. The decoder comprises a masked autoregressive transformer that aims to reconstruct the whole token sequence, which is then translated back to a mel-spectrogram using a pretrained VQVAE decoder, followed by audio generation using a Mel_GAN vocoder. Both VQVAE amd Mel_GAN models are pretrained here and are not trained in this repository. For the VQVAE (pre-)training, plase see my other repository: (https://github.com/karchkha/MelSpec_VQVAE).

The second part of the project investigates a VAE generative model for the task of incorporating distributed latent representations of entire audio sequences. Unlike the standard autoregresive tranformers, which generates audio sequences one token at a time, this model also considers global audio representation and holistic properties of audio sequences such as audio content, style, and high-level features. The conversion of prior audio sequences into a compact representation via the encoder enables the generation of diverse and coherent audio sequences through deterministic decoder. By exploring the latent space, it is possible to create novel audio that interpolate between known audio sequences. This project presents techniques to address the difficult learning problem traditionally posed by VAE sequence models, and aims to demonstrate the effectiveness of the model in imputing missing audio tokens in the future.

# Installation

To install MelSpec_VQVAE, follow these steps:

Clone the repository to your local machine
```bash
$ git clone https://github.com/karchkha/MelSpec_VQVAE
```

To run the code in this repository, you will need:

python 3.8.10 

Navigate to the project directory and install the required dependencies

```bash
$ pip -r install requirements.txt
```

# Training GPT CLASS

The training procedure for the first component of the project, where GPT-3 is conditioned on audio class categories, is governed by the following parameters:

* `dataset`: The dataset to be used for training the model. This argument is required.
* `experiment`: The name of the experiment. This argument is also required and will be used to name the checkpoints and tensorboard logs.
* `train`: Indicates whether to start the training process. If set to 1, the training process will begin, if set to 0, the training process will not begin.
* `resume`: If set, the training process will resume from the specified checkpoint.
* `workers`: Number of workers for data processing.
* `eval`: If set to 1, the model will be evaluated, if set to 0, the model will not be evaluated.
* `test`: If set to 1, the model will be tested, if set to 0, the model will not be tested.
* `logging_frequency`: The number of steps for text logging.
* `reconstruct_spec`: The model checkpoint path for mel-spectrogram reconstruction.
* `vocoder`: The model checkpoint for the vocoder for audio reconstruction.

The training process can be started by passing the appropriate values for these arguments to the script. The model will be trained on the specified dataset for the specified number of epochs and the checkpoints will be saved in the directory specified by experiment. The details of the model architecture can be found in the config directory in the file config_GPT_{dataset_name}.py. This file also allows for changes to be made to the architecture if desired.

For Example the running with default velues would look like:
```bash

$ python GPT_train.py \
          --experiment {experiment_name} \
          --dataset {dataset_name} \
          --logging_frequency 200 \
          --workers 2 \
          --train 1 \
          --vocoder 'vocoder/logs/vggsound/' \
          --eval 0 \
          --test 0 \
          --reconstruct_spec "lightning_logs/2021-06-06T19-42-53_vas_codebook.pt" \
          --resume "lightning_logs/{experiment_name}/checkpoints/version_{version_number}/last.ckpt"
```

# Training GPT VAE

The training procedure VAE component of the project is governed by the following parameters:

* `dataset`: The dataset to be used for training the model. This argument is required.
* `experiment`: The name of the experiment. This argument is also required and will be used to name the checkpoints and tensorboard logs.
* `train`: A flag to start the training process. The default value is False.
* `resume`: A checkpoint file to resume training from. The default value is None.
* `workers`: The number of workers to use for data loading. The default value is 4.
* `eval`: A flag to evaluate the model. The default value is False.
* `test`: A flag to test the model. The default value is False.
* `logging_frequency`: The number of steps for text logging.
* `reconstruct_spec`: The model checkpoint path for mel-spectrogram reconstruction.
* `vocoder`: The model checkpoint for the vocoder for audio reconstruction.
* `load_path`:: The path to the model checkpoint from where we take pretrained encoder on the 2nd stage of the training. The type of the argument is str. The default value is an empty string.
* `test_interpolation`: Whether to test and visualize an interpolation between 2 sounds. The type of the argument is int. The default value is False.
* `warm_up`: This argument takes an integer as input, and specifies the number of annealing epochs for a process. The default value is set to 10.
* `kl_start`: This argument takes a float as input, and specifies the starting KL weight for a process. The default value is set to 1.0.
* `beta`:  This argument takes a float as input, and determines if the model is an autoencoder (AE) beta = 0.0 or a variational autoencoder (VAE) beta = 1.0. The default value is set to 1.0.
* `fb`: this argument controls the regularisation for KL loss. particularly the KL Thresholding / Free Bits (FB). FB is a modification of the KL-Divergence term used in the ELBO loss, which is commonly used in variational autoencoders. The FB technique replaces the KL term with a hinge loss term, which maximizes each component of the original KL term with a constant. The hinge loss is defined as: max[threshold; KL(q(z_i|x) || p(z_i))], where threshold is the target rate, and z_i is the i-th dimension in the latent variable z. In the code argument fb = 0 means no fb; fb = 1 means fb and fb = 2 means max(target_kl, kl) for each dimension.
* `target_kl`: This argument takes a float as input, and sets the target KL of the free bits trick. The default value is set to -1, which means the target KL is not set.


The training process can be started by passing the appropriate values for these arguments to the script. The model will be trained on the specified dataset for the specified number of epochs and the checkpoints will be saved in the directory specified by experiment. The details of the model architecture can be found in the config directory in the file config_GPT_VAE_{dataset_name}.py. This file also allows for changes to be made to the architecture if desired.

To avoid the problem of collapsing posterior, the training process is split into two stages. First, we train a simple autoencoder For Example the running with default velues would look like:

```bash
$ python GPT_VAE_train.py \
          --experiment {experiment_name_1} \
          --dataset {dataset_name} \
          --beta 0 \
          --logging_frequency 200 \
          --workers 8 \
          --train 1 \
          --reconstruct_spec "lightning_logs/2021-06-06T19-42-53_vas_codebook.pt" \
          --vocoder 'vocoder/logs/vggsound/' \
          --eval 0 \
          --test 0 \
          --test_interpolation 1 \
          --resume "lightning_logs/{experiment_name_1}/checkpoints/version_{version_number}/last.ckpt" \

```

Please note that setting the value of beta to 0 will result in the above model becoming a simple autoencoder (AE), rather than a variational autoencoder (VAE).

During the second stage of training, a variational autoencoder is trained with its encoder initialized by the encoder of the simple autoencoder trained in stage one. To further prevent the encoder from collapsing, the methods of KL weight annealing and Free Bits (FB) are employed. An example of running the model with default values would look like this:

```bash
$ python GPT_VAE_train.py \
    --experiment {experiment_name_2} \
    --dataset {dataset_name} \
    --kl_start 0 \
    --warm_up 10 \
    --target_kl 8 \
    --fb 2 \
    --logging_frequency 200 \
    --workers 8 \
    --train 1 \
    --eval 0 \
    --test 0 \
    --load_path "lightning_logs/{experiment_name_1}/checkpoints/version_{version_number}/last.ckpt" \
    --reconstruct_spec "lightning_logs/2021-06-06T19-42-53_vas_codebook.pt" \
    --vocoder 'vocoder/logs/vggsound/' \
    --resume "lightning_logs/{experiment_name_2}/checkpoints/version_{version_number}/last.ckpt" \
    --test_interpolation 1
```

The performance of the model can be monitored using Tensorboard. During training, the loss and other metrics, as well as the mel-spectrogram images and reconstructed audio, are logged and can be viewed in Tensorboard.

```bash
$ tensorboard --logdir lightning_logs --bind_all
```
There are few differnt models but I ended up using big_model_attn_gan.py. This one works more-less well. Takes very long to train though.

# Data

In this project, the VAS data is used [VAS](https://github.com/PeihaoChen/regnet#download-datasets).
VAS can be downloaded directly using the link provided in the [RegNet](https://github.com/PeihaoChen/regnet#download-datasets) repository.

Another way, VAS data with ready Mel_spectrograms can be downloaded by running below code:
```bash
cd ./data
# ~7GB we only download spectrograms
bash ./download_vas_features.sh
```
# Vocoder

For reconstructing Mel_spectrograms to Audio the Mel_GAN vocoder is used. The pretrained vocoder from: https://github.com/v-iashin/SpecVQGAN/tree/main/vocoder/logs/vggsound is present in the repository in folder: vocoder/logs/vggsound folder. 

