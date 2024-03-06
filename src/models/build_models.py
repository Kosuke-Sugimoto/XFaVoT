import os
import os.path as osp

import copy
import math

from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import DataParallel
from StarGANv2VC.models import Generator as AudioGenerator, Discriminator as AudioDiscriminator, StyleEncoder as AudioStyleEncoder, MappingNetwork
from starganv2.core.model import Generator as ImageGenerator, Discriminator as ImageDiscriminator, StyleEncoder as ImageStyleEncoder
from starganv2.core.wing import FAN

def build_model(args, F0_model, ASR_model):
    audio_generator = DataParallel(AudioGenerator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf, F0_channel=args.F0_channel), device_ids=[0, 1])
    mapping_network = DataParallel(MappingNetwork(args.latent_dim, args.style_dim, args.audio_num_domains, hidden_dim=args.max_conv_dim), device_ids=[0, 1])
    audio_style_encoder = DataParallel(AudioStyleEncoder(args.dim_in, args.style_dim, args.image_num_domains, args.max_conv_dim), device_ids=[0, 1])
    audio_discriminator = DataParallel(AudioDiscriminator(args.dim_in, args.audio_num_domains, args.max_conv_dim, args.n_repeat), device_ids=[0, 1])
    # generator_ema = copy.deepcopy(generator)
    # mapping_network_ema = copy.deepcopy(mapping_network)
    # style_encoder_ema = copy.deepcopy(style_encoder)
    image_generator = DataParallel(ImageGenerator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf), device_ids=[0, 1])
    image_style_encoder = DataParallel(ImageStyleEncoder(args.dim_in, args.style_dim, args.image_num_domains, args.max_conv_dim), device_ids=[0, 1])
    image_discriminator = DataParallel(ImageDiscriminator(args.dim_in, args.image_num_domains, args.max_conv_dim), device_ids=[0, 1])
        
    nets = Munch(audio_generator=audio_generator,
                 mapping_network=mapping_network,
                 audio_style_encoder=audio_style_encoder,
                 audio_discriminator=audio_discriminator,
                 f0_model=F0_model,
                 asr_model=ASR_model,
                 image_generator=image_generator,
                 image_style_encoder=image_style_encoder,
                 image_discriminator=image_discriminator)
    
    # nets_ema = Munch(generator=generator_ema,
    #                  mapping_network=mapping_network_ema,
    #                  style_encoder=style_encoder_ema)

    # return nets, nets_ema
    return nets