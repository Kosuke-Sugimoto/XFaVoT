import os
import wandb
import torch
import json
import numpy as np
from torch import nn
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from scipy.io.wavfile import write
from torch.nn import DataParallel
from BigVGAN.env import AttrDict
from BigVGAN.models import BigVGAN as Vocoder
from BigVGAN.inference_e2e import scan_checkpoint, load_checkpoint
from collections import defaultdict
from starganv2.core.solver import compute_d_loss as compute_image_d_loss, compute_g_loss as compute_image_g_loss
from StarGANv2VC.losses import compute_d_loss as compute_audio_d_loss, compute_g_loss as compute_audio_g_loss
from src.datasets.ADatasetUtils import MAX_WAV_VALUE

class Trainer(object):
    
    def __init__(
        self,
        args,
        model,
        optimizer,
        scaler,
        train_dataloader,
        val_dataloader,
        initial_steps=0,
        initial_epochs=0,
        device=torch.device("cuda"),
    ):
        self.model = model
        self.args = args
        self.optimizer = optimizer
        self.scaler = scaler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.steps = initial_steps
        self.epochs = initial_epochs
        self.device = device
        self.wandb_run_name = datetime.now().strftime("%Y%m%d_%H%M")
        
        h = None
        with Path("BigVGAN/exp/config.json").open() as f:
            data = f.read()
        json_config = json.loads(data)
        h = AttrDict(json_config)
        self.vocoder = Vocoder(h).to(self.device)
        state_dict_g = load_checkpoint("BigVGAN/exp/g_05000000.zip", self.device)
        self.vocoder.load_state_dict(state_dict_g["generator"])
        self.vocoder.eval()
        self.vocoder.remove_weight_norm()
        self.vocoder = DataParallel(self.vocoder, device_ids=[0, 1])

        wandb.init(
            project = self.args.project_name,
            name = self.wandb_run_name,
            config = dict(self.args)
        )

        self.artifact = wandb.Artifact(
            name=self.wandb_run_name,
            type="model_weight",
            description="model weights"
        )

    @torch.no_grad()
    def log_audio(
        self,
        mel,
        ref_mel,
        ref_img,
        latent_code,
        ref_audio_id,
        ref_audio_gender,
        ref_img_gender
    ):
        """
        Note:
        _val_epochで呼び出すことを前提としているため、
        model群は.eval()にされているという前提
        """
        with torch.no_grad():
            style_mapping = self.model.mapping_network(latent_code, ref_audio_id)
            style_audio = self.model.audio_style_encoder(ref_mel, ref_audio_gender)
            style_img = self.model.image_style_encoder(ref_img, ref_img_gender)

            _, GAN_F0, _ = self.model.f0_model(mel)

            converted_by_mapping = self.model.audio_generator(mel, style_mapping, masks=None, F0=GAN_F0)
            converted_by_audio = self.model.audio_generator(mel, style_audio, masks=None, F0=GAN_F0)
            converted_by_image = self.model.audio_generator(mel, style_img, masks=None, F0=GAN_F0)

            # 正規化の影響を戻す
            # params: mean = -4, std = 4
            mel = mel*4 - 4
            converted_by_mapping = converted_by_mapping*4 - 4
            converted_by_audio = converted_by_audio*4 - 4
            converted_by_image = converted_by_image*4 - 4
            
            out = self.vocoder(mel.squeeze(dim=1))
            out_by_mapping = self.vocoder(converted_by_mapping.squeeze(dim=1))
            out_by_audio = self.vocoder(converted_by_audio.squeeze(dim=1))
            out_by_image = self.vocoder(converted_by_image.squeeze(dim=1))

            audio = out[0].squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype("int16")

            audio_by_mapping = out_by_mapping[0].squeeze()
            audio_by_mapping = audio_by_mapping * MAX_WAV_VALUE
            audio_by_mapping = audio_by_mapping.cpu().numpy().astype("int16")

            audio_by_audio = out_by_audio[0].squeeze()
            audio_by_audio = audio_by_audio * MAX_WAV_VALUE
            audio_by_audio = audio_by_audio.cpu().numpy().astype("int16")

            audio_by_image = out_by_image[0].squeeze()
            audio_by_image = audio_by_image * MAX_WAV_VALUE
            audio_by_image = audio_by_image.cpu().numpy().astype("int16")

            os.makedirs(f"{self.args.result_dir}/{self.wandb_run_name}/{self.epochs}", exist_ok=True)
            write(f"{self.args.result_dir}/{self.wandb_run_name}/{self.epochs}/src.wav", self.args.sampling_rate, audio)
            write(f"{self.args.result_dir}/{self.wandb_run_name}/{self.epochs}/converted_by_mapping.wav", self.args.sampling_rate, audio_by_mapping)
            write(f"{self.args.result_dir}/{self.wandb_run_name}/{self.epochs}/converted_by_audio.wav", self.args.sampling_rate, audio_by_audio)
            write(f"{self.args.result_dir}/{self.wandb_run_name}/{self.epochs}/converted_by_image.wav", self.args.sampling_rate, audio_by_image)

            wandb.log({
                "source": wandb.Audio(
                    audio,
                    caption="source audio sample",
                    sample_rate=self.args.sampling_rate
                ),
                "converted_by_mapping": wandb.Audio(
                    audio_by_mapping,
                    caption="converted sample by Mapping Network",
                    sample_rate=self.args.sampling_rate
                ),
                "converted_by_audio": wandb.Audio(
                    audio_by_audio,
                    caption="converted sample by Audio Style Encoder",
                    sample_rate=self.args.sampling_rate
                ),
                "converted_by_image": wandb.Audio(
                    audio_by_image,
                    caption="converted sample by Image Sytle Encoder",
                    sample_rate=self.args.sampling_rate
                )
            })


    def log_artifact(self):
        wandb.log_artifact(self.artifact)

    def save_checkpoint(self, checkpoint_path, add_artifact=False):
        """Save checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be saved.
        """
        state_dict = {
            "optimizer": self.optimizer.state_dict(),
            "epochs": self.epochs,
            "model": {key: self.model[key].state_dict() for key in self.model}
        }
        # if self.model_ema is not None:
        #     state_dict['model_ema'] = {key: self.model_ema[key].state_dict() for key in self.model_ema}

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

        if(add_artifact):
            self.artifact.add_file(local_path=checkpoint_path, name=f"epoch{self.epochs}_weight")

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        for key in self.model:
            self._load(state_dict["model"][key], self.model[key])

        # if self.model_ema is not None:
        #     for key in self.model_ema:
        #         self._load(state_dict["model_ema"][key], self.model_ema[key])
        
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer.load_state_dict(state_dict["optimizer"])


    def _load(self, states, model, force_load=True):
        model_states = model.state_dict()
        for key, val in states.items():
            try:
                if key not in model_states:
                    continue
                if isinstance(val, nn.Parameter):
                    val = val.data

                if val.shape != model_states[key].shape:
                    self.logger.info("%s does not have same shape" % key)
                    print(val.shape, model_states[key].shape)
                    if not force_load:
                        continue

                    min_shape = np.minimum(np.array(val.shape), np.array(model_states[key].shape))
                    slices = [slice(0, min_index) for min_index in min_shape]
                    model_states[key][slices].copy_(val[slices])
                else:
                    model_states[key].copy_(val)
            except:
                self.logger.info("not exist :%s" % key)
                print("not exist ", key)

    def _train_epoch(self):
        self.epochs += 1

        _ = [ self.model[k].train() for k in self.model ]

        use_con_reg = (self.epochs >= self.args.con_reg_epoch)
        use_adv_cls = (self.epochs >= self.args.adv_cls_epoch)

        for train_step, batch in enumerate(tqdm(self.train_dataloader, desc="[Train]"), 1):
            
            # load data
            batch = [ b.to(self.device) for b in batch ]
            mel, ref_mel1, ref_mel2, audio_gender, audio_id, ref_audio_gender, ref_audio_id, \
                img, ref_img1, ref_img2, img_gender, ref_img_gender, latent_code, latent_code2 = batch
            
            # この順番なら先にある通常のタスクで調整した後にチャレンジングなタスクに挑戦

            # ===== image-guided image translation =====
            # discriminator ( mapping )
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.args.use_amp):
                d_loss, d_losses_mapping_iit = compute_image_d_loss(
                    self.model, self.args, img, img_gender, ref_audio_gender, z_trg=latent_code
                )
            self.optimizer.zero_grad()
            self.scaler.scale(d_loss).backward()
            self.optimizer.step("image_discriminator", self.scaler)

            # discriminator ( style )
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.args.use_amp):
                d_loss, d_losses_ref_iit = compute_image_d_loss(
                    self.model, self.args, img, img_gender, ref_img_gender, x_ref=ref_img1
                )
            self.optimizer.zero_grad()
            self.scaler.scale(d_loss).backward()
            self.optimizer.step("image_discriminator", self.scaler)

            # generator     ( mapping )
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.args.use_amp):
                g_loss, g_losses_mapping_iit = compute_image_g_loss(
                    self.model, self.args, img, img_gender, ref_audio_gender, z_trgs=[latent_code, latent_code2]
                )
            self.optimizer.zero_grad()
            self.scaler.scale(g_loss).backward()
            self.optimizer.step("image_generator", self.scaler)
            self.optimizer.step("mapping_network", self.scaler)
            self.optimizer.step("image_style_encoder", self.scaler)

            # generator     ( style )
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.args.use_amp):
                g_loss, g_losses_ref_iit = compute_image_g_loss(
                    self.model, self.args, img, img_gender, ref_img_gender, x_refs=[ref_img1, ref_img2]
                )
            self.optimizer.zero_grad()
            self.scaler.scale(g_loss).backward()
            self.optimizer.step("image_generator", self.scaler)

            #  ===== audio-guided voice conversion =====
            # discriminator ( mapping )
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.args.use_amp):
                d_loss, d_losses_mapping_avc = compute_audio_d_loss(
                    self.model, self.args.d_loss, mel, audio_gender, audio_id, ref_audio_gender, ref_audio_id, z_trg=latent_code, use_adv_cls=use_adv_cls, use_con_reg=use_con_reg
                )
            self.optimizer.zero_grad()
            self.scaler.scale(d_loss).backward()
            self.optimizer.step("audio_discriminator", self.scaler)

            # discriminator ( style )
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.args.use_amp):
                d_loss, d_losses_ref_avc = compute_audio_d_loss(
                    self.model, self.args.d_loss, mel, audio_gender, audio_id, ref_audio_gender, ref_audio_id, x_ref=ref_mel1, use_adv_cls=use_adv_cls, use_con_reg=use_con_reg
                )
            self.optimizer.zero_grad()
            self.scaler.scale(d_loss).backward()
            self.optimizer.step("audio_discriminator", self.scaler)
            
            # generator     ( mapping )
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.args.use_amp):
                g_loss, g_losses_mapping_avc = compute_audio_g_loss(
                    self.model, self.args.g_loss, mel, audio_gender, audio_id, ref_audio_gender, ref_audio_id, z_trgs=[latent_code, latent_code2], use_adv_cls=use_adv_cls
                )
            self.optimizer.zero_grad()
            self.scaler.scale(g_loss).backward()
            self.optimizer.step("audio_generator", self.scaler)
            self.optimizer.step("mapping_network", self.scaler)
            self.optimizer.step("audio_style_encoder", self.scaler)
            
            # generator     ( style )
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.args.use_amp):
                g_loss, g_losses_ref_avc = compute_audio_g_loss(
                    self.model, self.args.g_loss, mel, audio_gender, audio_id, ref_audio_gender, ref_audio_id, x_refs=[ref_mel1, ref_mel2], use_adv_cls=use_adv_cls
                )
            self.optimizer.zero_grad()
            self.scaler.scale(g_loss).backward()
            self.optimizer.step("audio_generator", self.scaler)

            # ===== audio-guided image translation =====
            # discriminator ( mapping )
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.args.use_amp):
                d_loss, d_losses_mapping_ait = compute_image_d_loss(
                    self.model, self.args, img, img_gender, ref_audio_gender, z_trg=latent_code
                )
            self.optimizer.zero_grad()
            self.scaler.scale(d_loss).backward()
            self.optimizer.step("image_discriminator", self.scaler)

            # discriminator ( style )
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.args.use_amp):
                d_loss, d_losses_ref_ait = compute_image_d_loss(
                    self.model, self.args, img, img_gender, ref_audio_gender, x_ref=ref_mel1, withAudio=True
                )
            self.optimizer.zero_grad()
            self.scaler.scale(d_loss).backward()
            self.optimizer.step("image_discriminator", self.scaler)

            # generator     ( mapping )
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.args.use_amp):
                g_loss, g_losses_mapping_ait = compute_image_g_loss(
                    self.model, self.args, img, img_gender, ref_audio_gender, z_trgs=[latent_code, latent_code2]
                )
            self.optimizer.zero_grad()
            self.scaler.scale(g_loss).backward()
            # style encoderはmapping networkにそって学習
            # mapping networkに合わせるような感じになる
            # ↑ 例えば style reconstruction とかね
            self.optimizer.step("image_generator", self.scaler)
            self.optimizer.step("mapping_network", self.scaler)
            self.optimizer.step("image_style_encoder", self.scaler)

            # generator     ( style )
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.args.use_amp):
                g_loss, g_losses_ref_ait = compute_image_g_loss(
                    self.model, self.args, img, img_gender, ref_audio_gender, x_refs=[ref_mel1, ref_mel2], withAudio=True
                )
            self.optimizer.zero_grad()
            self.scaler.scale(g_loss).backward()
            self.optimizer.step("image_generator", self.scaler)

            # ===== image-guided voice conversion =====
            # discriminator ( mapping )
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.args.use_amp):
                d_loss, d_losses_mapping_ivc = compute_audio_d_loss(
                    self.model, self.args.d_loss, mel, audio_gender, audio_id, ref_audio_gender, ref_audio_id, z_trg=latent_code, use_adv_cls=use_adv_cls, use_con_reg=use_con_reg
                )
            self.optimizer.zero_grad()
            self.scaler.scale(d_loss).backward()
            self.optimizer.step("audio_discriminator", self.scaler)

            # discriminator ( style )
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.args.use_amp):
                d_loss, d_losses_ref_ivc = compute_audio_d_loss(
                    self.model, self.args.d_loss, mel, audio_gender, audio_id, ref_img_gender, ref_audio_id, x_ref=ref_img1, use_adv_cls=use_adv_cls, use_con_reg=use_con_reg, withImage=True
                )
            self.optimizer.zero_grad()
            self.scaler.scale(d_loss).backward()
            self.optimizer.step("audio_discriminator", self.scaler)

            # generator     ( mapping )
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.args.use_amp):
                g_loss, g_losses_mapping_ivc = compute_audio_g_loss(
                    self.model, self.args.g_loss, mel, audio_gender, audio_id, ref_audio_gender, ref_audio_id, z_trgs=[latent_code, latent_code2], use_adv_cls=use_adv_cls
                )
            self.optimizer.zero_grad()
            self.scaler.scale(g_loss).backward()
            self.optimizer.step("audio_generator", self.scaler)
            self.optimizer.step("mapping_network", self.scaler)
            self.optimizer.step("audio_style_encoder", self.scaler)

            # generator     ( style )
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.args.use_amp):
                g_loss, g_losses_ref_ivc = compute_audio_g_loss(
                    self.model, self.args.g_loss, mel, audio_gender, audio_id, ref_img_gender, ref_audio_id, x_refs=[ref_img1, ref_img2], use_adv_cls=use_adv_cls, withImage=True
                )
            self.optimizer.zero_grad()
            self.scaler.scale(g_loss).backward()
            self.optimizer.step("audio_generator", self.scaler)

            self.optimizer.scheduler()

            # wandbのChartは/で管理しているっぽいので/で分割
            wandb.log({
                "train/by_task/iit/d_mapping": dict(d_losses_mapping_iit),
                "train/by_task/iit/d_style": dict(d_losses_ref_iit),
                "train/by_task/iit/g_mapping": dict(g_losses_mapping_iit),
                "train/by_task/iit/g_style": dict(g_losses_ref_iit),
                "train/by_task/ait/d_mapping": dict(d_losses_mapping_ait),
                "train/by_task/ait/d_style": dict(d_losses_ref_ait),
                "train/by_task/ait/g_mapping": dict(g_losses_mapping_ait),
                "train/by_task/ait/g_style": dict(g_losses_ref_ait),
                "train/by_task/avc/d_mapping": dict(d_losses_mapping_avc),
                "train/by_task/avc/d_style": dict(d_losses_ref_avc),
                "train/by_task/avc/g_mapping": dict(g_losses_mapping_avc),
                "train/by_task/avc/g_style": dict(g_losses_ref_avc),
                "train/by_task/ivc/d_mapping": dict(d_losses_mapping_ivc),
                "train/by_task/ivc/d_style": dict(d_losses_ref_ivc),
                "train/by_task/ivc/g_mapping": dict(g_losses_mapping_ivc),
                "train/by_task/ivc/g_style": dict(g_losses_ref_ivc),
                "train/by_loss/d_losses/mapping/iit": dict(d_losses_mapping_iit),
                "train/by_loss/d_losses/mapping/ait": dict(d_losses_mapping_ait),
                "train/by_loss/d_losses/mapping/avc": dict(d_losses_mapping_avc),
                "train/by_loss/d_losses/mapping/ivc": dict(d_losses_mapping_ivc),
                "train/by_loss/d_losses/style/iit": dict(d_losses_ref_iit),
                "train/by_loss/d_losses/style/ait": dict(d_losses_ref_ait),
                "train/by_loss/d_losses/style/avc": dict(d_losses_ref_avc),
                "train/by_loss/d_losses/style/ivc": dict(d_losses_ref_ivc),
                "train/by_loss/g_losses/mapping/iit": dict(g_losses_mapping_iit),
                "train/by_loss/g_losses/mapping/ait": dict(g_losses_mapping_ait),
                "train/by_loss/g_losses/mapping/avc": dict(g_losses_mapping_avc),
                "train/by_loss/g_losses/mapping/ivc": dict(g_losses_mapping_ivc),
                "train/by_loss/g_losses/style/iit": dict(g_losses_ref_iit),
                "train/by_loss/g_losses/style/ait": dict(g_losses_ref_ait),
                "train/by_loss/g_losses/style/avc": dict(g_losses_ref_avc),
                "train/by_loss/g_losses/style/ivc": dict(g_losses_ref_ivc)
            })

    @torch.no_grad()
    def _val_epoch(self, audio_log=False):
        use_adv_cls = (self.epochs >= self.args.adv_cls_epoch)

        _ = [self.model[k].eval() for k in self.model]

        for eval_step, batch in enumerate(tqdm(self.val_dataloader, desc="[Val]"), 1):

            # load data
            batch = [ b.to(self.device) for b in batch ]
            mel, ref_mel1, ref_mel2, audio_gender, audio_id, ref_audio_gender, ref_audio_id, \
                img, ref_img1, ref_img2, img_gender, ref_img_gender, latent_code, latent_code2 = batch
            
            # この順番なら先にある通常のタスクで調整した後にチャレンジングなタスクに挑戦

            # ===== image-guided image translation =====
            # discriminator ( mapping )
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.args.use_amp):
                _, d_losses_mapping_iit = compute_image_d_loss(
                    self.model, self.args, img, img_gender, ref_audio_gender, z_trg=latent_code, use_r1_reg=False
                )
            
            # discriminator ( style )
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.args.use_amp):
                _, d_losses_ref_iit = compute_image_d_loss(
                    self.model, self.args, img, img_gender, ref_img_gender, x_ref=ref_img1, use_r1_reg=False
                )
            
            # generator     ( mapping )
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.args.use_amp):
                _, g_losses_mapping_iit = compute_image_g_loss(
                    self.model, self.args, img, img_gender, ref_audio_gender, z_trgs=[latent_code, latent_code2]
                )
            
            # generator     ( style )
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.args.use_amp):
                _, g_losses_ref_iit = compute_image_g_loss(
                    self.model, self.args, img, img_gender, ref_img_gender, x_refs=[ref_img1, ref_img2]
                )
            
            #  ===== audio-guided voice conversion =====
            # discriminator ( mapping )
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.args.use_amp):
                _, d_losses_mapping_avc = compute_audio_d_loss(
                    self.model, self.args.d_loss, mel, audio_gender, audio_id, ref_audio_gender, ref_audio_id, z_trg=latent_code, use_adv_cls=use_adv_cls, use_r1_reg=False
                )
            
            # discriminator ( style )
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.args.use_amp):
                _, d_losses_ref_avc = compute_audio_d_loss(
                    self.model, self.args.d_loss, mel, audio_gender, audio_id, ref_audio_gender, ref_audio_id, x_ref=ref_mel1, use_adv_cls=use_adv_cls, use_r1_reg=False
                )
            
            # generator     ( mapping )
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.args.use_amp):
                _, g_losses_mapping_avc = compute_audio_g_loss(
                    self.model, self.args.g_loss, mel, audio_gender, audio_id, ref_audio_gender, ref_audio_id, z_trgs=[latent_code, latent_code2], use_adv_cls=use_adv_cls
                )
            
            # generator     ( style )
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.args.use_amp):
                _, g_losses_ref_avc = compute_audio_g_loss(
                    self.model, self.args.g_loss, mel, audio_gender, audio_id, ref_audio_gender, ref_audio_id, x_refs=[ref_mel1, ref_mel2], use_adv_cls=use_adv_cls
                )
            
            # ===== audio-guided image translation =====
            # discriminator ( mapping )
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.args.use_amp):
                _, d_losses_mapping_ait = compute_image_d_loss(
                    self.model, self.args, img, img_gender, ref_audio_gender, z_trg=latent_code, use_r1_reg=False
                )
            
            # discriminator ( style )
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.args.use_amp):
                _, d_losses_ref_ait = compute_image_d_loss(
                    self.model, self.args, img, img_gender, ref_audio_gender, x_ref=ref_mel1, withAudio=True, use_r1_reg=False
                )
            
            # generator     ( mapping )
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.args.use_amp):
                _, g_losses_mapping_ait = compute_image_g_loss(
                    self.model, self.args, img, img_gender, ref_audio_gender, z_trgs=[latent_code, latent_code2]
                )
            
            # generator     ( style )
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.args.use_amp):
                _, g_losses_ref_ait = compute_image_g_loss(
                    self.model, self.args, img, img_gender, ref_audio_gender, x_refs=[ref_mel1, ref_mel2], withAudio=True
                )
            
            # ===== image-guided voice conversion =====
            # discriminator ( mapping )
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.args.use_amp):
                _, d_losses_mapping_ivc = compute_audio_d_loss(
                    self.model, self.args.d_loss, mel, audio_gender, audio_id, ref_audio_gender, ref_audio_id, z_trg=latent_code, use_adv_cls=use_adv_cls, use_r1_reg=False
                )
            
            # discriminator ( style )
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.args.use_amp):
                _, d_losses_ref_ivc = compute_audio_d_loss(
                    self.model, self.args.d_loss, mel, audio_gender, audio_id, ref_img_gender, ref_audio_id, x_ref=ref_img1, use_adv_cls=use_adv_cls, use_r1_reg=False, withImage=True
                )
            
            # generator     ( mapping )
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.args.use_amp):
                _, g_losses_mapping_ivc = compute_audio_g_loss(
                    self.model, self.args.g_loss, mel, audio_gender, audio_id, ref_audio_gender, ref_audio_id, z_trgs=[latent_code, latent_code2], use_adv_cls=use_adv_cls
                )
            
            # generator     ( style )
            # ここのloss_f0のcompute_mean_f0がバッチサイズ1に対応していない
            # そのため、dataloaderでdrop_lastしていないと余った分が1個だった場合エラーが出る
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.args.use_amp):
                _, g_losses_ref_ivc = compute_audio_g_loss(
                    self.model, self.args.g_loss, mel, audio_gender, audio_id, ref_img_gender, ref_audio_id, x_refs=[ref_img1, ref_img2], use_adv_cls=use_adv_cls, withImage=True
                )
            
            wandb.log({
                "val/by_tasktask/iit/d_mapping": dict(d_losses_mapping_iit),
                "val/by_tasktask/iit/d_style": dict(d_losses_ref_iit),
                "val/by_tasktask/iit/g_mapping": dict(g_losses_mapping_iit),
                "val/by_tasktask/iit/g_style": dict(g_losses_ref_iit),
                "val/by_tasktask/ait/d_mapping": dict(d_losses_mapping_ait),
                "val/by_tasktask/ait/d_style": dict(d_losses_ref_ait),
                "val/by_tasktask/ait/g_mapping": dict(g_losses_mapping_ait),
                "val/by_tasktask/ait/g_style": dict(g_losses_ref_ait),
                "val/by_tasktask/avc/d_mapping": dict(d_losses_mapping_avc),
                "val/by_tasktask/avc/d_style": dict(d_losses_ref_avc),
                "val/by_tasktask/avc/g_mapping": dict(g_losses_mapping_avc),
                "val/by_tasktask/avc/g_style": dict(g_losses_ref_avc),
                "val/by_tasktask/ivc/d_mapping": dict(d_losses_mapping_ivc),
                "val/by_tasktask/ivc/d_style": dict(d_losses_ref_ivc),
                "val/by_tasktask/ivc/g_mapping": dict(g_losses_mapping_ivc),
                "val/by_tasktask/ivc/g_style": dict(g_losses_ref_ivc),
                "val/by_taskloss/d_losses/mapping/iit": dict(d_losses_mapping_iit),
                "val/by_taskloss/d_losses/mapping/ait": dict(d_losses_mapping_ait),
                "val/by_taskloss/d_losses/mapping/avc": dict(d_losses_mapping_avc),
                "val/by_taskloss/d_losses/mapping/ivc": dict(d_losses_mapping_ivc),
                "val/by_taskloss/d_losses/style/iit": dict(d_losses_ref_iit),
                "val/by_taskloss/d_losses/style/ait": dict(d_losses_ref_ait),
                "val/by_taskloss/d_losses/style/avc": dict(d_losses_ref_avc),
                "val/by_taskloss/d_losses/style/ivc": dict(d_losses_ref_ivc),
                "val/by_taskloss/g_losses/mapping/iit": dict(g_losses_mapping_iit),
                "val/by_taskloss/g_losses/mapping/ait": dict(g_losses_mapping_ait),
                "val/by_taskloss/g_losses/mapping/avc": dict(g_losses_mapping_avc),
                "val/by_taskloss/g_losses/mapping/ivc": dict(g_losses_mapping_ivc),
                "val/by_taskloss/g_losses/style/iit": dict(g_losses_ref_iit),
                "val/by_taskloss/g_losses/style/ait": dict(g_losses_ref_ait),
                "val/by_taskloss/g_losses/style/avc": dict(g_losses_ref_avc),
                "val/by_taskloss/g_losses/style/ivc": dict(g_losses_ref_ivc)
            })

        if audio_log:
            self.log_audio(mel, ref_mel1, ref_img1, latent_code, ref_audio_id,ref_audio_gender,ref_img_gender)
