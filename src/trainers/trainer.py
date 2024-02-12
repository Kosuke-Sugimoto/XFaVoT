import torch
from tqdm import tqdm
from collections import defaultdict
from starganv2.core.solver import compute_d_loss as compute_image_d_loss, compute_g_loss as compute_image_g_loss
from StarGANv2VC.losses import compute_d_loss as compute_audio_d_loss, compute_g_loss as compute_audio_g_loss

class Trainer(object):
    
    def __init__(
        self,
        args,
        model,
        epochs,
        optimizer,
        train_dataloader,
        val_dataloader,
        initial_steps=0,
        initial_epochs=0,
        device=torch.device("cuda"),
    ):
        self.model = model
        self.args = args
        self.epochs = epochs
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.initial_steps = initial_steps
        self.initial_epochs = initial_epochs
        self.device = device

    def _train_epoch(self):
        self.epochs += 1

        train_losses = defaultdict(list)
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
            d_loss, d_losses_mapping_iit = compute_image_d_loss(
                self.model, self.args, img, img_gender, ref_audio_gender, z_trg=latent_code
            )
            self.optimizer.zero_grad()
            d_loss.backward()
            self.optimizer.step()

            # discriminator ( style )
            d_loss, d_losses_ref_iit = compute_image_d_loss(
                self.model, self.args, img, img_gender, ref_img_gender, x_ref=ref_img1
            )
            self.optimizer.zero_grad()
            d_loss.backward()
            self.optimizer.step()

            # generator     ( mapping )
            g_loss, g_losses_mapping_iit = compute_image_g_loss(
                self.model, self.args, img, img_gender, ref_audio_gender, z_trgs=[latent_code, latent_code2]
            )
            self.optimizer.zero_grad()
            g_loss.backward()
            self.optimizer.step()

            # generator     ( style )
            g_loss, g_losses_ref_iit = compute_image_g_loss(
                self.model, self.args, img, img_gender, ref_img_gender, x_refs=[ref_img1, ref_img2]
            )
            self.optimizer.zero_grad()
            g_loss.backward()
            self.optimizer.step()

            #  ===== audio-guided voice conversion =====
            # discriminator ( mapping )
            d_loss, d_losses_mapping_avc = compute_audio_d_loss(
                self.model, self.args.d_loss, mel, audio_gender, ref_audio_gender, ref_audio_id, z_trg=latent_code, use_adv_cls=use_adv_cls, use_con_reg=use_con_reg
            )
            self.optimizer.zero_grad()
            d_loss.backward()
            self.optimizer.step()

            # discriminator ( style )
            d_loss, d_losses_ref_avc = compute_audio_d_loss(
                self.model, self.args.d_loss, mel, audio_gender, ref_audio_gender, ref_audio_id, x_ref=ref_mel1, use_adv_cls=use_adv_cls, use_con_reg=use_con_reg
            )
            self.optimizer.zero_grad()
            d_loss.backward()
            self.optimizer.step()
            
            # generator     ( mapping )
            g_loss, g_losses_mapping_avc = compute_audio_g_loss(
                self.model, self.args.g_loss, mel, audio_gender, ref_audio_gender, ref_audio_id, z_trgs=[latent_code, latent_code2], use_adv_cls=use_adv_cls
            )
            self.optimizer.zero_grad()
            g_loss.backward()
            self.optimizer.step()
            
            # generator     ( style )
            g_loss, g_losses_ref_avc = compute_audio_g_loss(
                self.model, self.args.g_loss, mel, audio_gender, ref_audio_gender, ref_audio_id, x_refs=[ref_mel1, ref_mel2], use_adv_cls=use_adv_cls
            )
            self.optimizer.zero_grad()
            g_loss.backward()
            self.optimizer.step()

            # ===== audio-guided image translation =====
            # discriminator ( mapping )
            d_loss, d_losses_mapping_ait = compute_image_d_loss(
                self.model, self.args, img, img_gender, ref_audio_gender, z_trg=latent_code
            )
            self.optimizer.zero_grad()
            d_loss.backward()
            self.optimizer.step()

            # discriminator ( style )
            d_loss, d_losses_ref_ait = compute_image_d_loss(
                self.model, self.args, img, img_gender, ref_audio_gender, x_ref=ref_mel1, withAudio=True
            )
            self.optimizer.zero_grad()
            d_loss.backward()
            self.optimizer.step()

            # generator     ( mapping )
            g_loss, g_losses_mapping_ait = compute_image_g_loss(
                self.model, self.args, img, img_gender, ref_audio_gender, z_trgs=[latent_code, latent_code2]
            )
            self.optimizer.zero_grad()
            g_loss.backward()
            self.optimizer.step()

            # generator     ( style )
            g_loss, g_losses_ref_ait = compute_image_g_loss(
                self.model, self.args, img, img_gender, ref_audio_gender, x_refs=[ref_mel1, ref_mel2], withAudio=True
            )
            self.optimizer.zero_grad()
            g_loss.backward()
            self.optimizer.step()

            # ===== image-guided voice conversion =====
            # discriminator ( mapping )
            d_loss, d_losses_mapping_ivc = compute_audio_d_loss(
                self.model, self.args.d_loss, mel, audio_gender, ref_audio_gender, ref_audio_id, z_trg=latent_code, use_adv_cls=use_adv_cls, use_con_reg=use_con_reg
            )
            self.optimizer.zero_grad()
            d_loss.backward()
            self.optimizer.step()

            # discriminator ( style )
            d_loss, d_losses_ref_ivc = compute_audio_d_loss(
                self.model, self.args.d_loss, mel, audio_gender, ref_img_gender, ref_audio_id, x_ref=ref_img1, use_adv_cls=use_adv_cls, use_con_reg=use_con_reg, withImage=True
            )
            self.optimizer.zero_grad()
            d_loss.backward()
            self.optimizer.step()

            # generator     ( mapping )
            g_loss, g_losses_mapping_ivc = compute_audio_g_loss(
                self.model, self.args.g_loss, mel, audio_gender, ref_audio_gender, ref_audio_id, z_trgs=[latent_code, latent_code2], use_adv_cls=use_adv_cls
            )
            self.optimizer.zero_grad()
            g_loss.backward()
            self.optimizer.step()

            # generator     ( style )
            g_loss, g_losses_ref_ivc = compute_audio_g_loss(
                self.model, self.args.g_loss, mel, audio_gender, ref_img_gender, ref_audio_id, x_refs=[ref_img1, ref_img2], use_adv_cls=use_adv_cls, withImage=True
            )
            self.optimizer.zero_grad()
            g_loss.backward()
            self.optimizer.step()

            print(d_losses_mapping_iit, d_losses_mapping_ait)
            print(d_losses_mapping_avc, d_losses_mapping_ivc)
            print(d_losses_ref_iit, d_losses_mapping_ait)
            print(d_losses_ref_avc, d_losses_mapping_ivc)
            print(g_losses_mapping_iit, g_losses_mapping_ait)
            print(g_losses_mapping_avc, g_losses_mapping_ivc)
            print(g_losses_ref_iit, g_losses_ref_ait)
            print(g_losses_ref_avc, g_losses_ref_ivc)
