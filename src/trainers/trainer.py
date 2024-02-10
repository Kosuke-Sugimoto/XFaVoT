import torch
from tqdm import tqdm
from collections import defaultdict

class Trainer(object):
    
    def __init__(
        self,
        args,
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        scheduler = None,
        initial_steps=0,
        initial_epochs=0,
        device=torch.device("cuda"),
    ):
        self.model = model
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler
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
                img, ref_img1, ref_img2, img_gender, ref_img_gender = batch
            
            # この順番なら先にある通常のタスクで調整した後にチャレンジングなタスクに挑戦

            # image-guided image translation
            # discriminator ( mapping )
            # discriminator ( style )
            # generator     ( mapping )
            # generator     ( style )

            # audio-guided voice conversion
            # discriminator ( mapping )
            # discriminator ( style )
            # generator     ( mapping )
            # generator     ( style )

            # audio-guided image translation
            # discriminator ( mapping )
            # discriminator ( style )
            # generator     ( mapping )
            # generator     ( style )

            # image-guided voice conversion
            # discriminator ( mapping )
            # discriminator ( style )
            # generator     ( mapping )
            # generator     ( style )
