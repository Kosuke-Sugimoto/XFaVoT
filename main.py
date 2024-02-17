import click
import torch
import yaml
import os.path as osp
from munch import Munch
from omegaconf import OmegaConf

from StarGANv2VC.Utils.ASR.models import ASRCNN
from StarGANv2VC.Utils.JDC.model import JDCNet
from src.trainers.trainer import Trainer
from src.utils.seed_everything import seed_everything
from src.models.build_models import build_model
from src.optimizers.build_optimizers import build_optimizer
from src.datasets.IADataset import build_train_dataloader, build_val_dataloader

@click.command()
@click.option("--config_path", type=str, default="Configs/train_config.yml")
def main(config_path):
    config = OmegaConf.load(config_path)

    seed_everything(config.seed)

    batch_size = config.batch_size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = config.epochs
    vctk_data_dir = config.vctk_data_dir
    celeb_data_dir = config.celeb_data_dir
    num_workers = config.num_workers
    img_size = config.img_size
    prob = config.prob
    max_mel_length = config.max_mel_length
    latent_dim = config.latent_dim
    sample_freq = config.sample_freq
    save_freq = config.save_freq
    wandb_save_freq = config.wandb_save_freq
    log_dir = config.log_dir

    train_dataloader = build_train_dataloader(
        vctk_data_dir=vctk_data_dir,
        celeb_data_dir=celeb_data_dir,
        img_size=img_size,
        batch_size=batch_size,
        prob=prob,
        num_workers=num_workers,
        max_mel_length=max_mel_length,
        latent_dim=latent_dim
    )
    val_dataloader = build_val_dataloader(
        vctk_data_dir=vctk_data_dir,
        celeb_data_dir=celeb_data_dir,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        max_mel_length=max_mel_length,
        latent_dim=latent_dim
    )

    # Load ASR Model
    ASR_config = config.ASR_config
    ASR_path = config.ASR_path
    with open(ASR_config) as file:
        ASR_config = yaml.safe_load(file)
    ASR_model_config = ASR_config["model_params"]
    ASR_model = ASRCNN(**ASR_model_config)
    params = torch.load(ASR_path, map_location='cpu')['model']
    ASR_model.load_state_dict(params)
    _ = ASR_model.eval()

    # Load F0 Model
    F0_path = config.F0_path
    F0_model = JDCNet(num_class=1, seq_len=192)
    params = torch.load(F0_path, map_location='cpu')['net']
    F0_model.load_state_dict(params)

    # Build Model & Optimizer
    model = build_model(config, F0_model, ASR_model)

    scheduler_params = {
        "max_lr": float(config.lr),
        "pct_start": float(config.pct_start),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }

    _ = [model[key].to(device) for key in model]
    scheduler_params_dict = {key: scheduler_params.copy() for key in model}
    scheduler_params_dict['mapping_network']['max_lr'] = 2e-6
    optimizer = build_optimizer({key: model[key].parameters() for key in model},
                                      scheduler_params_dict=scheduler_params_dict)
    
    trainer = Trainer(
        args = config,
        model = model,
        optimizer = optimizer,
        device = device,
        train_dataloader = train_dataloader,
        val_dataloader = val_dataloader
    )

    # log_dir update
    log_dir += f"/{trainer.wandb_run_name}"

    for _ in range(1, epochs+1):
        now_epoch = trainer.epochs

        trainer._train_epoch()
        trainer._val_epoch(audio_log=((now_epoch % sample_freq) == 0))

        if (now_epoch % save_freq) == 0:
            trainer.save_checkpoint(
                osp.join(log_dir, 'epoch_%05d.pth' % now_epoch),
                add_artifact=((now_epoch % wandb_save_freq) == 0)
            )

    trainer.log_artifact()

    
if __name__=="__main__":
    main()
