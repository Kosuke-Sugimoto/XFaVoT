import torch
import random
import librosa
import numpy as np
from PIL import Image
from munch import Munch
from typing import Any, Union
from pathlib import Path
from omegaconf import OmegaConf
from librosa.util import normalize
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from src.utils.align_pathobj import align_pathobj
from src.datasets.Datalist import VCTKDatalist, CelebAHQDatalist
from src.datasets.ADatasetUtils import load_wav, mel_spectrogram, MAX_WAV_VALUE

class IADataset(Dataset):
    """
    要件：
        画像データセットと音声データセットとでドメインごとのデータ数を揃える
        ドメイン⇒男・女
        今回は圧倒的に少ないと思われる音声に合わせるが別verもすぐに導入できるようにしたい
    """
    
    def __init__(
        self,
        celeb_transformer,
        seed: int = 777,
        is_train: bool = True,
        test_size: float = 0.1,
        vctk_data_dir: Union[Path, str] = \
            "./Datasets/VCTK-Corpus",
        vctk_info_txt_filepath: Union[Path, str] = \
            "./Datasets/VCTK-Corpus/speaker-info.txt",
        vctk_ext: str = "wav",
        vctk_config_filepath: Union[Path, str] = \
            "./Configs/train_config.yml",
        celeb_data_dir: Union[Path, str] = \
            "./Datasets/CelebA-HQ",
        celeb_ext: str = "jpg",
        max_used_ids: int = 20
    ):
        self.celeb_transformer = celeb_transformer
        self.seed = seed
        self.is_train = is_train
        self.test_size = test_size
        self.vctk_data_dir = vctk_data_dir
        self.vctk_info_txt_filepath = vctk_info_txt_filepath
        self.vctk_config_filepath = align_pathobj(vctk_config_filepath)
        self.vctk_ext = vctk_ext
        self.celeb_data_dir = celeb_data_dir
        self.celeb_ext = celeb_ext
        self.max_used_ids = max_used_ids

        self.vctk_datalist_obj = VCTKDatalist(
            seed=self.seed,
            test_size=self.test_size,
            data_dir=self.vctk_data_dir,
            info_txt_filepath=self.vctk_info_txt_filepath,
            ext=self.vctk_ext,
            max_used_ids=max_used_ids
        )
        self.vctk_datalist_obj.setup()
        self.vctk_datalist, self.vctk_labels, self.vctk_usedid2idx, self.vctk_usedid2idx_reverse, self.vctk_id2trval = \
            self.vctk_datalist_obj.get()
        self.vctk_datalist = \
            self.vctk_datalist.train if is_train else self.vctk_datalist.val
        random.shuffle(self.vctk_datalist)

        self.vctk_config_obj = OmegaConf.load(self.vctk_config_filepath)

        self.celeb_datalist_obj = CelebAHQDatalist(
            seed=self.seed,
            data_dir=self.celeb_data_dir,
            ext=self.celeb_ext
        )
        self.celeb_datalist_obj.setup()
        self.celeb_datalist, self.celeb_labels, self.celeb_label2paths = self.celeb_datalist_obj.get()
        self.celeb_datalist = \
            self.celeb_datalist.train if is_train else self.celeb_datalist.val
        random.shuffle(self.celeb_datalist)

        # 音声側に合わせる
        self.celeb_datalist = self.celeb_datalist[:len(self.vctk_datalist)]

        self.gender_str2int = {"M": 0, "F": 1, "male": 0, "female": 1}

        # TODO
        # 同じラベルのデータを抽出するitem
        # ドメインの数調整(M:0, F:1, 話者:2〜)

    def __len__(self):
        """
        Note:
        ひとまず仮置き
        どうにかしてvctk, celebahqのどちらともの長さを取得できるようにしたい
        ⇒もしかして整数値でなくても大丈夫？(例えば文字列とか)
        """
        return len(self.vctk_datalist)

    def __load_mel_tensor(self, audio_data_filepath: Path):
        """
        Note:
        これはBIGVGANのリポジトリを参考に作ったもの
        StarGANv2-VCの方にあったロバスト性のためのスケーリングは入れるか迷う

        if not self.validation: # random scale for robustness
            random_scale = 0.5 + 0.5 * np.random.random()
            wave_tensor = random_scale * wave_tensor

        多分だがBIGVGANリポジトリにあるようにaudioの段階でsplitなりしてから
        spectrogramに変換したほうが計算効率が良いと思われる
        ⇒要検討
        """

        wave_data, sr = load_wav(audio_data_filepath.as_posix(), self.vctk_config_obj.sampling_rate)
        wave_data = wave_data / MAX_WAV_VALUE

        wave_data = normalize(wave_data) * 0.95

        wave_data = torch.FloatTensor(wave_data)
        wave_data = wave_data.unsqueeze(0)

        mel_data = mel_spectrogram(wave_data, self.vctk_config_obj.n_fft, self.vctk_config_obj.num_mels,
                                    self.vctk_config_obj.sampling_rate, self.vctk_config_obj.hop_size, self.vctk_config_obj.win_size,
                                    self.vctk_config_obj.fmin, self.vctk_config_obj.fmax, center=False)
        mel_data = mel_data.squeeze() # (1, num_mels, times) -> (num_mels, times) ?

        mel_length = mel_data.size(1)
        if mel_length > self.vctk_config_obj.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.vctk_config_obj.max_mel_length)
            mel_data = mel_data[:, random_start:random_start + self.vctk_config_obj.max_mel_length]

        labels = self.vctk_labels[audio_data_filepath]
        gender = self.gender_str2int[labels.gender]
        id_idx = self.vctk_usedid2idx[labels.id]

        return mel_data, Munch(gender=gender, id_idx=id_idx)
    
    def __load_img_tensor(self, img_datapath: Path):
        img = Image.open(img_datapath).convert("RGB")
        img_tensor = self.celeb_transformer(img)

        label = self.celeb_labels[img_datapath]
        gender = self.gender_str2int[label.gender]

        return img_tensor, Munch(gender=gender)

    def __getitem__(self, idx):
        
        # Audio
        vctk_datapath = self.vctk_datalist[idx]
        ref_vctk_datapath1 = random.choice(self.vctk_datalist)

        mel_tensor, label_obj_audio = self.__load_mel_tensor(vctk_datapath)
        ref_mel_tensor1, ref_label_obj_audio = self.__load_mel_tensor(ref_vctk_datapath1)

        ref_vctk_datapath2 = random.choice(
            self.vctk_id2trval[f"p{self.vctk_usedid2idx_reverse[ref_label_obj_audio.id_idx]}"].train \
                if(self.is_train) else self.vctk_id2trval[f"p{self.vctk_usedid2idx_reverse[ref_label_obj_audio.id_idx]}"].val
        )

        ref_mel_tensor2, _ = self.__load_mel_tensor(ref_vctk_datapath2)

        # Image
        celeb_datapath = self.celeb_datalist[idx]
        ref_celeb_datapath1 = random.choice(self.celeb_datalist)
        
        img_tensor, label_obj_img = self.__load_img_tensor(celeb_datapath)
        ref_img_tensor1, ref_label_obj_img = self.__load_img_tensor(ref_celeb_datapath1)

        ref_celeb_datapath2 = random.choice(self.celeb_label2paths[ref_label_obj_audio.gender])

        ref_img_tensor2, _ = self.__load_img_tensor(ref_celeb_datapath2)

        return mel_tensor, ref_mel_tensor1, ref_mel_tensor2, label_obj_audio, ref_label_obj_audio, \
            img_tensor, ref_img_tensor1, ref_img_tensor2, label_obj_img, ref_label_obj_img
        # return mel_tensor, ref_mel_tensor1, ref_mel_tensor2, \
        #     img_tensor, ref_img_tensor1, ref_img_tensor2


class Collater(object):
    def __init__(
        self,
        max_mel_length,
        latent_dim
    ):
        self.max_mel_length = max_mel_length
        self.latent_dim = latent_dim

    def __call__(self, batch):
        batch_size = len(batch)
        nmels = batch[0][0].size(0)
        nchannels = batch[0][5].size(0)
        img_size1 = batch[0][5].size(1)
        img_size2 = batch[0][5].size(2)
        mel_tensor = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref_mel_tensor1 = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref_mel_tensor2 = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        audio_label_id = torch.zeros((batch_size)).long()
        audio_label_gender = torch.zeros((batch_size)).long()
        ref_audio_label_id = torch.zeros((batch_size)).long()
        ref_audio_label_gender = torch.zeros((batch_size)).long()
        img_tensor = torch.zeros((batch_size, nchannels, img_size1, img_size2)).float()
        ref_img_tensor1 = torch.zeros((batch_size, nchannels, img_size1, img_size2)).float()
        ref_img_tensor2 = torch.zeros((batch_size, nchannels, img_size1, img_size2)).float()
        img_label_gender = torch.zeros((batch_size)).long()
        ref_img_label_gender = torch.zeros((batch_size)).long()

        for bid, (mel, rmel1, rmel2, alabel, ralabel, img, rimg1, rimg2, ilabel, rilabel) in enumerate(batch):
            mel_size = mel.size(1)
            mel_tensor[bid, :, :mel_size] = mel

            ref_mel_size1 = rmel1.size(1)
            ref_mel_tensor1[bid, :, :ref_mel_size1] = rmel1

            ref_mel_size2 = rmel2.size(1)
            ref_mel_tensor2[bid, :, :ref_mel_size2] = rmel2

            audio_label_id[bid] = torch.tensor(alabel.id_idx).long()
            audio_label_gender[bid] = torch.tensor(alabel.gender).long()

            ref_audio_label_id[bid] = torch.tensor(ralabel.id_idx).long()
            ref_audio_label_gender[bid] = torch.tensor(ralabel.gender).long()

            img_tensor[bid, :, :] = img
            ref_img_tensor1[bid, :, :, :] = rimg1
            ref_img_tensor2[bid, :, :, :] = rimg2

            img_label_gender[bid] = torch.tensor(ilabel.gender).long()
            ref_img_label_gender[bid] = torch.tensor(rilabel.gender).long()

        # StarGANv2-VCのコードではz_trgに相当
        # y_trgはref_audio_labelに依存させる
        # 多分、ここは画像と音声で共通でなくても良いはず…
        latent_code = torch.randn(batch_size, self.latent_dim)
        latent_code2 = torch.randn(batch_size, self.latent_dim)

        mel_tensor, ref_mel_tensor1, ref_mel_tensor2 = \
            mel_tensor.unsqueeze(1), ref_mel_tensor1.unsqueeze(1), ref_mel_tensor2.unsqueeze(1)

        return  mel_tensor, ref_mel_tensor1, ref_mel_tensor2, audio_label_gender, audio_label_id, ref_audio_label_gender, ref_audio_label_id, \
            img_tensor, ref_img_tensor1, ref_img_tensor2, img_label_gender, ref_img_label_gender, latent_code, latent_code2

def build_train_dataloader(
    img_size=256,
    batch_size=8,
    prob=0.5,
    num_workers=2,
    max_mel_length=192
):
    crop = transforms.RandomResizedCrop(
        img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
    rand_crop = transforms.Lambda(
        lambda x: crop(x) if random.random() < prob else x)

    transform = transforms.Compose([
        rand_crop,
        transforms.Resize([img_size, img_size]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    dataset = IADataset(celeb_transformer=transform)
    collate_fn = Collater(max_mel_length)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        # pin_memory=True,
        drop_last=True
    )


def build_val_dataloader(
    img_size=256,
    shuffle=True,
    batch_size=8,
    num_workers=2,
    max_mel_length=192
):
    height, width = img_size, img_size
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.Resize([height, width]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = IADataset(
        celeb_transformer=transform,
        is_train=False
    )
    collate_fn = Collater(max_mel_length)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        # pin_memory=True,
        drop_last=False
    )

if __name__=="__main__":
    import json
    from scipy.io.wavfile import write
    from BigVGAN.env import AttrDict
    from BigVGAN.models import BigVGAN as Generator
    from BigVGAN.inference_e2e import scan_checkpoint, load_checkpoint
    
    dataloader = build_val_dataloader()

    h = None
    device = None
    torch.backends.cudnn.benchmark = False

    with Path("BigVGAN/exp/config.json").open() as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint("BigVGAN/exp/g_05000000.zip", device)
    generator.load_state_dict(state_dict_g["generator"])
    generator.eval()
    generator.remove_weight_norm()

    for i, data in enumerate(dataloader):
        mel, ref_mel1, ref_mel2, audio_gender, audio_id, ref_audio_gender, ref_audio_id, \
        img, ref_img1, ref_img2, img_gender, ref_img_gender, latent_code, latent_code2 = data

        mel = mel.to(device)
        ref_mel1 = ref_mel1.to(device)
        ref_mel2 = ref_mel2.to(device)
        audio_gender = audio_gender.to(device)
        audio_id = audio_id.to(device)
        ref_audio_gender = ref_audio_gender.to(device)
        ref_audio_id = ref_audio_id.to(device)
        img = img.to(device)
        ref_img1 = ref_img1.to(device)
        ref_img2 = ref_img2.to(device)
        img_gender = img_gender.to(device)
        ref_img_gender = ref_img_gender.to(device)
        latent_code = latent_code.to(device)
        latent_code2 = latent_code2.to(device)

        with torch.no_grad():
            # BigVGAN required size: (ch, mel, time)
            
            out = generator(mel.squeeze())

            audio = out[0].squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype("int16")

            print(out.size(), audio.shape)

            if i == 0:
                write("Samples/IADataset_test_sample2.wav", h.sampling_rate, audio)
