import torch
import random
import librosa
import numpy as np
from munch import Munch
from typing import Union
from pathlib import Path
from omegaconf import OmegaConf
from librosa.util import normalize
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
        seed: int = 777,
        test_size: float = 0.1,
        vctk_data_dir: Union[Path, str] = \
            "./Datasets/VCTK-Corpus",
        vctk_info_txt_filepath: Union[Path, str] = \
            "./Datasets/VCTK-Corpus/speaker-info.txt",
        vctk_ext: str = "wav",
        vctk_config_filepath: Union[Path, str] = \
            "./Configs/audio_config.yml",
        celeb_data_dir: Union[Path, str] = \
            "./Datasets/CelebA-HQ",
        celeb_ext: str = "jpg"
    ):
        self.seed = seed
        self.test_size = test_size
        self.vctk_data_dir = vctk_data_dir
        self.vctk_info_txt_filepath = vctk_info_txt_filepath
        self.vctk_config_filepath = align_pathobj(vctk_config_filepath)
        self.vctk_ext = vctk_ext
        self.celeb_data_dir = celeb_data_dir
        self.celeb_ext = celeb_ext

        self.vctk_datalist_obj = VCTKDatalist(
            seed=self.seed,
            test_size=self.test_size,
            data_dir=self.vctk_data_dir,
            info_txt_filepath=self.vctk_info_txt_filepath,
            ext=self.vctk_ext
        )
        self.vctk_datalist_obj.setup()
        self.vctk_datalist, self.vctk_labels = self.vctk_datalist_obj.get()

        self.vctk_config_obj = OmegaConf.load(self.vctk_config_filepath)

        self.celeb_datalist_obj = CelebAHQDatalist(
            seed=self.seed,
            data_dir=self.celeb_data_dir,
            ext=self.celeb_ext
        )
        self.celeb_datalist_obj.setup()
        self.celeb_datalist, self.celeb_labels = self.celeb_datalist_obj.get()

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

        label = self.vctk_labels[audio_data_filepath].gender
        label = self.gender_str2int[label]

        return mel_data, label

    def __getitem__(self, idx):
        
        vctk_datapath = self.vctk_datalist.train[idx]

        mel_tensor, label = self.__load_mel_tensor(vctk_datapath)

        return mel_tensor

if __name__=="__main__":
    import json
    from scipy.io.wavfile import write
    from BigVGAN.env import AttrDict
    from BigVGAN.models import BigVGAN as Generator
    from BigVGAN.inference_e2e import scan_checkpoint, load_checkpoint
    
    iadataset = IADataset()

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

    with torch.no_grad():
        # required size: (ch, mel, time)
        tensor = iadataset[0].unsqueeze(dim=0).to(device)
        
        out = generator(tensor)

        audio = out.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype("int16")

        write("Samples/IADataset_test_sample.wav", h.sampling_rate, audio)
