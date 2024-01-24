from munch import Munch
from typing import Union
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from src.datasets.Datalist import VCTKDatalist, CelebAHQDatalist

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
        celeb_data_dir: Union[Path, str] = \
            "./Datasets/CelebA-HQ",
        celeb_ext: str = "jpg"
    ):
        self.seed = seed
        self.test_size = test_size
        self.vctk_data_dir = vctk_data_dir
        self.vctk_info_txt_filepath = vctk_info_txt_filepath
        self.vctk_ext = vctk_ext
        self.celeb_data_dir = celeb_data_dir
        self.celeb_ext = celeb_ext

        self.vctk_datalist = VCTKDatalist(
            seed=self.seed,
            test_size=self.test_size,
            data_dir=self.vctk_data_dir,
            info_txt_filepath=self.info_txt_filepath,
            ext=self.vctk_ext
        )
        self.vctk_datalist.setup()
        
        self.celeb_datalist = CelebAHQDatalist(
            seed=self.seed,
            data_dir=self.celeb_data_dir,
            ext=self.celeb_ext
        )
        self.celeb_datalist.setup()
