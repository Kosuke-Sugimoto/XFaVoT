from munch import Munch
from typing import Union
from pathlib import Path

from src.datasets.get_datalist import get_paths_mappings

def get_row_data(
    # if later than python3.10, below Type Check is
    # Path | str | None
    image_data_dir: Union(Path, str, None),
    audio_data_dir: Union(Path, str, None),
    video_data_dir: Union(Path, str, None),
    image_datalist_path: Union(Path, str) = Path("./Datalists/image_train_datalist.txt"),
    audio_datalist_path: Union(Path, str) = Path("./Datalists/audio_train_datalist.txt")
) -> Munch:
    
    if video_data_dir is None:
        if image_data_dir is None or audio_data_dir is None:
            raise ValueError("Required video dir path or image and audio dir path")
        else:
            image_data_dir = convert_str2pathobj(image_data_dir)
            audio_data_dir = convert_str2pathobj(audio_data_dir)
            image_datalist_path = convert_str2pathobj(image_datalist_path)
            audio_datalist_path = convert_str2pathobj(audio_datalist_path)
            row_data = get_data_from_ia(image_data_dir, audio_data_dir, image_datalist_path, audio_datalist_path)
    else:
        row_data = get_data_from_video(video_data_dir, image_datalist_path, audio_datalist_path)

    return row_data

def get_data_from_ia(
    image_data_dir: Path,
    audio_data_dir: Path,
    image_datalist_path: Path,
    audio_datalist_path: Path
) -> Munch:
    """
    Returns:
        row_data (Munch): 学習データの元がdict形式になったもの

    Note:
        row_dataのプロパティは以下\n
        ・image_datalist⇒学習画像データへのパスが格納されたリスト\n
        ・audio_datalist⇒学習音声データへのパスが格納されたリスト\n
        ・image_mappings⇒学習画像データのドメインとパスのマッピング\n
        ・audio_mappings⇒学習音声データのドメインとパスのマッピング\n
        audio_mappigsに関してはマッピングが複数になるのでdict形式になる可能性あり 
    """
    
    image_datalist, image_mappings = get_paths_mappings(
        image_data_dir,
        image_datalist_path,
        "CelebA-HQ"
    )
    audio_datalist, audio_mappings = get_paths_mappings(
        audio_data_dir,
        audio_datalist_path,
        "VCTK"
    )

    return Munch(
        image_datalist = image_datalist,
        audio_datalist = audio_datalist,
        image_mappings = image_mappings,
        audio_mappings = audio_mappings
    )

def get_data_from_video(
    video_data_dir: Path,
    image_datalist_path: Path,
    audio_datalist_path: Path
) -> Munch:
    raise NotImplementedError

def convert_str2pathobj(target: Union(Path, str)) -> Path:

    if isinstance(target, Path):
        target = target.resolve()
    else:
        target = Path(target).resolve()

    return target
