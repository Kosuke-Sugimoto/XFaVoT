import json
import random
from typing import Union
from pathlib import Path
from munch import Munch
from sklearn.model_selection import train_test_split
from src.utils.print_loading import print_loading_factory

def get_paths_mappings(
    data_dir: Path,
    data_list_path: Path,
    dataset_name: str,
    seed: int = 777
) -> tuple[list, dict]:
    
    random.seed(seed)

    datalist, mappings = getattr(globals(), f"get_from_{dataset_name}")(
        data_dir,
        data_list_path,
        seed
    )

    return datalist, mappings

def get_from_VCTK(
    data_dir: Path,
    data_list_path: Path,
    seed: int
):
    
    if data_list_path.exists():
        datalist, mappings = read_from_VCTK(data_list_path)
        # datalist = read_from_VCTK(data_list_path)
    else:
        print(f"{data_list_path.name} is not exist")
        make_from_VCTK(data_dir=data_dir, 
                       data_list_path=data_list_path, 
                       seed=seed)
        datalist, mappings = read_from_VCTK(data_list_path)
        # datalist = read_from_VCTK(data_list_path)
    print(datalist)
    print(mappings)

    return datalist, mappings

@print_loading_factory("Make dataset now", 10)
def make_from_VCTK(
    data_dir: Path,
    data_list_path: Path,
    seed: int,
    spk_num: int = 20,
    spk_info_filename: str = "speaker-info.txt",
    mappings_filename: str = "mappings.json"
):
    """
    Note:
        speaker-info.txtを参照する部分でVCTK-Corpusをハードコーディングしてるので注意
        XFaVoTでは音声に使うドメインが性別＋話者ID
        そのため、0:M, 1:Fをgender、それ以降を話者IDとする
    """
    
    subdirs_iter = data_dir.glob("VCTK*/wav48/*")
    spks = sorted(list(set(d.name for d in subdirs_iter)))
    trg_spks = random.sample(spks, spk_num)
    spk2idx = dict(zip(trg_spks, range(len(trg_spks))))

    with data_dir.joinpath("VCTK-Corpus", spk_info_filename).open() as file:
        spk_info_list = file.readlines()
    spk_info_list.pop(0) # delete column
    spk2gender = {f"p{info.split()[0]}": 0 if info.split()[2] == 'M' else 1 for info in spk_info_list}

    # iteratorをリスト化する場合は注意
    # 一度リスト化してしまうと、iteratorの中身がなくなる？
    paths_iter = data_dir.glob("VCTK*/wav48/**/*.wav")
    paths = list(filter(lambda x: extract_upper_dirname(x) in trg_spks, paths_iter))

    # 本来なら話者のwavごとに↓を適用した方が公平なデータの分布になる
    # 今回はパス
    train_paths, val_paths = train_test_split(paths, shuffle=True, random_state=seed)

    # ここまで作ってからdata_list_pathがtrain, evalで複数個あることに思い至る
    # 変えるのはここまで作って変更加えるのは少し手間なので、
    # data_list_pathを「必ず」trainの方の名前で指定することで対処
    data_list_path.parent.mkdir(parents=True, exist_ok=True)
    with data_list_path.open("w") as file:
        cont_all = []
        for train_path in train_paths:
            train_path = train_path.as_posix()
            spk = train_path.split("/")[-2]
            cont = f"{train_path}|{spk2gender[spk]}|{spk2idx[spk]}\n"
            cont_all.append(cont)
        file.writelines(cont_all)

    # data_list_pathは{}_train_datalist.txtのPathオブジェクトのハズなので
    # まずはvalの方を指すファイル名へ変換
    data_list_path = data_list_path.with_name(
        f"{data_list_path.name.replace('train', 'val')}"
    )
    with data_list_path.open("w") as file:
        cont_all = []
        for val_path in val_paths:
            val_path = val_path.as_posix()
            spk = val_path.split("/")[-2]
            cont = f"{val_path}|{spk2gender[spk]}|{spk2idx[spk]}\n"
            cont_all.append(cont)
        file.writelines(cont_all)

    # ひっじょーに実装汚くなっちゃうがここでmappingsを保存
    mappings = Munch(spk2idx=spk2idx, spk2gender=spk2gender)
    mappings_json = json.dumps(mappings)
    with Path("Datalists").joinpath(mappings_filename).open("w") as file:
        file.write(mappings_json)

def extract_upper_dirname(target: Path):
    return target.parent.as_posix().split("/")[-1]

# @print_loading_factory("Read dataset now", 10)
def read_from_VCTK(
    data_list_path: Path,
    mappings_filename: str = "mappings.json"
):
    with data_list_path.open("r") as file:
        lines = file.readlines()
        train_data_list = {line.split("|")[0]: [line.split("|")[1], line.split("|")[2].strip()] for line in lines} 

    with Path("Datalists").joinpath(mappings_filename).open("r") as file:
        mappings_json = json.load(file)
        mappings = Munch(mappings_json)

    return train_data_list, mappings
    

if __name__ == "__main__":
    _, _ = get_from_VCTK(Path("Datasets"), Path("./Datalists/audio_train_datalist.txt"), 777)
