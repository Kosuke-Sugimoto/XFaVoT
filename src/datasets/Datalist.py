import random
from munch import Munch
from pathlib import Path
from collections import defaultdict
from typing import Union, Generator
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split

from src.utils.align_pathobj import align_pathobj

class DatalistInterface(ABC):
    """
    Note:
    __init__()にて必要な情報はsetすること

    Munchは型引数を取らないため、型チェックではMunchオブジェクトかどうかまでしか判定しない
    　⇒今回指定しているのは、プログラムの可読性を上げるため
    """

    @abstractmethod
    def setup() -> None:
        raise NotImplementedError
    
    @abstractmethod
    def get() -> tuple[ Munch[list, list], dict[str, Munch] ]:
        raise NotImplementedError
    
class VCTKDatalist(DatalistInterface):

    def __init__(
        self,
        seed: int,
        test_size: float,
        data_dir: Union[Path, str],
        info_txt_filepath: Union[Path, str],
        ext: str = "wav",
        max_used_ids: int = 20
    ):
        self.seed: int = seed
        self.test_size: float = test_size # for train_test_split
        self.data_dir: Path = align_pathobj(data_dir)
        self.info_txt_filepath: Path = align_pathobj(info_txt_filepath)
        self.ext = ext
        self.max_used_ids = max_used_ids

    def setup(self):

        random.seed(self.seed)

        # 形式
        # id: label obj
        self.label_info: dict[ str, Munch[str, str, str, str, str] ] = \
            self.__parse_info(self.info_txt_filepath)
        
        self.datalist: Munch(list, list) = \
            self.__walk_dir(self.ext, self.data_dir)
        
        # 形式
        # path: label obj
        # pathから抜き出したidはp{}の形だが、label_infoのkeyはpなし
        # speaker-info.txtには情報がないものも存在するのでそいつは除外
        self.labels: dict[ Path, Munch[str, str, str, str, str] ] = \
            {path: self.label_info[self.__extract_id_from_pathobj(path)[1:]]
                for path in self.pathlist if self.__extract_id_from_pathobj(path)[1:] in self.label_info.keys()}
        
        self.used_ids: list[ str ] = \
            list(map(lambda x: x, list(set([self.labels[path].id for path in self.datalist.train]))))
        
        # M/F の分、2つほどずらす
        self.usedid2idx: dict[ str, int ] = \
            dict(zip(self.used_ids, range(2, len(self.used_ids)+2)))
        
        self.usedid2idx_reverse: dict[ int, str ] = \
            dict(zip(range(2, len(self.used_ids)+2), self.used_ids))

    def get(self):
        return self.datalist, self.labels, self.usedid2idx, self.usedid2idx_reverse, self.id2trval

    def __parse_info(
        self,
        info_txt_filepath: Path
    ) -> dict[ str, Munch[str, str, str, str, str] ]:
        
        lines: str = info_txt_filepath.read_text()
        lines: list[str] = lines.split("\n")

        def parse_line(line: str) -> Munch[str, str, str, str, str]:
            parts:list = [part for part in line.split(" ") if part]
            return Munch(
                id=parts[0],
                age=parts[1],
                gender=parts[2],
                accents=parts[3],
                region="".join(parts[4:])
            )

        # ヘッダと末尾の改行による空要素は無視
        lines: list[Munch] = [parse_line(line) for line in lines[1:-2]]

        label_info: dict[ str, Munch[str, str, str, str, str] ] = \
            {line.id: line for line in lines}
        
        return label_info
    
    def __extract_id_from_pathobj(
        self,
        target: Path
    ) -> str:
            return target.parent.as_posix().split("/")[-1]
    
    def __walk_dir(
        self,
        ext: str,
        data_dir: Path
    ) -> Munch[list, list]:
        """
        Note:
        walkはディレクトリ構造を再帰的に走査することを意味するらしい
        """

        path_generator: Generator[Path, None, None] = data_dir.glob(f"**/*.{ext}")
        # generatorは一度イテレートするとその要素を「消費」し、二度と使えなくなってしまう
        # 使い勝手が悪すぎるのでlistに変換しておく
        # 一応pathlistのみselfでも保管
        pathlist: list[Path] = list(path_generator)
        self.pathlist: list[Path] = pathlist

        id_set: set[str] = set([self.__extract_id_from_pathobj(path) for path in pathlist])
        id2paths: dict[list] = \
            {id: [path for path in pathlist if id in path.as_posix()] for id in id_set}
        
        def train_val_split_per_id(
            target: dict[list],
            seed: int = self.seed,
            test_size: float = self.test_size
        ) -> dict[ Munch[list, list] ]:
            id2trval = {}
            for id, paths in target.items():
                train, val = train_test_split(
                    paths,
                    test_size=test_size,
                    random_state=seed,
                    shuffle=True
                )
                id2trval[id] = Munch(train=train, val=val)

            return id2trval

        id2trval: dict[ Munch[list, list] ] = train_val_split_per_id(id2paths)
        self.id2trval = id2trval

        def make_datalist(
            this_id2trval: dict[ Munch[list, list] ],
            num_use_ids: int = 20
        ) -> Munch[list, list]:
            train, val = [], []
            cnt = 0
            existed_ids = list(set(map(lambda x: f"p{x.id}", list(self.label_info.values()))))
            for id, trval in this_id2trval.items():
                if id not in existed_ids:
                    continue
                if cnt >= num_use_ids:
                    break
                train.extend(trval.train)
                val.extend(trval.val)
                cnt += 1
            return Munch(train=train, val=val)

        datalist: Munch[list, list] = make_datalist(id2trval, self.max_used_ids)

        return datalist

class CelebAHQDatalist(DatalistInterface):

    def __init__(
        self,
        seed: int,
        data_dir: Union[Path, str],
        ext: str = "jpg"
    ):
        self.seed: int = seed
        self.data_dir: Path = align_pathobj(data_dir)
        self.ext: str = ext

    def setup(self):

        random.seed(self.seed)

        self.datalist, self.labels = self.__walk_dir()

    def get(self):
        return self.datalist, self.labels, self.label2paths

    def __walk_dir(self) -> tuple[ Munch[list, list], dict[ Path, Munch[str]] ]:
        """
        Note:
        返り値は
        {}.train, {}.valでアクセスできるパス一覧
        {}[path].genderでアクセス可能なパスに対応するドメイン一覧
        """

        path_generator: Generator[Path, None, None] = self.data_dir.glob(f"**/*.{self.ext}")
        pathlist: list[Path] = list(path_generator)

        labels: dict[ Path, Munch[str] ] = \
            {path: Munch(gender=self.__extract_part_from_pathobj(-1, path)) for path in pathlist}
        
        datalist: Munch[list, list] = \
            Munch(
                train = [path for path in pathlist if self.__extract_part_from_pathobj(-2, path) == "train"],
                val = [path for path in pathlist if self.__extract_part_from_pathobj(-2, path) == "val"]
            )
        
        label2paths: dict[ str, list[Path] ] = defaultdict(list)
        for key, value in labels.items():
            this_key = 0 if(value.gender=="male") else 1
            label2paths[this_key].append(key)
        self.label2paths = label2paths
        
        return datalist, labels

    def __extract_part_from_pathobj(
        self,
        idx: int,
        target: Path
    ) -> str:
        return target.parent.as_posix().split("/")[idx]


if __name__=="__main__":
    vctk = VCTKDatalist(
        777,
        0.1,
        "./Datasets/VCTK-Corpus",
        "./Datasets/VCTK-Corpus/speaker-info.txt"
    )
    vctk.setup()
    print("===== VCTK Test =====")
    print(vctk.get()[0].train[:10])
    print(vctk.get()[1].popitem())

    print()

    celeb = CelebAHQDatalist(
        777,
        "./Datasets/CelebA-HQ"
    )
    celeb.setup()
    print("===== CelebA-HQ Test =====")
    print(celeb.get()[0].train[:10])
    print(celeb.get()[1].popitem())