from pathlib import Path
from typing import Union

def align_pathobj(target: Union[Path, str]) -> Path:
    return target.resolve() if isinstance(target, Path) else Path(target).resolve()
