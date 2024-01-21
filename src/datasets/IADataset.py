from munch import Munch
from torch.utils.data import Dataset, DataLoader

class IADataset(Dataset):
    """
    要件：
        画像データセットと音声データセットとでドメインごとのデータ数を揃える
        ドメイン⇒男・女
        今回は圧倒的に少ないと思われる音声に合わせるが別verもすぐに導入できるようにしたい
    """
    
    def __init__(self):
        self.row_data: Munch = get_row_data()
