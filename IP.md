# Indian Pines

## データの読み込み
`load_path`は，自分がこのデータセットを保存しているディレクトリのパスを指す．
```
import numpy as np
import scipy.io as sio

data = np.array(sio.loadmat(load_path + 'Indian_pines_corrected.mat')['indian_pines_corrected'])

target = np.array(sio.loadmat(load_path + 'Indian_pines_gt.mat')['indian_pines_gt'])
```
