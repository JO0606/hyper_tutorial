# Botawana

## データの読み込み
`load_path`は，自分がこのデータセットを保存しているディレクトリのパスを指す．
```
import numpy as np
import scipy.io as sio

data = np.array(sio.loadmat(load_path + 'Botswana.mat')['Botswana'])

target = np.array(sio.loadmat(load_path + 'Botswana_gt.mat')['Botswana_gt'])
```