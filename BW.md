# Botawana

## データの読み込み
`load_path`は，自分がこのデータセットを保存しているディレクトリのパスを指す．
```
import numpy as np
import scipy.io as sio

data = np.array(sio.loadmat(load_path + 'Botswana.mat')['Botswana'])

target = np.array(sio.loadmat(load_path + 'Botswana_gt.mat')['Botswana_gt'])
```

## クラス毎のピクセル数
- class0はラベル付けされていないピクセルなので，実験では使わない．

![pic](./figs/bw_class.png)


## 実験で使うクラスの選択
- 「クラス毎のピクセル数」に添付した画像のclass14のピクセル数（95）に他のクラスも合わせるため，ランダムにピクセルを選択した（重複がないように選択する）．
- 一例として，`numpy.random.choice()`は重複なしでピクセルを取り出せると思う．


![pic](./figs/use_bw_class.png)

