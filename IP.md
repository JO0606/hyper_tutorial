# Indian Pines

## データの読み込み
`load_path`は，自分がこのデータセットを保存しているディレクトリのパスを指す．
```
import numpy as np
import scipy.io as sio

data = np.array(sio.loadmat(load_path + 'Indian_pines_corrected.mat')['indian_pines_corrected'])

target = np.array(sio.loadmat(load_path + 'Indian_pines_gt.mat')['indian_pines_gt'])
```

## クラス毎のピクセル数
- class0はラベル付けされていないピクセルなので，実験では使わない．

![pic](./figs/ip_class.png)


## 実験で使うクラスの選択
- 「クラス毎のピクセル数」に添付した画像のclass13のピクセル数（205）に他のクラスも合わせるため，ランダムにピクセルを選択した（重複がないように選択する）．
- 205ピクセル未満のクラスは実験で使わない．
- 一例として，`numpy.random.choice()`は重複なしでピクセルを取り出せると思う．


![pic](./figs/use_ip_class.png)

