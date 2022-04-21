# Hyper Tutorial

## 「データの読み込み」〜「Pytorchに対応させる」を自分でやる．
### データの読み込み

Indian Pines（IP）とBotswana（BW）のデータの詳細は，それぞれ`IP.md`，`BW.md`
を参考にしてください．

### ハイパースペクトル画像からラベル付けされた画素を全て取り出す．
`IP.md`，`BW.md`で読み込んだ`target`を参考にして，ラベル付けされた画素を全て取り出してください．

### 使わないクラスを除く

### 学習（Train）と検証（Validation）

```
python train.py
```

### 評価（Test）
```
python evaluate.py
```