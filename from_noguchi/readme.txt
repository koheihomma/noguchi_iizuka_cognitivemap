データセット(data.h5)
    hdf5形式．
    エージェントの視覚・運動，位置の時系列データが含まれる．（位置は学習には使わない）

    中身:
        train: 学習用 (論文でいうrestricted area に入らない)
        test, test2: テスト用 (restricted area に入る)
            test2はフィールドの中心と4つの角を直線的に移動 (論文7.4の実験用)

        データ数：
            train: 100系列
            test: 10系列
            test2: 10系列

        ※ 1系列の長さ: 1001


各データの説明
    vision: 画像を1次元に直したもの (show.py中で可視化するために2次元画像に再整形している)
        size: 768
    motion: エージェントの x, y 位置の変位
        size: 2
    position: エージェントの x, y 位置
        size: 2



データ可視化プログラム
    vision, motion, position の系列をアニメーションで表示

    実行方法:

        $ python show.py train 0

        引数1: データセットの選択 (train, test, test2)
        引数2: 表示するデータのインデックス

