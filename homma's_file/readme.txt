データセット(data_homma.h5)
    hdf5形式．
    エージェントの視覚・運動，位置の時系列データが含まれる．（位置は学習には使わない）

    中身:
        train: 学習用 
	test: 未実装
  
        データ数：
            train: 100系列
	    test: 10系列

        ※ 1系列の長さ: 1003


各データの説明
    vision: 画像を1次元に直したもの (show_homma.py中で可視化するために2次元画像に再整形している)
        size: 768
    motion: エージェントの x, y 位置の変位
        size: 2
    position: エージェントの x, y 位置
        size: 2



データ可視化プログラム(show_homma.py)
    vision, motion, position の系列をアニメーションで表示

    実行方法:

        $ python show_homma.py data_homma.h5 train 0

	引数1: 表示するデータセットのpath
        引数2: データセットの選択 (train, test)
        引数3: 表示するデータのインデックス


学習モデル(noguchi_iizuka_1L_homma.py)
    単層GRUによる学習
    コードのhdfpathの変数を変えることでデータを指定する
    前から1系列1000ステップ取ってきて学習(1003ステップあっても同様)

    実行方法:
	
	$ python noguchi_iizuka_1L_homma.py --train
	
	モデルパラメータ(gru.pt)とlossデータ(loss.txt)を出力

	$ python noguchi_iizuka_1L_homma.py --test
	vision出力データ(vision_output.txt)とmotion出力データ(vision_output.txt)と
	lossデータ(test_loss.txt)を出力
	

