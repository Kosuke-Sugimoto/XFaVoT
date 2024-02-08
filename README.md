## 利用方法
- Local環境内
```bash
docker compose up -d --build
docker compose exec vc bash
```
- Docker環境内
```bash
poetry install --no-root
pip install matplotlib
```

## Pythonにおけるprint関数のカーソル移動について
一般にANSIエスケープコードとして知られている模様  
基本的なカーソル移動のエスケープコードは
- カーソル上移動：`\033[{n}A`：n行上に移動する
- カーソル下移動：`\033[{n}B`：n行下に移動する
- カーソル右移動：`\033[{n}C`：n個右に移動する
- カーソル左移動：`\033[{n}D`：n個左に移動する
- カーソル位置保存：`\033[s`：現在のカーソル位置を保存
- カーソル位置ロード：`\033[u`：保存したカーソル位置をロード
- 行の残りをクリア：`\033[K`

## try+finally+return
finallyはtryブロック内でエラーが発生しようとしなかろうと必ず実行されるブロック  
では、try内でreturnに到達したらどうなるのか？  
　⇒return直前にfinallyの処理が割り込まれる  
今回はそれを利用してreturnで抜ける直前にスレッドへとイベントをセットしている([このファイル](src/utils/print_loading.py)のl.33～l.36)

## 注意
`StarGANv2-VC/Utils/ASR/epoch_00100.pth` が 87.44MB であり、GitHub の 推奨 size が 50.00MB のため少々オーバーしている  
別環境で構築した際、これが開けない場合が考えられるので再ダウンロードも考えた方が良い
