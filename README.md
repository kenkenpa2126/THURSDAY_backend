# T.H.U.R.S.D.A.Y backend

## 概要
このリポジトリは、T.H.U.R.S.D.A.Y.のバックエンドのリポジトリです。
フロントエンドのリポジトリは[こちら](https://github.com/kenkenpa2126/THURSDAY)。

## 使い方
実行時.envにAPIキーを書く必要があります

[OpenAIキー発行ページ](https://platform.openai.com/account/api-keys)

1. リンクにアクセス 
2. 2Create a new API keyをクリック 
3. API key nameに任意の名前を入力  
4. Create secret keyを作成 
5. .envに以下のように記述
```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

### Qdrantを起動する

dockerをインストールして以下のコマンドを実行してください。
```
docker pull qdrant/qdrant
docker run -p 6333:6333 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

### 文書の配置
pdfフォルダ直下に読み込ませたいpdfまたはテキストファイルを配置してください。
なお、デスクトップアプリからアップロードできるのは現状pdfファイルのみです。

### サーバーの起動
以下のコマンドを実行してください。(.env.sampleを参考に.envを作成してください。
```
$ docker-compose up --build
```

### API docs
以下のURLにアクセスすると、APIのドキュメントが見れます。
```
http://localhost:8000/docs#/
```
