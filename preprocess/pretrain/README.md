## 处理英文维基百科语料

### 1.下载维基语料

```bash
cd data/
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
```

### 2.安装WikiExtractor

```bash
git clone https://github.com/attardi/wikiextractor.git
cd wikiextractor/
sudo python setup.py install
```

### 3.处理维基语料

```bash
cd wikiextractor/
python3 WikiExtractor.py -o ../data/enwiki ../data/enwiki-latest-pages-articles.xml.bz2
```

### 4.转化语料为BERT需要的格式

```bash
python3 prepare_en_wiki.py --input ../data/enwiki/ --percent 100
```

