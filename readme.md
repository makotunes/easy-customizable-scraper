# Easy Customizable Scraper

Easy customizable scraping starter kit.

General-purpose Web scraping tool with text analysis function.

The following features help users start development.

- Easy setup
- Customizability
- Text analysis function (tagging / visualization)

![Visualization](https://github.com/makotunes/easy-customizable-scraper/blob/master/workspace/figure.png)

## Features

- Web scraping
- Automatic language detection
- Morphological analysis
- Feature tagging algorithm (original)
- 2D map visualization technology (original)


## Dependency

- Docker

## Install

```Shell
docker build -t scanner .
```

or

```Shell
./build.sh
```

## Run

```Shell
docker run --rm -it -v "$PWD":/usr/src/app \
--name scanner --force scanner  \
-e 'ENTRY_URL=http://recipe.hacarus.com/' \
-e 'ALLOW_RULE=/recipe/' \
-e 'IMAGE_XPATH=//*[@id="root"]/div/div/section/div/div/div[1]/figure/img' \
-e 'DOCUMENT_XPATH=//td/text()|//p/text()' \
-e 'PAGE_LIMIT=2000' \
-e 'EXCLUDE_REG=\d(年|月|日|時|分|秒|ｇ|\u4eba|\u672c|cm|ml|g|\u5206\u679a\u5ea6)|hacarusinc|allrightsreserved' \
scanner:latest /usr/src/app/entrypoint.sh
```

or 

```Shell
./run.sh
```

## Parametes

Set Environment Variable of Docker Container.

If you have at least ENTRY_URL, it will automatically scan the page and pull out the text.
If no options are specified, it is optimized for curated media and can be fully automated, such as extracting the text of articles.

| Environment Variable | Description                                                                                  |
|----------------------|----------------------------------------------------------------------------------------------|
| ENTRY_URL            | (Required) Site top URL to start scanning. All the pages are automatically scanned.          |
| ALLOW_RULE           | Allow filter rule of target urls.                                                            |
| DENY_RULE            | Deny filter rule of target precedence overurls.                                              |
| IMAGE_XPATH          | Specify the image you want to get on the page with XPATH.                                    |
| DOCUMENT_XPATH       | XPATH of the top node in the page where text is to be extracted.                             |
| PAGE_LIMIT           | Scaned limittation of number of pages. -1 means unlimited number.                            |
| EXCLUDE_REG          | Regular expression of word rule not extracted by morphological analysis.                     |


## Result

result/res.json

## Customize

#### custom/_formatter.py

Edit XPATH for required HTML nodes like below.

```Python
def formatter(sel):
    res = {}

    n_howtomake = int(len(sel.xpath('//*[@id="root"]/div/div/section/div/div/div[2]/div[1]/table[2]/tbody/tr/td/text()').extract()) / 2)
    res["n_howtomake"] = n_howtomake

    return res
```


#### custom/_finalizer.py

Edit post-process to generate your expected output like below.


```Python
import pandas as pd

def finalizer(res):
    pages = res["scatter"]
    pages = list(map(lambda x: x["user_meta"], pages))
    df = pd.DataFrame(pages)

    corr_df = df.loc[:, ["time", "n_howtomake", "n_components"]].corr()

    res["analyzed"] = {}
    res["analyzed"]["correlation"] = {}
    res["analyzed"]["correlation"]["time-n_howtomake"] = corr_df.loc["time", "n_howtomake"]

    return res
```

## Contrubution for example

http://recipe.hacarus.com/

> If you can not access it, please open it with secret browser.

## Sample Result

### Automatic tagging

result/tagged.csv

|title                           |tag1  |tag2     |tag3    |
|--------------------------------|------|---------|--------|
|なすとトマトの中華和え(１５分)                |なす    |トマト      |大葉      |
|ぶりの照り焼き(45分)                    |両面    |照り焼き     |水気      |
|おでん風煮(2時間)                      |大根    |こんにゃく    |竹輪      |
|大根とツナのサラダ(15分)                  |ツナ    |大根       |わかめ     |
|鶏の照り焼き丼(20分)                    |片栗粉   |にんにく     |れんこん    |
|筑前煮(６０分)                        |れんこん  |ごぼう      |こんにゃく   |
|白菜とわかめの酢の物(15分)                 |白菜    |わかめ      |しめじ     |
|鮭のホイル焼き(25分)                    |玉ねぎ   |しめじ      |ピーマン    |


### 2D map visualization

![Visualization](https://github.com/makotunes/easy-customizable-scraper/blob/master/workspace/figure.png)
