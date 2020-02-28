# Easy Customizable Scraper

Easy customizable scraping starter kit.

![Visualization](https://github.com/makotunes/easy-customizable-scraper/blob/master/workspace/figure.png)

## Features

- Web Scrapying
- Auto Language Detection
- Morphological Analysis
- Labling by main topic for each page
- Visualization with 2D map for each page


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

| Environment Variable | Description                                                                                  |
|----------------------|----------------------------------------------------------------------------------------------|
| ENTRY_URL            | (Required) Initial url to start to scan. sub paths of specified one are scanned recursively. |
| ALLOW_RULE           | Allow filter rule of target urls.                                                            |
| DENY_RULE            | Deny filter rule of target urls.                                                             |
| IMAGE_XPATH          | Xpath to get a image path.                                                                   |
| DOCUMENT_XPATH       | Xpath for root node for target range.                                                        |
| PAGE_LIMIT           | Scaned limittation of number of pages. -1 means unlimited number.                            |
| EXCLUDE_REG          | Excluded words with regular expression for morphological analysis.                           |


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

def finalaizer(res):
    pages = res["scatter"]
    pages = list(map(lambda x: x["user_meta"], pages))
    df = pd.DataFrame(pages)

    res["analyzed"] = {}
    res["analyzed"]["correlation"] = {}
    res["analyzed"]["correlation"]["time-n_howtomake"] = corr_df.loc["time", "n_howtomake"]

    return res
```

## Contrubution for example

http://recipe.hacarus.com/

> If you can not access it, please open it with secret browser.