import MeCab
from urllib import request
import re

class Tokenizer:
    def __init__(self, parser=None, include_pos=None, exclude_posdetail=None, exclude_reg=r"\d(年|月|日|時|分|秒)"):

        res = request.urlopen(
            "http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt")
        stopwords = [line.decode("utf-8").strip() for line in res]

        res = request.urlopen(
            "http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/English.txt")
        stopwords += [line.decode("utf-8").strip() for line in res]

        self.stopwords = stopwords
        #self.include_pos = include_pos if include_pos else  ["名詞", "動詞", "形容詞"]
        self.include_pos = include_pos if include_pos else ["名詞", ]
        self.include_pos_ex = [
            "名詞-一般",
            "名詞-固有名詞-一般",
            "名詞-固有名詞-人名",
            "名詞-固有名詞-人名-姓",
            "名詞-固有名詞-人名-名",
            "名詞-固有名詞-組織",
            "名詞-固有名詞-地域",
            "名詞-固有名詞-地域-国"
        ]
        self.exclude_posdetail = exclude_posdetail if exclude_posdetail else [
            "接尾", "数"]
        self.exclude_posdetail_ex = ["名詞-数", ]
        self.exclude_reg = exclude_reg if exclude_reg else r"$^"  # no matching reg
        if parser:
            self.parser = parser
        else:
            #if platform.dist()[0] == 'debian':
            #    mecab = MeCab.Tagger("-Ochasen -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
            #else:
            #    mecab = MeCab.Tagger("-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/")

            if True:
                mecab = MeCab.Tagger(
                    "-Ochasen -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
            else:
                mecab = MeCab.Tagger(
                    "-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/")

            self.parser = mecab.parse

    def tokenize(self, text, show_pos=False):
        text = re.sub(
            r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+", "", text)  # URL
        text = re.sub(r"\"?([-a-zA-Z0-9.`?{}]+\.jp)\"?", "", text)  # xxx.jp
        text = text.lower()
        l = [line.split("\t") for line in self.parser(text).split("\n")]
        res = [
            i[2] if not show_pos else (i[2], i[3]) for i in l
            if len(i) >= 4  # has POS.
            #and i[3].split("-")[0] in self.include_pos
            and i[3] in self.include_pos_ex
            #and i[3].split("-")[1] not in self.exclude_posdetail
            and i[3].split("-") not in self.exclude_posdetail_ex
            and not re.search(r"(-|−)\d", i[2])
            and not re.search(self.exclude_reg, i[2])
            and i[2] not in self.stopwords
        ]
        return res
