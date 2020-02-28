from models import URL

from categorizer import Categorizer
from finalizer import finalizer
from formatter import formatter
from tokenizer import Tokenizer


from mongoengine import connect
import re
from multiprocessing import Process, Queue
import scrapy.crawler as crawler
import uuid
from scrapy import signals
import shutil
from scrapy.utils.gz import gunzip, gzip_magic_number
from scrapy.utils.sitemap import Sitemap, sitemap_urls_from_robots
from scrapy.http import Request, XmlResponse
from scrapy.spiders import Spider
from bson.objectid import ObjectId
from langdetect import detect
import xml.etree.ElementTree as ET
from urllib.parse import urljoin
from urllib.parse import urlparse
import pickle
import pprint
import os
import base64
import scrapy
import simplejson as json
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from scrapy.selector import Selector
from twisted.internet import reactor
from scrapy.crawler import CrawlerRunner
from scrapy.utils.log import configure_logging

error_urls = []

#from janome.tokenizer import Tokenizer
try:
    import nltk.data
except:
    import imp
    import sys
    sys.modules["sqlite"] = imp.new_module("sqlite")
    sys.modules["sqlite3.dbapi2"] = imp.new_module("sqlite.dbapi2")
    import nltk

    import nltk.data


def output_pickle(dic, filename):
  output = open(filename + '.pkl', 'w')
  pickle.dump(dic, output)
  output.close()


def pp(obj):
  pp = pprint.PrettyPrinter(indent=4, width=160)
  str = pp.pformat(obj)
  return re.sub(r"\\u([0-9a-f]{4})", lambda x: unichr(int("0x"+x.group(1), 16)), str)


def load_pickle(filename):
  pkl_file = open("Supplements.pkl")
  data1 = pickle.load(pkl_file)
  print(pp(data1))
  pkl_file.close()


regex = re.compile(
    r'^(?:http|ftp)s?://'  # http:// or https://
    # domain...
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
    r'localhost|'  # localhost...
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
    r'(?::\d+)?'  # optional port
    r'(?:/?|[/?]\S+)$', re.IGNORECASE)


class Spider(scrapy.spiders.SitemapSpider):
    name = 'items'
    custom_settings = {
        'BOT_NAME': 'stand-alone',
        'HTTPERROR_ALLOWED_CODES': 'True',
        'DEFAULT_REQUEST_HEADERS': {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'jp',
            'User-Agents': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
            #'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:48.0) Gecko/20100101 Firefox/48.0'
        },
        #'USER_AGENT':'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
        #'USER_AGENT':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36',
        #'REDIRECT_ENABLED':'False',
        'scrapy.telnet.TelnetConsole': None,
        'CONCURRENT_ITEMS': os.getenv('SCRAPY_CONCURRENT_ITEMS', '1000'),
        'CONCURRENT_REQUESTS': os.getenv('SCRAPY_CONCURRENT_REQUESTS', '1000'),
        'DOWNLOAD_DELAY': float(os.getenv('SCRAPY_DOWNLOAD_DELAY', 0.1)),
        'DOWNLOAD_MAXSIZE': 1024 * 128 * 64
        #'CONCURRENT_REQUESTS_PER_DOMAIN': 30,
        #'CONCURRENT_REQUESTS_PER_IP': 10
        #'ROBOTSTXT_OBEY': True,
        #'SPIDER_MIDDLEWARES': {
        #  'scrapy.downloadermiddlewares.robotstxt.RobotsTxtMiddleware': 543
        #}

    }

    allowed_domains = []

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super(Spider, cls).from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider._handle_error,
                                signal=signals.spider_error)

        return spider

    def _handle_error(self, failure, response, spider):
        self.logger.error("### HANDLE_ERROR ###")
        self.logger.error(failure.type)
        self.logger.error(failure.getErrorMessage())

    def __init__(self, *args, **kwargs):
        super(Spider, self).__init__(*args, **kwargs)
        config = kwargs.get('config')
        self.config = config
        self.kwargs = kwargs.get('kwargs')

        self.entry_url = kwargs.get('entry_url')
        self.document_xpath = kwargs.get('document_xpath')
        self.image_xpath = kwargs.get('image_xpath')
        self.allow_rule = kwargs.get('allow_rule')
        self.deny_rule = kwargs.get('deny_rule')
        self.page_limit = kwargs.get('page_limit')
        self.request_id = kwargs.get('request_id')
        self.job_id = kwargs.get('job_id')
        self.exclude_reg = kwargs.get('exclude_reg')

        self.URL = URL

        self.logger.info('document_xpath=%s', self.document_xpath)
        self.logger.info('image_xpath=%s', self.image_xpath)
        self.logger.info('allow_rule=%s', self.allow_rule)
        self.logger.info('deny_rule=%s', self.deny_rule)
        self.logger.info('page_limit=%s', self.page_limit)
        self.logger.info('request_id=%s', self.request_id)
        self.logger.info('exclude_reg=%s', self.exclude_reg)

        if self.entry_url is not None:
            if re.match(regex, self.entry_url) is not None:
                domain = urlparse(self.entry_url).netloc
                self.allowed_domains.append(domain)
            else:
                raise Exception
        else:
            raise Exception

        self.stored_sitemap = {}

        if not os.path.exists('workspace'):
            os.mkdir('workspace')
        if not os.path.exists('workspace/temp'):
            os.mkdir('workspace/temp')

        if os.path.isfile('./workspace/temp/store.json'):
            with open('./workspace/temp/store.json', "r") as f:
                self.stored_sitemap = json.loads(f.read())

        self.enable_page_limit = False
        if self.page_limit >= 0:
            self.enable_page_limit = True
            self.max_requests = self.page_limit
        elif self.page_limit == -1:
            pass
        else:
            raise Exception

        self.request_counter = 0
        self.tokenizer = Tokenizer(exclude_reg=self.exclude_reg)

    def start_requests(self):
        yield Request(self.entry_url, self._parse_root)

    def _errback(self, response):
        url = response.request.url
        if url.endswith('/robots.txt') or url.endswith('/robots.txt/'):
            url = urlparse(url).scheme + "://" + \
                urlparse(url).netloc + "/sitemap.xml"
            yield Request(url, self._parse_root, errback=self._errback)
        elif url.endswith('/sitemap.xml'):
            url = urlparse(url).scheme + "://" + urlparse(url).netloc
            yield Request(url, self._parse_page, dont_filter=True)

    def _parse_root(self, response):

        url = response.request.url
        base_url = urlparse(url).scheme + "://" + urlparse(url).netloc

        if url.endswith('/robots.txt') or url.endswith('/robots.txt/'):
            if "Sitemap:" in str(response.body):
                yield Request(url, self._parse_sitemap,  dont_filter=True)
            else:
                yield Request(base_url, self._parse_page, dont_filter=True)
        elif url.endswith('/sitemap.xml') or url.endswith('/sitemap') or url.endswith('/sitemap/'):
            yield Request(url, self._parse_sitemap,  dont_filter=True)
        else:
            expect_url = base_url + "/robots.txt"
            yield Request(expect_url, self._parse_root, errback=self._errback)

    def _get_sitemap_body(self, response):
        """Return the sitemap body contained in the given response,
        or None if the response is not a sitemap.
        """
        if isinstance(response, XmlResponse):
            return response.body
        elif gzip_magic_number(response):
            return gunzip(response.body)

        elif response.url.endswith('.xml') or response.url.endswith('.xml.gz'):
            return response.body

        try:
            root = ET.fromstring(response.body)
            return response.body
        except:
            pass

    def _check_lastmod(self, d):

        if not d["loc"] in self.stored_sitemap:
            return True

        if not "lastmod" in self.stored_sitemap[d["loc"]]:
            return True

        return False

    def _iterloc(self, it, alt=False):
        for d in it:
            if not "loc" in d:
                yield
            elif not "lastmod" in d:
                yield d['loc'], None
            elif self._check_lastmod(d):
                yield d['loc'], d["lastmod"]

            if alt and 'alternate' in d:
                for l in d['alternate']:
                    yield l, None

    scaned_dict = []

    def _parse_page(self, response):
        sel = Selector(response)
        links = sel.xpath('//a/@href').extract()
        parsed = urlparse(response.request.url)

        for link in links:

            if link.startswith("//"):
                link = parsed.scheme + "://" + parsed.netloc + link[1:]
            elif link.startswith("/"):
                link = parsed.scheme + "://" + parsed.netloc + link
            elif not parsed.netloc == urlparse(link).netloc:
                self.logger.info('###OUTDOMAIN_URL=%s', (link))
                continue

            if link in self.scaned_dict:
                continue

            for r, c in self._cbs:
                if self.enable_page_limit and self.request_counter >= self.max_requests:
                    return
                if not self._filter(link):
                    self.request_counter += 0.2
                    break
                self.request_counter += 1
                request = Request(link, callback=c)
                request.meta['lastmod'] = None
                self.scaned_dict.append(response.request.url)
                yield request
                break

    def _parse_sitemap(self, response):

        if response.url.endswith('/robots.txt'):

            for url in sitemap_urls_from_robots(response.text, base_url=response.url):
                yield Request(url, callback=self._parse_sitemap)

        else:
            body = self._get_sitemap_body(response)
            if body is None:
                return

            s = Sitemap(body)
            it = self.sitemap_filter(s)

            if s.type == 'sitemapindex':
                for loc, lastmod in self._iterloc(it, self.sitemap_alternate_links):
                    if any(x.search(loc) for x in self._follow):
                        request = Request(loc, callback=self._parse_sitemap)
                        request.meta['lastmod'] = lastmod
                        yield request
            elif s.type == 'urlset':
                for loc, lastmod in self._iterloc(it, self.sitemap_alternate_links):
                    for r, c in self._cbs:
                        if r.search(loc):
                            if self.enable_page_limit and self.request_counter >= self.max_requests:
                                return
                            if not self._filter(loc):
                                self.request_counter += 0.2
                                break
                            self.request_counter += 1
                            request = Request(loc, callback=c)
                            request.meta['lastmod'] = lastmod
                            yield request
                            break

    def _filter(self, url):
        domain = urlparse(url).netloc
        if not domain in self.allowed_domains:
            return False

        if self.allow_rule:
            if not re.search(self.allow_rule, url):
                self.logger.info('DENY=%s', url)
                return False

        if self.deny_rule:
            if re.search(self.deny_rule, url):
                self.logger.info('DENY=%s', url)
                return False
        self.logger.info('ALLOW=%s', url)

        return True

    def closed(self, reason):

        #self.logger.info('REASON=%s',reason)

        scraped = self.URL.objects.filter(requestId=self.request_id)

        store = {}
        for url in scraped:
            if len(url.words) > 0:
                store[url.url] = {}
                if hasattr(scraped, "lastmod"):
                    store[url.url]["lastmod"] = scraped.lastmod

        json_store = json.dumps(store, indent=4, sort_keys=True)
        with open('./workspace/temp/store.json', "w") as f:
            f.write(json_store)

        passage_dir = "./passages"
        if os.path.exists(passage_dir):
            shutil.rmtree(passage_dir)
            os.mkdir(passage_dir)
        else:
            os.mkdir(passage_dir)

        input_metas = []
        separated_docs = []
        for url in scraped:
            if len(url.words) > 0:
                separated_docs.append(list(url.words))

                meta = {}
                meta["url"] = url.url
                meta["title"] = url.title
                meta["passage"] = url.passage
                meta["user_meta"] = url.user_meta
                if len(url.imageUrls) != 0:
                    meta["img_url"] = list(url.imageUrls)[0]
                else:
                    meta["img_url"] = ""

                if not os.path.exists('./passages/'+self.request_id):
                    os.mkdir('./passages/'+self.request_id)

                stream = url.passage
                stream = "\n" + stream
                stream = stream.encode("utf-8")

                passage_file_name = base64.b64encode(
                    bytes(url.url.encode("utf-8"))).decode("ascii")
                with open('./passages/{0}/{1}.txt'.format(self.request_id, passage_file_name), mode='wb') as f:
                    f.write(stream)

                input_metas.append(meta)

        json_scraped = scraped.to_json(indent=4, sort_keys=True)
        with open('./workspace/temp/scraped.json', "w") as f:
            f.write(json_scraped)

        sitemap_url = self.entry_url
        request_id = self.request_id

        res = self._create_model(separated_docs, input_metas)

        if not os.path.exists('result'):
            os.mkdir('result')

        res = finalizer(res)

        resj = json.dumps(res, indent=4, sort_keys=True)
        with open('./result/res.json', "w") as f:
            f.write(resj)

        self.scraped_length = len(scraped)
        self._export(resj)

    def _export_error(self):
        return

    def _export(self, resj):
        return

    def _create_model(self, separated_docs, input_metas):
        cat = Categorizer(separated_docs, input_metas,
                          self.entry_url, self.request_id, mode="eco")
        res, all_topics, metas = cat.train(num_topics=len(
            separated_docs), filter_n_most_frequent=0, auto=False)

        return res

    def _get_image(self, sels, response, allow_outer_domain=False):

        image_urls = []
        for img in sels:
            image_url = img.xpath("@src").extract_first()
            data_lazy_src = img.xpath("@data-lazy-src").extract_first()
            if data_lazy_src is not None:
                image_url = data_lazy_src

            img_width = img.xpath("@width").extract_first()
            img_height = img.xpath("@height").extract_first()

            image = {}

            if image_url is not None and len(image_url) > 1:
                image["url"] = urljoin(response.request.url, image_url)
            else:
                continue

            if "base64," in image["url"]:
                continue

            if img_width is None:
                img_width = 150
            if img_height is None:
                img_height = 150

            try:
                image["width"] = float(img_width)
            except:
                image["width"] = 0.1
            try:
                image["height"] = float(img_height)
            except:
                image["height"] = 0.1
            try:
                image["src"] = image_url
            except:
                image["src"] = ""

            image_urls.append(image)

        if len(image_urls) == 0:
            self.logger.info('empty0')
            return image_urls

        image_urls = list(filter(
            lambda p: p["width"]/p["height"] > 0.1 and p["height"]/p["width"] > 0.1, image_urls))

        if image_urls is None:
            self.logger.info('empty1')
            return []

        if not allow_outer_domain:
            image_urls = list(filter(lambda p: urlparse(p["url"]).netloc == urlparse(
                response.request.url).netloc, image_urls))

        if image_urls is None:
            self.logger.info('empty2')
            return []

        if not list(image_urls):
            self.logger.info('empty3')
            return []

        max_one = max(image_urls, key=lambda p: p["width"])
        image_urls = [max_one["url"], ]
        return image_urls

    def parse(self, response):

        url = response.request.url

        domain = urlparse(url).netloc
        if not domain in self.allowed_domains:
            return

        lastmod = response.request.meta["lastmod"]

        sel = Selector(response)

        title = sel.xpath('//title/text()').extract_first()

        if self.image_xpath is not None:
            image_urls = self._get_image(
                sel.xpath(self.image_xpath), response, allow_outer_domain=True)
        else:
            image_urls = self._get_image(
                sel.xpath("//article//img"), response, allow_outer_domain=True)

            if len(image_urls) == 0:
                image_urls = self._get_image(
                    sel.xpath("//img"), response, allow_outer_domain=True)

        user_meta = {}

        obj = formatter(sel)
        #obj = dict()
        #exec(self.json_format, {"sel": sel,
        #                        "re": re, "logger": self.logger}, obj)
        user_meta = obj  # ["res"]

        title = (title.replace('\n', ''))
        passage = title + "\n"
        if self.document_xpath is not None:
            passageXPath = self.document_xpath
        else:
            passageXPath = "//*[self::h1 or self::h2 or self::h3 or self::h4 or self::h5 or self::h6]/text()|//article//div/text()|//article//p/text()"
        passage_word_list = sel.xpath(passageXPath).extract()

        regex = r".+"
        for p in passage_word_list:
            pp = (p.replace('\n', ''))
            pp = (pp.replace(' ', ''))
            passage += " " + pp

        lang = detect(passage)

        if lang == 'ja':
            words = self.molph_mecab(passage)
        else:
            words = self.molph_nltk(passage)

        json_words = json.dumps(words)

        if True:
            if len(words) > 0:
                objectId = ObjectId()
                molphed = self.URL(url=url)
                molphed.title = title  # one
                molphed.lang = lang
                molphed.words = words
                molphed.words_len = len(words)
                molphed.imageUrls = image_urls
                molphed.passage = passage
                molphed.user_meta = user_meta

                molphed.lastmod = lastmod
                molphed.requestId = self.request_id
                molphed.save()

        return

    def molph_mecab(self, line):
        return self.tokenizer.tokenize(line)

    def molph_nltk(self, line):
        words = []
        #print(line)
        tokens = nltk.word_tokenize(line)
        taggeds = nltk.pos_tag(tokens)
        for tagged in taggeds:
            #print(tagged)
            w = str(tagged[1])
            if (w == 'NN' or w == 'NNS' or w == 'NNP' or w == 'NNPS') and len(tagged[0]) > 1:
                a = tagged[0]
                words.append(a.lower())  # Lowercase only
            #if re.search('NN', w):
            #    a = tagged[0]
            #    words.append(a)
        return words


#from scrapy.xlib.pydispatch import dispatcher


class ScannerWrapper():
    def __init__(self, entry_url, document_xpath=None, image_xpath=None, allow_rule=None, deny_rule=None, page_limit=1000, exclude_reg=r"\d(年|月|日|時|分|秒)"):

        self.entry_url = entry_url
        self.document_xpath = document_xpath
        self.image_xpath = image_xpath
        self.allow_rule = allow_rule
        self.deny_rule = deny_rule
        self.page_limit = page_limit
        self.exclude_reg = exclude_reg

        MONGO_ADDR = "localhost"
        MONGO_PORT = 27017
        MONGO_DB = "test"
        connect(MONGO_DB, host=MONGO_ADDR, port=MONGO_PORT)

        self.request_id = str(uuid.uuid4())
        self.job_id = str(uuid.uuid4())

    def main(self):

        return self.spy_items()
        #return self.spy_items_runner()

    def spy_items(self):

        process = CrawlerProcess(get_project_settings())

        process.crawl(Spider,
                      entry_url=self.entry_url,
                      document_xpath=self.document_xpath,
                      image_xpath=self.image_xpath,
                      allow_rule=self.allow_rule,
                      deny_rule=self.deny_rule,
                      page_limit=self.page_limit,
                      request_id=self.request_id,
                      job_id=self.job_id,
                      exclude_reg=self.exclude_reg
                      )

        process.start()
        #process.start(stop_after_crawl=False)
        #process.stop()
        return

    def spy_items_runner(self):

        self.sitemap_urls = [self.sitemap_url, ]

        def f(q):
            try:
                runner = crawler.CrawlerRunner(get_project_settings())
                deferred = runner.crawl(spider,
                                        entry_url=self.entry_url,
                                        document_xpath=self.document_xpath,
                                        image_xpath=self.image_xpath,
                                        allow_rule=self.allow_rule,
                                        deny_rule=self.deny_rule,
                                        page_limit=self.page_limit,
                                        exclude_reg=self.exclude_reg
                                        )
                deferred.addBoth(lambda _: reactor.stop())
                reactor.run()
                q.put(None)
            except Exception as e:
                q.put(e)

        q = Queue()
        p = Process(target=f, args=(q,))
        p.start()
        result = q.get()
        p.join()

        if result is not None:
            raise result

        return "{}"
