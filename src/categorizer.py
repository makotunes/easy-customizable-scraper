import sys
import io

from sklearn.manifold import TSNE
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import matplotlib
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from collections import defaultdict
from gensim.models.ldamodel import LdaModel
import gensim
import logging
import os
import pickle
import pprint
import numpy as np
import math
import time
from gensim import models, corpora
from gensim.test.utils import datapath
import random
import simplejson as json
SCAN_ONLY = int(os.getenv('SCAN_ONLY', 1))
pp = pprint.PrettyPrinter(width=41, compact=True)
LOG = logging.getLogger()
LOG.setLevel('INFO')


meeting_id = 1
upper_limit_topics = 10
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


matplotlib.use('Agg')  # -----(1)


class Categorizer:
    def __init__(self, separated_docs, input_metas, sitemap_url, request_id, mode="eco", logger=logging.getLogger()):

        self.LOG = logger
        self.LOG.setLevel('ERROR')
        self.LOG.info("start!!!")

        self.separated_docs = separated_docs
        self.input_metas = input_metas

        self.sitemap_url = sitemap_url
        self.request_id = request_id

        self.MAX_PER_PATITION = 500

        self.n_node = 8

        self.np_metas = None

        self.mode = mode

        if not os.path.exists('workspace'):
            os.mkdir('workspace')
        if not os.path.exists('workspace/temp'):
            os.mkdir('workspace/temp')

        return

    def load_premodel(self):
        model = gensim.models.KeyedVectors.load_word2vec_format(
            'workspace/model.vec', binary=False)

    def partitionizer(self):
        retrain = False

        for sc in self.res["partitionized_scatters"]:
            #print(len(sc))
            if len(sc) > self.MAX_PER_PATITION:
                #return False

                retrain = True
                break
        if retrain:
            print("RETRAIN!")
            topics = len(self.res["partitionized_scatters"])+1
            print(topics)
            #topics = 0
            self.train(num_topics=topics, filter_n_most_frequent=20)
            self.build_response()

    def _cos_sim(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def update_doc(self):
        return

    def get_doc(self):
        cooccurrence_matrix_with_scatter = self.cooccurrence_matrix_with_scatter
        return cooccurrence_matrix_with_scatter

    def predict_doc(self, other_texts):
        dmodel = self.dmodel
        dmodel = Doc2Vec.load("workspace/docmodel")

        other_documents = [TaggedDocument(doc, [i])
                           for i, doc in enumerate(other_texts)]
        dmodel.build_vocab(documents=other_documents,
                           corpus_file=None, update=True)
        dmodel.train(self.documents,
                     total_examples=dmodel.corpus_count, epochs=dmodel.iter)

        entities = []
        for l, ldoc in enumerate(self.documents):
            tag = ldoc.tags[0]
            url = self.input_metas[tag]["url"]
            dis = dmodel.docvecs.distance(
                ldoc.tags[0], other_documents[0].tags[0])
            sim = dmodel.docvecs.similarity_unseen_docs(
                dmodel, other_texts[0], ldoc.words)
            entities.append((tag, url, dis, sim))
        entities.sort(key=lambda x: x[3])
        entities.reverse()

        self.dmodel = dmodel

        dmodel.save("workspace/docmodel")

    n_node = 8

    def train_doc(self, num_topics=0, filter_n_most_frequent=30):
        molph_corpus = self.separated_docs
        documents = [TaggedDocument(doc, [i])
                     for i, doc in enumerate(molph_corpus)]
        self.documents = documents

        dmodel = Doc2Vec(documents)

        cooccurrence_matrix = []

        for l, ldoc in enumerate(documents):
            entities = []
            subtag = 0
            for m, mdoc in enumerate(documents):
                if l == m:
                    continue
                tag = mdoc.tags[0]

                url = self.input_metas[tag]["url"]

                dis = dmodel.docvecs.distance(ldoc.tags[0], tag)
                sim = dmodel.docvecs.similarity(ldoc.tags[0], tag)

                entities.append((tag, url, dis, sim))

            entities.sort(key=lambda x: x[3])
            entities.reverse()

            if l == 99:
                print(entities)
            cooccurrence_matrix.append(entities[0:self.n_node+1])

        cooccurrence_matrix_with_scatter = []
        for l, ldoc in enumerate(cooccurrence_matrix):
            entities = []
            subtag = 0
            normsim = 0
            for m, mdoc in enumerate(ldoc):
                if m == 0:
                    subtag = mdoc[0]
                    x = 0
                    y = mdoc[3]
                    normsim = y
                subsim = dmodel.docvecs.similarity(subtag, mdoc[0])
                y = (3*(subsim**2) + (sim**2) - 3 * (normsim**2))/-4*normsim
                x = ((y**2) + (sim**2)) ** (1/2)
                entities.append((mdoc[0], mdoc[1], float(mdoc[2]), float(
                    mdoc[3]), float(subsim), float(x), float(y)))
            cooccurrence_matrix_with_scatter.append(entities)

        dmodel.save("workspace/docmodel")
        self.cooccurrence_matrix_with_scatter = cooccurrence_matrix_with_scatter

        self.dmodel = dmodel

    def train(self, num_topics=0, filter_n_most_frequent=30, auto=False):
        start = time.time()
        self.lda(num_topics=num_topics, filter_n_most_frequent=30, auto=auto)
        self.build_response()
        self.compress_dimension()
        print("elapsed_time:{0}, volume: {1}".format(
            time.time() - start, len(self.separated_docs)))

        return self.res, None, None

    def train_update(self, other_texts, input_metas, num_topics=0, filter_n_most_frequent=30):
        start = time.time()

        self.input_metas.extend(input_metas)

        self.get_modelset()

        other_corpus = [self.dictionary.doc2bow(text) for text in other_texts]

        self.model.update(other_corpus)
        self.dictionary.add_documents(other_texts)

        self.corpus.append(other_corpus)

        topics = self.model.get_document_topics(
            other_corpus, minimum_probability=0)

        self.build_response()
        self.compress_dimension()
        res = self.get_response()
        print("elapsed_time:{0}, volume: {1}".format(
            time.time() - start, len(self.separated_docs)))
        return res

    def get_modelset(self):
        self.model = LdaModel.load("workspace/model/ldamodel")

        self.dictionary = corpora.Dictionary.load_from_text(
            './workspace/dictionary/{0}.txt'.format(self.request_id))

        self.mmcorpus = corpora.MmCorpus(
            './workspace/mmcorpus/{0}.mm'.format(self.request_id))

        with open('workspace/temp/corpus.pkl', 'rb') as f:
            self.corpus = pickle.load(f)

        self.matlix = np.genfromtxt(
            "./workspace/temp/doc_lda_tensor.tsv", delimiter="\t")
        np_metas = np.genfromtxt(
            "./workspace/temp/doc_lda_metadata.tsv", delimiter="\t")
        self.np_metas = np.delete(np_metas, 0, 0)

        if self.mode == "full":
            with open('./workspace/temp/model_meta.json', "r") as f:
                self.model_meta = json.loads(f.read())

    def lda(self, update=False, num_topics=0, filter_n_most_frequent=30, auto=False):

        logging.getLogger("gensim").setLevel(logging.ERROR)
        molph_corpus = self.separated_docs

        if self.mode == "full":
            self.frequency = defaultdict(int)
            for text in molph_corpus:
                for token in text:
                    self.frequency[token] += 1

        self.dictionary = corpora.Dictionary(molph_corpus)
        if not os.path.exists('workspace/dictionary'):
            os.mkdir('workspace/dictionary')
        self.dictionary.save_as_text(
            './workspace/dictionary/{0}.txt'.format(self.request_id))

        self.dictionary.filter_n_most_frequent(filter_n_most_frequent)
        self.corpus = [self.dictionary.doc2bow(text) for text in molph_corpus]

        with open('workspace/temp/corpus.pkl', 'wb') as f:
            pickle.dump(self.corpus, f)

        if not os.path.exists('workspace/mmcorpus'):
            os.mkdir('workspace/mmcorpus')
        corpora.MmCorpus.serialize(
            './workspace/mmcorpus/{0}.mm'.format(self.request_id), self.corpus)

        if len(self.separated_docs) < 3:
            return False
        elif len(self.separated_docs) == 3:
            test_rate = 1 / len(self.separated_docs)
        else:
            test_rate = 1 / 4

        rate_or_size = test_rate
        most = [1.0e30, None]
        training, test = self.split_corpus(self.corpus, rate_or_size)

        topics = num_topics
        if not auto:

            self.model = models.LdaModel(
                corpus=self.corpus, id2word=self.dictionary, num_topics=topics)
        else:

            if topics == None or topics == 0:
                topic_range = range(2, upper_limit_topics + 1)
            else:
                topic_range = range(topics, topics+1)

            for t in topic_range:
                m = models.LdaModel(
                    corpus=training, id2word=self.dictionary, num_topics=t, iterations=250, passes=5)

                p1 = np.exp(-m.log_perplexity(training))
                p2 = np.exp(-m.log_perplexity(test))

                if p2 < most[0]:
                    most[0] = p2
                    most[1] = m

                self.perplexity, self.model = most[0], most[1]

        if isinstance(self.model, type(None)) == True:
            raise

        temp_file = datapath("model")
        if not os.path.exists('workspace/model'):
            os.mkdir('workspace/model')
        self.model.save("workspace/model/ldamodel")

        if self.mode == "full":
            self.num_of_trainning = sum(
                count for self.dictionary in training for id, count in self.dictionary)
            self.num_of_test = sum(
                count for self.dictionary in test for id, count in self.dictionary)

            self.model_meta = {}
            self.model_meta["num_words"] = self.num_of_trainning + \
                self.num_of_test
            self.model_meta["num_words_trainning"] = self.num_of_trainning
            self.model_meta["num_words_test"] = self.num_of_test

            with open('./workspace/temp/model_meta.json', "w") as f:
                model_meta = json.dumps(
                    self.model_meta, indent=4, sort_keys=True)
                f.write(model_meta)

        return

    def build_response(self, num_words=10):

        sample_size = len(self.separated_docs)
        width = 1

        d_topics = []
        did_topics = {}
        t_documents = {}
        samples = random.sample(range(len(self.corpus)), sample_size)

        for s in samples:
            ts = self.model.__getitem__(self.corpus[s], -1)
            d_topics.append([v[1] for v in ts])
            max_topic = max(ts, key=lambda x: x[1])

            did_topics[s] = max_topic
            if max_topic[0] not in t_documents:
                t_documents[max_topic[0]] = []
            t_documents[max_topic[0]
                        ] += [(s, float(max_topic[1]), self.input_metas[s]["url"])]

        result = {}

        for t in t_documents:
            sorted_docs = sorted(
                t_documents[t], key=lambda x: x[1], reverse=True)

            out = {}
            for doc in sorted_docs:
                out[doc[2]] = doc[1]

            result[t] = out

        document_topics, topic_documents = d_topics, t_documents

        topic_top10_tags = {}
        topic_labels = []

        for cell in self.model.show_topics(num_topics=-1, num_words=num_words, formatted=False):
            topic_top10_tags[cell[0]] = {}

            for k, w in enumerate(cell[1]):
                topic_top10_tags[cell[0]][w[0]] = float(w[1])

            tmps = []
            label = ""

            for k, v in sorted(topic_top10_tags[cell[0]].items(), key=lambda x: -x[1]):
                tmp = {}
                tmp["word"] = k
                tmp["freq"] = v
                tmps.append(tmp)
                label += k + " "

            topic_top10_tags[cell[0]] = tmps

            topic_labels.append(label)

        articles = {}
        for tpc in result:
            for a in result[tpc]:
                articles[a] = str(tpc)

        metas = []
        for l, name in enumerate(self.input_metas):

            meta = {}
            meta["name"] = name["url"]
            meta["topic"] = articles[name["url"]]

            bwords = []
            for tp in self.model.get_document_topics(self.corpus[l], per_word_topics=False):
                _bwords = self.model.show_topic(tp[0])
                bwords.extend(_bwords)
            bwords = list(sorted(bwords, key=lambda x: -x[1]))
            bwords = list(map(lambda x: x[0], bwords))
            bwords = list(
                filter(lambda x: x in self.separated_docs[l], bwords))
            bwords = list(dict.fromkeys(bwords))

            if len(bwords) >= 3:
                bwords = bwords[0:3]
            else:
                import collections
                l = self.separated_docs[l]
                c = collections.Counter(l)
                __bwords = c.most_common()
                __bwords = list(map(lambda x: x[0], __bwords))
                __bwords = list(dict.fromkeys(__bwords))
                bwords = (bwords + __bwords)[0:3]

            meta["labels"] = bwords

            meta["title"] = name["title"]
            meta["passage"] = name["passage"]

            meta["img_url"] = name["img_url"]
            meta["user_meta"] = name["user_meta"]
            metas.append(meta)

        self.metas = metas

        all_topics = self.model.get_document_topics(
            self.corpus, minimum_probability=0)

        #print(all_topics[0])

        res = {}

        res["topic_details"] = topic_top10_tags

        res["categorized_doc"] = result
        res["num_topics"] = self.model.num_topics
        res["topic_labels"] = topic_labels
        res["perplexity"] = self.perplexity if hasattr(
            self, "perplexity") else 0

        if self.mode == "full":
            res["num_words"] = self.model_meta["num_words"]
            res["num_words_trainning"] = self.model_meta["num_words_trainning"]
            res["num_words_test"] = self.model_meta["num_words_test"]

            self.frequency = defaultdict(int)
            for text in self.separated_docs:
                for token in text:
                    self.frequency[token] += 1

            res["words"] = self.frequency

        res["sitemap_url"] = self.sitemap_url
        res["request_id"] = self.request_id

        self.all_topics = all_topics
        self.metas = metas
        self.res = res

        with open('./workspace/temp/doc_lda_tensor.tsv', 'w') as w:
            for doc_topics in self.all_topics:
                for l, topics in enumerate(doc_topics):
                    w.write(str(topics[1]))
                    if len(doc_topics)-1 > l:
                        w.write("\t")
                w.write("\n")

        self.mat_metas = []
        with open('./workspace/temp/doc_lda_metadata.tsv', 'w', encoding='utf-8') as w:
            w.write('Titles\tGenres\n')
            for m in self.metas:
                w.write("%s\t%s\n" % (m["name"], m["topic"]))
                self.mat_metas.append([m["name"], m["topic"]])

    def compress_dimension(self):

        res = self.res
        metas = self.metas

        matrix = [[topics[1] for l, topics in enumerate(
            doc_topics)] for doc_topics in self.all_topics]

        np_matrix = np.array(matrix)

        if self.np_metas is None:
            self.np_metas = np.array(self.mat_metas)

        decomp = TSNE(n_components=2)
        X_decomp = decomp.fit_transform(np_matrix)

        all_diffs = np.expand_dims(X_decomp, axis=1) - \
            np.expand_dims(X_decomp, axis=0)
        degree_distance = np.sqrt(np.sum(all_diffs ** 2, axis=-1))
        distance = (1 * degree_distance).astype(np.int32)

        sorted_distance = np.sort(distance, axis=1)
        sorted_index = np.argsort(distance, axis=1)

        ranked_index = sorted_index[:, 1:self.n_node+1]
        ranked_distance = sorted_distance[:, 1:self.n_node+1]

        ranked_index = ranked_index.tolist()
        ranked_distance = ranked_distance.tolist()

        decomp_list = X_decomp.tolist()

        cmap = get_cmap("tab10")
        scatter = []
        for l, meta in enumerate(metas):

            i = int(self.np_metas[l][1])

            marker = "$" + str(i) + "$"
            plt.scatter(X_decomp[l, 0], X_decomp[l, 1],
                        marker=marker, color=cmap(i))

            point = {}
            point["title"] = meta["title"]
            point["url"] = meta["name"]

            _topic = res["topic_labels"][int(meta["topic"])].split()

            point["topic"] = meta["labels"]

            point["topic_id"] = int(meta["topic"])
            point["passage"] = meta["passage"]

            if meta["img_url"] != "":
                point["img_url"] = meta["img_url"]
            else:
                point["img_url"] = None
            point["x"] = decomp_list[l][0]
            point["y"] = decomp_list[l][1]

            point["user_meta"] = meta["user_meta"]

            neighborhoods = [{
                "url": self.input_metas[idx]["url"],
                "distance": ranked_distance[l][m],
                "title": metas[idx]["title"],
                "passage": metas[idx]["passage"],
                "topic_id": int(metas[idx]["topic"]),
                "img_url": metas[idx]["img_url"] if metas[idx]["img_url"] != "" else None,
                "x": decomp_list[idx][0],
                "y": decomp_list[idx][1]

            } for m, idx in enumerate(ranked_index[l])]

            if not SCAN_ONLY or len(metas)-1 == l:
                point["neighborhoods"] = neighborhoods
            scatter.append(point)

        plt.title(f"t-SNE")
        plt.savefig('./workspace/figure.png')

        res["scatter"] = scatter

        resj = json.dumps(res, indent=4, sort_keys=True)
        with open('./workspace/res.json', "w") as f:
            f.write(resj)

        self.res = res
        return

    def get_response(self):

        res = self.res
        all_topics = self.all_topics
        metas = self.metas
        return res, all_topics, metas

    def split_corpus(self, c, rate_or_size):

        size = 0
        if isinstance(rate_or_size, float):
            size = math.floor(len(c) * rate_or_size)
        else:
            size = rate_or_size

        left = c[:-size]
        right = c[-size:]

        return left, right
