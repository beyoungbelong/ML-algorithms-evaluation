import csv
import matplotlib.pyplot as plt
import json
import re
import gensim
from gensim.corpora import Dictionary
from numpy import character, nan
import pandas as pd
import warnings

warnings.filterwarnings("ignore")  # 忽略某些不影响程序的提示
import pyLDAvis
import pyLDAvis.gensim
import os
from pathlib import Path


class Topic(object):
    def __init__(self, cwd):
        """
        初始化
        :param cwd: lda模型相关文件的存储路径，为lda模型/nobelow=50_topic=13_iteration=2000
        """
        if cwd == False:
            return
        self.output = Path(cwd).joinpath('output')
        self.output.mkdir(exist_ok=True)
        # 模型文件的存储路径，包括model、向量文件等等
        bestmodelspath = Path(cwd).joinpath('output', 'model')
        bestmodelspath.mkdir(exist_ok=True)

    def create_dictionary(self, documents, no_below=30, no_above=0.5):
        """
        输入带documents构建词典dictionary (同时会默认将词典命名为dictionary.dict存储到output文件内)
        :param documents: 列表； 注意documents中的每个document是词语列表，即文本分词后的结果
        :param no_below: 整数；构建词典空间时，词语出现次数低于no_below的词语剔除掉
        :param no_above: 小数范围(0, 1)； 构建词典空间时，词语词频比高于no_above的词语剔除掉。
        :return: 返回corpus
        """
        self.documents = self.__add_bigram(documents=documents, min_count=30)  # 将documents更新，将词组放在一起
        # self.documents = documents
        self.dictionary = Dictionary(self.documents)
        self.dictionary.filter_extremes(no_below=no_below, no_above=no_above)
        fpath = str(self.output.joinpath('dictionary.dict'))
        self.dictionary.save(fpath)

    def load_dictionary(self, dictpath='lda模型/nobelow=50_topic=13_iteration=2000/output/dictionary.dict'):
        """
        导入词典(同时在此步骤可自动构建corpus)
        :param dictpath: 词典文件的路径
        :return: 返回corpus
        """

        self.dictionary = Dictionary.load(dictpath)
        return self.dictionary

    def create_corpus(self, documents):
        """
        输入documents构建corpus;
        :param documents: 列表； 注意documents中的每个document是词语列表
        :return: 返回corpus
        """
        self.corpus = [self.dictionary.doc2bow(document) for document in documents]
        self.documents = documents

    def __add_bigram(self, documents, min_count):
        """
        分词容易把一个意思的词组分成多个词，考虑词组整体作为一个整体。
        :param documents: 文档集(语料库)
        :param min_count: 词组出现次数少于于min_count，该词组不会被加到documents中
        :return: 更新后的documents
        """
        bigram = gensim.models.Phrases(documents, min_count=min_count)
        for idx in range(len(documents)):
            for token in bigram[documents[idx]]:
                if '_' in token:
                    # Token is a bigram, add to document.
                    documents[idx].append(token)
        return documents

    def train_lda_model(self, n_topics, fname='lda.model', epochs=20, iterations=300):
        """
        训练lda话题模型，运行时间较长，请耐心等待~
        :param n_topics:  指定的话题数
        :param fname:  模型的文件名（用来保存训练好的lda模型）,默认存放在output文件夹内
        :param epochs: 使用数据集训练的轮数。epochs越大，训练越慢
        :param iterations:  对每个document迭代(学习)的次数；iterations越大，训练越慢
        :return:  返回训练好的lda模型
        """



        self.model = gensim.models.LdaMulticore(corpus=self.corpus,
                                                num_topics=n_topics,
                                                id2word=self.dictionary.id2token,
                                                workers=4,
                                                chunksize=1000,
                                                eval_every=None,
                                                passes=epochs,
                                                iterations=iterations,
                                                batch=True,
                                                random_state=5)  # 5的结果还行
        fpath = str(self.output.joinpath('model', fname))
        self.model.save(fpath)
        return self.model

    def load_lda_model(self, modelpath='lda模型/nobelow=30_topic=13_iteration=2000/output/model/lda.model'):
        """
        导入之前训练好的lda模型 modelpath='output/model/lda.model'
        :param modelpath: lda模型的文件路径 (存放在output中的best_model和models文件夹中内)
        :return:
        """
        self.model = gensim.models.LdaModel.load(modelpath, mmap='r')
        return self.model

    def show_topics(self, formatted=True):
        """
        显示话题与对应的特征词之间的权重关系
        :param formatted: 特征词与权重是否以字符串形式显示
        :return: 列表
        """
        return self.model.show_topics(formatted=formatted)

    def visualize_lda(self, fname='vis.html'):
        """
            可视化LDA模型。如果notebook中无法看到可视化效果，请在output文件夹内找到 fname所对应的html文件，用浏览器打开观看
            :param fname: 可视化html文件，默认存放在output文件夹内，运行结束后找到vis.html并用浏览器打开
            :return:
        """
        vis = pyLDAvis.gensim.prepare(self.model, self.corpus, self.dictionary, sort_topics=False)
        fpath = str(self.output.joinpath(fname))
        pyLDAvis.save_html(vis, fpath)

    def get_document_topics(self, document):
        """
        :param document: 词语列表
        :return:
        """
        return self.model.get_document_topics(self.dictionary.doc2bow(document))

    def topic_distribution(self, raw_documents, title, year, month, quarter, PMID, fname='文本话题归类.csv'):
        """
        将raw_documents与对应的所属话题一一对应，结果以fname的csv文件存储。存储记录格式为，主题、标题、年、月、季度、全文
        :param quarter:
        :param month:
        :param year:
        :param title:
        :param raw_documents: 列表；原始的文档数据集
        :param fname: csv文件名, 默认存放在output文件夹内
        :return:
        """
        topics = []

        for document in self.corpus:
            # 判断文章的主题
            topic_probs = self.model.get_document_topics(document)
            topic = sorted(topic_probs, key=lambda k: k[1], reverse=True)[0][0]
            topics.append(topic)

        df = pd.DataFrame({'topicID': topics, 'title': title, 'year': year, 'month': month, 'quarter': quarter,
                           'fulltext': raw_documents, 'PMID': PMID})
        fpath = str(self.output.joinpath(fname))
        df.to_csv(fpath, mode='w', encoding='utf-8', index=False)
        print('------每篇文献的话题归类完成------')
        return df['topicID'].value_counts()


    def get_corpus(self):
        print("get")
        return self.corpus

    def get_dictionary(self):
        return self.dictionary

    def statistic_algoriths(self):
        # 横坐标-时间，间隔两个月；纵坐标各个主题的文章数量，每个主题数量在同一时段数量的占比
        result1 = {}  # 记录每个主题每个月份下有多少篇
        totalCount1 = {}  # 记录每个月份下有多少篇
        monthList = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "#"]
        yearList = ["2020", "2021", '2022','2023', "#"]
        for i in range(10):  # self.ntopics 初始化result，totalcount字典，用来计数
            result1[i] = {}
            for year in yearList:
                result1[i][year] = {}
                for month in monthList:
                    result1[i][year][month] = 0
        df = pd.read_csv("统计绘图\算法匹配结果.csv")
        for index, row in df.iterrows():
            result1[row['topicID']][str(row['year'])][str(row['month']).zfill(2)]=result1[row['topicID']][str(row['year'])][str(row['month']).zfill(2)]+1

        # x轴数据
        x1 = ["20-" + k for k in ["02", "04",  "06",  "08",  "10",  "12"]]
        x2 = ["21-" + k for k in ["02", "04",  "06",  "08",  "10",  "12"]]
        x3 = ["22-" + k for k in ["02", "04", "05"]]
        x = x1 + x2 + x3

        y = []  # y轴数据（频次）
        for item in result1.items():  # y轴是频次
            y1 = [value for key, value in item[1]["2020"].items() if key != "#"]
            y2 = [value for key, value in item[1]["2021"].items() if key != "#"]
            y3 = [value for key, value in item[1]["2022"].items() if key != "#"]
            temp=[]
            yy=y1+y2+y3
            for i in range(0,29,2):
                if i==28:
                    temp.append(yy[i])
                else:
                    temp.append(yy[i]+yy[i+1])
        self.__numberof(x, y)


    def __numberof(self,x,y):
        plt.figure(figsize=(20, 10))
        for topicID in range(8):  # self.ntopics
            plt.plot(x, y[topicID], label="Topic "+str(topicID+1))
        plt.ylabel('Number of Papers', fontsize=20)
        plt.yticks(fontsize=18)
        plt.xticks(fontsize=15, rotation=30)
        plt.legend(fontsize=15)
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['savefig.dpi'] = 500  # 图片像素
        plt.style.use('bmh')  # 将背景颜色改为。。
        plt.savefig(self.output.joinpath('各主题论文数量.png'))
        plt.show()


    def get_algoriths(self):
        # 获取每篇文献中各个算法的出现情况
        print('------开始匹配每篇文章的算法出现情况------')
        articles = []
        path = self.output.joinpath('文本话题归类.csv')
        keyWords = []  # 读取算法列表list
        dfal = pd.read_csv('Machine Learning Algorithm List.csv')
        keyWords.append([line[0].lower() for line in dfal.values])
        keyWords.append([line[1] for line in dfal.values if line[1] is not nan])
        df = pd.read_csv(path)  # 读取文本话题归类文件，然后遍历寻找每一个算法的出现情况
        for index, row in df.iterrows():
            temp = {'topicID': row['topicID'], 'title': row['title'], 'year': row['year'], 'month': row['month'],
                    'quarter': row['quarter']}
            for key in keyWords[0]:
                if self.verdict(key, row['fulltext']):
                    temp[key] = 1
                else:
                    temp[key] = 0
            articles.append(temp)

        pd.DataFrame(list(articles)).to_csv("统计绘图\算法匹配结果.csv",
                                            index=False)  # 保存到CSV看看结果如何
        with open("统计绘图\算法匹配结果.json", "w",
                  encoding="utf-8") as f:  # 保存到json看看结果如何
            json.dump(list(articles), f, ensure_ascii=False)
        print('------每篇文章的算法出现情况匹配完成------')

    def verdict(self, keyword, content):
        # 这个函数只针对全称做了处理，没对算法缩写做处理
        keyword = keyword.lower().replace('-', ' ')
        content = content.lower().replace('-', ' ')
        # if keyword in content:
        #     return True
        # return False

        # 检查内容里面是否按顺序出现了所有单词（这些单词中间可以隔0个或多个其他单词），如果是，就算命中
        keyword_compile = r'\b' + r'\b.*\b'.join(re.escape(word) for word in keyword.split(' ')) + r'\b'
        return bool(re.search(keyword_compile, content))
