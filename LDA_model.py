import sys
from PIL import Image
import importlib
from gensim import corpora, models
import gensim
from wordcloud import WordCloud
importlib.reload(sys)
from chinesetopic import Topic
import json
import os
import time
from pathlib import Path
import warnings
import math
import matplotlib.pyplot as plt
import matplotlib
from pylab import xticks, yticks, np


class topicModel:
	"""
	这是运行的第4个文件,用于将分词后的结果用于主题建模

	"""

	path = "temp/分词后_abs.json"
	fullpapers = []
	title=[]
	year = []
	month = []
	quarter = []
	PMID = []

	def __init__(self):
		return

	def get_tokens(self):
		tokens = []
		with open(self.path, mode="r", encoding="utf-8") as f:
			articals = json.load(f)
		for artical in articals:
			tokens.append(artical['tokenize'])
			self.title.append(artical['Title'])
			self.fullpapers.append(artical['abstract'])
			self.year.append(artical['Year'])
			self.month.append(artical['Month'])
			self.quarter.append(artical['Quarter'])
			self.PMID.append(artical['PMID'])
		return tokens

	def topic_train(self, canshu):

		print("获取文档集中。。。。。。")
		tokens = self.get_tokens()
		print("获取文档集结束，开始训练--------------")
		save_path = canshu['output']
		topic = Topic(cwd=save_path)  
		topic.create_dictionary(documents=tokens, no_below=canshu['no_below'])
		topic.create_corpus(documents=tokens)  # 构建语料(将文本转为文档-词频矩阵)

		model=topic.train_lda_model(
			n_topics=canshu['ntopics'],
			fname='lda.model',
			epochs=canshu['epochs'],
			iterations=canshu['iterations'])  # 指定n_topics，构建LDA话题模型
		print("模型训练主题数量---",canshu['ntopics'])
		# 每个主题的词云图单独绘制
		for t in range(canshu['ntopics']):
			print("------正在绘制第",t,"个主题的词云图")
			d=dict()
			for k,v in model.show_topic(t, 100):
				d[k]=v
			print('主题',t,'高频词：',d.keys())
			plt.figure()
			plt.axis("off")
			plt.rcParams['savefig.dpi'] = 500  # 图片像素
			plt.imshow(WordCloud(background_color='white', max_font_size=90, min_font_size=10,width=600,height= 400).fit_words(d))
			plt.savefig(f'Statistic_graph/主题{int(t) + 1}词云图.png', bbox_inches='tight')
			# plt.show()
		topic.topic_distribution(raw_documents=self.fullpapers, title=self.title, year=self.year, month=self.month, quarter=self.quarter, PMID=self.PMID)
		topic.visualize_lda()
		self.modifyvis(canshu['ntopics'], save_path)
		return

	def plotwordcoloud(self, canshu):
		save_path = canshu['output']
		topic = Topic(cwd=save_path)
		model=topic.load_lda_model()
		# 绘制每个主题的词云图在一个画布中2*4
		d = dict()
		for i in range(8):
			d[i] = dict()
			for k, v in model.show_topic(i, 100):
				d[i][k] = v
		plt.figure(figsize=(20,9))
		plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
		for i in range(8):
			plt.subplot(2, 4, i + 1)
			plt.title("主题 " + str(i + 1),fontsize=25)
			plt.axis('off')
			plt.imshow(WordCloud(background_color='white',scale=4,width=512,height=384).fit_words(d[i]))
		plt.tight_layout()
		plt.rcParams['savefig.dpi'] = 500
		plt.savefig('Statistic_graph/主题词云图中文.png')


	def modifyvis(self, ntopics, save_path):
		# 修改可视化文件，将远程css、js资源加载链接改为本地
		path = save_path.joinpath('output')
		with open(path.joinpath('vis.html'), mode='r', encoding='utf-8') as f:
			text = f.read()
		text = text.replace("https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.css",
		                    "../../LDAvis_files/LDAvis.css")
		text = text.replace("https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min", "../../LDAvis_files/d3")
		text = text.replace("https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js",
		                    "../../LDAvis_files/LDAvis.js")
		text = text.replace("https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js", "../../LDAvis_files/d3.js")
		with open(path.joinpath('vis2.html'), mode='w', encoding='utf-8') as f:
			f.write(text)

	def getntopic(self):
		print("获取文档集中。。。。。。")
		tokens = self.get_tokens()
		print("获取文档集结束，开始训练--------------")
		topic = Topic(cwd='LDA')
		topic.create_dictionary(documents=tokens, no_below=30)
		topic.create_corpus(documents=tokens)
		perplexity_values = []
		coherence_values = []
		x = []
		for topicnum in range(1, 20,1):
			print('主题数量：', topicnum)
			lda_model = topic.train_lda_model(
				n_topics=topicnum,
				fname='lda.model',
				epochs=30,
				iterations=2000)  # 指定n_topics，构建LDA话题模型
			x.append(topicnum)
			perplexity_values.append(lda_model.log_perplexity(topic.corpus))
			coherencemodel = models.CoherenceModel(model=lda_model, texts=tokens, dictionary=topic.dictionary,
			                                       coherence='c_v')
			print(coherencemodel.get_coherence())
			coherence_values.append(coherencemodel.get_coherence())
		print('主题评价完成,将绘制主题一致性和困惑度图像')
		fig = plt.figure(figsize=(15, 5))
		plt.rcParams['font.sans-serif'] = ['SimHei']
		matplotlib.rcParams['axes.unicode_minus'] = False

		ax1 = fig.add_subplot(1, 2, 1)
		plt.plot(x, perplexity_values, marker="o")
		plt.title("主题建模-困惑度")
		plt.xlabel('主题数目')
		plt.ylabel('困惑度大小')
		xticks(np.linspace(1, 20, 20, endpoint=True))  # 保证x轴刻度为1

		ax2 = fig.add_subplot(1, 2, 2)
		plt.plot(x, coherence_values, marker="o")
		print(
			coherence_values)  #[0.16412819274376417, 0.5936617762290546, 0.6241363057611887, 0.633754238107984, 0.6398328035134343, 0.6939826059591864, 0.6578071796152357]
		plt.title("主题建模-一致性")
		plt.xlabel("主题数目")
		plt.ylabel("一致性大小")
		xticks(np.linspace(1, 20, 20, endpoint=True))

		plt.savefig('topic_plexity_coherence.png')
		plt.show()

canshu = {}
nobelow = 30
for ntopic in range(7,8):
    for iteration in [2000]:
        canshu['no_below'] = nobelow
        canshu['ntopics'] = ntopic
        canshu['epochs'] = 30
        canshu['iterations'] = iteration
        canshu['chunksize'] = 1000
        canshu['isAddBigram'] = True
        canshu['output'] = Path(os.getcwd()).joinpath(
            f'lda/nobelow={str(nobelow)}_topic={str(ntopic)}_iteration={str(iteration)}')

        print(".........nobelow=%d    主题数=%d  iteration=%d  的模型开始训练.........." % (nobelow, ntopic, iteration))
        canshu['output'].mkdir(parents=True, exist_ok=True)
        f_log = open(canshu['output'].joinpath('log.txt'), 'w',encoding='utf-8')
        f_log.write("开始时间：" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")
        f_log.write("no_below：" + str(canshu['no_below']) + "\n")
        f_log.write("ntopics：" + str(canshu['ntopics']) + "\n")
        f_log.write("epochs：" + str(canshu['epochs']) + "\n")
        f_log.write("iterations：" + str(canshu['iterations']) + "\n")
        f_log.write("chunksize：" + str(canshu['chunksize']) + "\n")
        f_log.write("isAddBigram：" + str(canshu['isAddBigram']) + "\n")

        analyze = topicModel()
        analyze.topic_train(canshu)  # 训练模型
        # analyze.plotwordcoloud(canshu)  # 绘制主题词云图
        # analyze.getntopic()

        f_log.write("结束时间：" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")
        f_log.close()
        print(".........nobelow=%d    主题数=%d  iteration=%d  的模型结束训练.........." % (nobelow, ntopic, iteration))


topic = 7
model = gensim.models.LdaModel.load(f'lda/nobelow=30_topic={topic}_iteration=2000/output/model/lda.model', mmap='r')
topic_list = []
for t in range(topic):
    d=dict()
    for k,v in model.show_topic(t, 100):
        d[k]=float(v)
    topic_list.append(d)
print(len(topic_list))
with open(f'topic{topic}_top100word.json', mode="w", encoding="utf-8") as f:
    json.dump(topic_list, f, ensure_ascii=False)

