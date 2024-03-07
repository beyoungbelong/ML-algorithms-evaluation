import pandas as pd
from pathlib import Path
import json
from numpy import character, nan
import re
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def verdict(keyword, content):
    # 这个函数只针对全称做了处理，没对算法缩写做处理
    keyword = keyword.lower().strip().replace('-', ' ')
    content = content.lower().strip().replace('-', ' ')
    # if keyword in content:
    #     return True
    # return False

    # 检查内容里面是否按顺序出现了所有单词（这些单词中间可以隔0个或多个其他单词），如果是，就算命中
    keyword_compile = r'\b' + r'\b.*\b'.join(re.escape(word) for word in keyword.split(' ')) + r'\b'
    return bool(re.search(keyword_compile, content))



# 获取每篇文献中各个算法的出现情况，以及主题
print('------开始匹配每篇文章的算法出现情况------')
keyWords = []  # 读取算法列表list
dfal = pd.read_csv('Machine Learning Algorithm List.csv').fillna('')



dfull = pd.read_csv("temp/topic_fulltext.csv",low_memory=False, header=0).fillna('')
articles = []
for index, row in dfull.iterrows():
    temp = {'topicID': row['topicID'], 'title': row['title'], 'year': row['year'], 'month': row['month'],
            'quarter': row['quarter'], 'PMID': row['PMID']}
    for no,keyline in dfal.iterrows():
        if len(keyline['Shorthand']) == 0:
            if verdict(keyline['Algorithms'], row['fulltext']):
                temp[keyline['Algorithms']]=1
            else:
                temp[keyline['Algorithms']]=0
        else:
            if verdict(keyline['Algorithms'], row['fulltext']) or verdict(keyline['Shorthand'],row['fulltext']):
                temp[keyline['Algorithms']]=1
            else:
                temp[keyline['Algorithms']]=0
    articles.append(temp)
    if len(articles) % 1000 == 0:
        print(f"正在获取第{len(articles)}篇的全文算法匹配结果")

pd.DataFrame(list(articles)).to_csv("temp/全文算法匹配结果.csv",index=False)  # 保存到CSV看看结果如何
print('------每篇文章的算法出现情况匹配完成------')


def plotNetwork(matrix,topic):
		plt.title('Topic' + str(topic+1),fontsize=25)
		G=nx.Graph()
		nodenum=[16,16,16,60,30,24,6,24,30,30,16,30,8]
		for i in range(len(matrix)):
			for j in range(i+1,len(matrix)):
				if matrix.iloc[i,j] != 0 and matrix.iloc[i,i] > nodenum[topic] and matrix.iloc[j,j] > nodenum[topic] and matrix.columns.values[i]!=matrix.columns.values[j]:
					G.add_nodes_from([matrix.columns.values[i],matrix.columns.values[j]])
					G.add_edge(matrix.columns.values[i],matrix.columns.values[j],weight=matrix.iloc[i,j])
		node_sizes = [d * 80 for _, d in G.degree()]
		w = [G[e[0]][e[1]]['weight']/30 for e in G.edges()]
		pos=nx.spring_layout(G)
		nx.draw(G,pos=pos,width=w,with_labels=True,node_color='#1C86EE',edge_color='#00CC00',node_size=node_sizes,font_size=20,alpha=0.7)
		plt.axis('on')
		plt.xticks([])
		plt.yticks([])

		plt.savefig('Landmark_analysis/主题'+str(topic)+'共现网络.png',dpi=500)


def Evaluation(matrix,topic):
		G=nx.Graph()
		nodenum=[16,16,16,60,30,24,6,24,30,30,16,30,8]
		for i in range(len(matrix)):
			for j in range(i+1,len(matrix)):
				if matrix.iloc[i,j] != 0 and matrix.iloc[i,i] > nodenum[topic] and matrix.iloc[j,j] > nodenum[topic] and matrix.columns.values[i]!=matrix.columns.values[j]:
					G.add_nodes_from([matrix.columns.values[i],matrix.columns.values[j]])
					G.add_edge(matrix.columns.values[i],matrix.columns.values[j],weight=matrix.iloc[i,j])


		# print('主题'+str(topic+1)+'提及论文数'+str(sorted([(matrix.columns.values[i], matrix.iloc[i, i]) for i in range(len(matrix))],key=lambda pair: pair[1], reverse=True)))
		# print('主题'+str(topic+1)+'节点的度'+str([(node,val) for (node,val) in sorted(G.degree(),key=lambda pair:pair[1],reverse=True)]))  # 节点的度
		# print('主题'+str(topic+1)+'节点的加权度'+str([(node, val) for (node, val) in sorted(G.degree(weight='weight'), key=lambda pair: pair[1], reverse=True)]))  # 节点的加权度
		# # 节点度中心性是指节点在其与之直接相连的邻居节点当中的中心程度
		# print('主题' + str(topic+1) + '节点的度中心性' + str([(node, val) for (node, val) in sorted(nx.degree_centrality(G).items(), key=lambda pair: pair[1],reverse=True)]))
		# # 介数中心性
		# print('主题' + str(topic+1) + '节点的介数中心性' + str([(node, val) for (node, val) in sorted(nx.betweenness_centrality(G).items(), key=lambda pair: pair[1],reverse=True)]))
		# # 接近度中心性
		# print('主题' + str(topic+1) + '节点的接近度中心性' + str([(node, val) for (node, val) in sorted(nx.closeness_centrality(G).items(), key=lambda pair: pair[1],reverse=True)]))
		# # 特征向量中心性
		# print('主题' + str(topic+1) + '节点的特征向量中心性' + str([(node, val) for (node, val) in sorted(nx.eigenvector_centrality(G).items(), key=lambda pair: pair[1],reverse=True)]))
		# 节点的加权度
		degree_weight={node:val for node,val in G.degree(weight='weight')}
		degree_wei_min=min(degree_weight.values())
		degree_wei_dis=max(degree_weight.values())-degree_wei_min
		# 节点度中心性是指节点在其与之直接相连的邻居节点当中的中心程度
		degree_cen={node:val for node,val in nx.degree_centrality(G).items()}
		degree_cen_min=min(degree_cen.values())
		degree_cen_dis=max(degree_cen.values())-degree_cen_min
		# 介数中心性
		degree_betw = {node: val for node, val in nx.betweenness_centrality(G).items()}
		degree_betw_min=min(degree_betw.values())
		degree_betw_dis=max(degree_betw.values())-degree_betw_min
		# 接近度中心性
		degree_clos = {node: val for node, val in nx.closeness_centrality(G).items()}
		degree_clos_min=min(degree_clos.values())
		degree_clos_dis=max(degree_clos.values())-degree_clos_min
		# 特征向量中心性
		degree_vec = {node: val for node, val in nx.eigenvector_centrality(G).items()}
		degree_vec_min=min(degree_vec.values())
		degree_vec_dis=max(degree_vec.values())-degree_vec_min

		matrix_min=min([matrix.iloc[i,i] for i in range(len(matrix))])
		matrix_dis=max([matrix.iloc[i,i] for i in range(len(matrix))])-min([matrix.iloc[i,i] for i in range(len(matrix))])


		average = [(al, sum([(matrix.loc[al, al] - matrix_min) / matrix_dis if al in degree_weight else 0,
                    (degree_weight[al] - degree_wei_min) / degree_wei_dis if al in degree_weight else 0,
                    (degree_cen[al] - degree_cen_min) / degree_cen_dis if al in degree_cen else 0,
                    (degree_betw[al] - degree_betw_min) / degree_betw_dis if al in degree_betw else 0,
                    (degree_clos[al] - degree_clos_min) / degree_clos_dis if al in degree_clos else 0,
                    (degree_vec[al] - degree_vec_min) / degree_vec_dis if al in degree_vec else 0]) / 6)
           			for al in matrix.columns.values]

		# print('主题' + str(topic+1) + '归一平均' + str([(node, val) for (node, val) in sorted(average, key=lambda pair: pair[1],reverse=True)]))

		output_file = open('Landmark_analysis/主题'+str(topic)+' Evaluation results.txt', 'w')
		output_file.write('主题'+str(topic+1)+'提及论文数'+str(sorted([(matrix.columns.values[i], matrix.iloc[i, i]) for i in range(len(matrix))],key=lambda pair: pair[1], reverse=True)))
		output_file.write('\n主题'+str(topic+1)+'节点的度'+str([(node,val) for (node,val) in sorted(G.degree(),key=lambda pair:pair[1],reverse=True)]))
		output_file.write('\n主题'+str(topic+1)+'节点的加权度'+str([(node, val) for (node, val) in sorted(G.degree(weight='weight'), key=lambda pair: pair[1], reverse=True)]))
		output_file.write('\n主题' + str(topic+1) + '节点的度中心性' + str([(node, val) for (node, val) in sorted(nx.degree_centrality(G).items(), key=lambda pair: pair[1],reverse=True)]))
		output_file.write('\n主题' + str(topic+1) + '节点的介数中心性' + str([(node, val) for (node, val) in sorted(nx.betweenness_centrality(G).items(), key=lambda pair: pair[1],reverse=True)]))
		output_file.write('\n主题' + str(topic+1) + '节点的接近度中心性' + str([(node, val) for (node, val) in sorted(nx.closeness_centrality(G).items(), key=lambda pair: pair[1],reverse=True)]))
		output_file.write('\n主题' + str(topic+1) + '节点的特征向量中心性' + str([(node, val) for (node, val) in sorted(nx.eigenvector_centrality(G).items(), key=lambda pair: pair[1],reverse=True)]))
		output_file.write('\n主题' + str(topic+1) + '归一平均' + str([(node, val) for (node, val) in sorted(average, key=lambda pair: pair[1],reverse=True)]))
		output_file.close()

		return average



keyWords = []  # 读取算法列表list
dfal = pd.read_csv('Machine Learning Algorithm List.csv').fillna('')
keyWords.append([line[0] for line in dfal.values])
keyWords.append([line[1] for line in dfal.values if line[1] is not nan])


dfull = pd.read_csv('temp/全文算法匹配结果.csv',low_memory=False, header=0).fillna('')
num=0
plt.figure(figsize=(16, 20))
# plt.rcParams['font.family'] = ['Times New Roman']
normalize_average_list = []
for topic in range(13):
	plt.subplot(5, 3, topic+1)
	print(f"正在统计主题{topic+1}的算法共现")
	matrix = pd.DataFrame(np.full((len(keyWords[0]), len(keyWords[0])), 0), columns=keyWords[0], index=keyWords[0])
	for index,row in dfull.iterrows():
		if row['topicID']==topic:
			for rowal in matrix.index.values:
				for colal in matrix.columns.values:
					if row[rowal] and row[colal] and rowal is not colal:
						matrix.loc[rowal,colal]=matrix.loc[rowal,colal]+1
						matrix.loc[colal,rowal]=matrix.loc[colal,rowal]+1
					if row[rowal] and row[colal] and rowal is colal:
						matrix.loc[rowal,colal]=matrix.loc[rowal,colal]+1
	matrix.to_csv('Landmark_analysis/主题'+str(topic)+'共现矩阵.csv',index=False)  # 保存到CSV看看结果如何
	normalize_average = Evaluation(matrix,topic)
	normalize_average_list.append(normalize_average)
	matrix.columns = keyWords[1]
	matrix.index = keyWords[1]
	plotNetwork(matrix,topic)
plt.tight_layout()
plt.rcParams['savefig.dpi'] = 400
plt.savefig('Landmark_analysis/共现网络.png',dpi=500)


category_algorithm = {}
dfal = pd.read_csv('Machine Learning Algorithm List.csv')
for line in dfal.values:
    category_algorithm[line[2]] = []
for line in dfal.values:
    category_algorithm[line[2]].append(line[0])
    if line[3] is not nan: category_algorithm[line[3]].append(line[0])

categories = list(category_algorithm.keys())

influences = []
for topic in range(13):
    # 为每个类别维护一个存储得分的字典
    category_scores = {category: [] for category in category_algorithm}
    # 遍历该主题下算法的得分情况
    for algorithm, score in normalize_average_list[topic]:
        # 属于某一类别就加入分数
        for category, algorithms in category_algorithm.items():
            if algorithm in algorithms and score > 0:
                category_scores[category].append(score)
    # 再维护一个计算平均分的字典
    category_average_scores = {}
    category_average_scores['topic'] = topic+1
    for category, scores in category_scores.items():
        if scores:
            average_score = sum(scores) / len(scores)
            category_average_scores[category] = average_score
        else:
            category_average_scores[category] = 0
    influences.append(category_average_scores)    

df_influences = pd.DataFrame.from_dict(influences)
df_influences.to_csv('Influence of different types of ML algorithms.csv', index=False)
df_influences