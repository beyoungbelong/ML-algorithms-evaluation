import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from numpy import nan
import re
from tqdm import tqdm
import json


matplotlib.use('TKAgg')

topic_num = 7

result1 = {}  # 记录每个主题每个月份下有多少篇
totalCount1 = {}  # 记录每个月份下有多少篇
monthList = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "#"]
yearList = ["2020", "2021", '2022', "2023", "#"]
for i in range(topic_num):  # self.ntopics 初始化result，totalcount字典，用来计数
    result1[i] = {}
    for year in yearList:
        result1[i][year] = {}
        for month in monthList:
            result1[i][year][month] = 0
df = pd.read_csv("lda/nobelow=30_topic=7_iteration=2000/output/文本话题归类.csv")
for index, row in df.iterrows():
    result1[row['topicID']][str(row['year'])][str(row['month']).zfill(2)] = result1[row['topicID']][str(row['year'])][str(row['month']).zfill(2)] + 1


def percentageof(x, y,ypercen):
    # 各个主题在当前月份发表数量占得比例。例如主题1在1月份发表100篇，1月份总共有1000篇，则主题一在一月份发表占比10%
    # 一个主题一条线，总共多少个主题就是多少条线
    colorList = ["#FF6666", "#666633", "#006699", "#FF9966", "#663300", "#339933", "#FFCC33", "#333399", "#33FF99",
          "#FFCCCC", "#663366", "#990099", "#FF99CC"]

    plt.figure(figsize=(6, 6))
    # plt.rcParams['font.family'] = ['SimHei']
    plt.subplot(2, 1, 1)
    for topicID in range(topic_num):
        plt.plot(x, y[topicID], label="Topic" + str(int(topicID) + 1), color=colorList[topicID],linewidth=1)
    plt.ylabel('Number of literature', fontsize=5)
    plt.xlabel('(a)', fontsize=8)
    plt.yticks(fontsize=5)
    plt.xticks(fontsize=5, rotation=30)
    plt.legend(fontsize=5, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.subplots_adjust(right=0.8) 
    # plt.savefig(self.output + '各主题论文数量百分比.png')

    plt.subplot(2, 1, 2)
    for topicID in range(topic_num):  # self.ntopics
        plt.plot(x, ypercen[topicID], label="Topic" + str(int(topicID) + 1), color=colorList[topicID],linewidth=1)
    plt.ylabel('Percentage of literature number', fontsize=5)
    plt.xlabel('(b)', fontsize=8)
    plt.yticks(fontsize=5)
    plt.xticks(fontsize=5, rotation=30)
    plt.legend(fontsize=5, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.subplots_adjust(right=0.8) 

    plt.rcParams['savefig.dpi'] = 1000  # 图片像素
    plt.style.use('seaborn-white') # 将背景颜色改为。。
    plt.tight_layout()
    plt.savefig('Statistic_graph/各主题论文数量及趋势合并中文.png')



def percentageof2(x, y,ypercen):
    # 各个主题在当前月份发表数量占得比例。例如主题1在1月份发表100篇，1月份总共有1000篇，则主题一在一月份发表占比10%
    # 一个主题一条线，总共多少个主题就是多少条线
    colorList = ["#FF6666", "#666633", "#006699", "#FF9966", "#663300", "#339933", "#FFCC33", "#333399", "#33FF99",
          "#FFCCCC", "#663366", "#990099", "#FF99CC"]

    plt.figure(figsize=(9, 6))
    # plt.rcParams['font.family'] = ['SimHei']
    plt.subplot(2, 2, 1)
    for topicID in range(topic_num-6):
        plt.plot(x, y[topicID], label="Topic" + str(int(topicID) + 1), color=colorList[topicID],linewidth=1)
    plt.ylabel('Number of literature', fontsize=5)
    plt.xlabel('(a)', fontsize=8)
    plt.yticks(fontsize=5)
    plt.xticks(fontsize=5, rotation=30)
    plt.legend(fontsize=5, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.subplots_adjust(right=0.8) 
    # plt.savefig(self.output + '各主题论文数量百分比.png')

    plt.subplot(2, 2, 2)
    for topicID in range(topic_num-6, topic_num):
        plt.plot(x, y[topicID], label="Topic" + str(int(topicID) + 1), color=colorList[topicID],linewidth=1)
    plt.ylabel('Number of literature', fontsize=5)
    plt.xlabel('(b)', fontsize=8)
    plt.yticks(fontsize=5)
    plt.xticks(fontsize=5, rotation=30)
    plt.legend(fontsize=5, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.subplots_adjust(right=0.8) 
    # plt.savefig(self.output + '各主题论文数量百分比.png')

    plt.subplot(2, 2, 3)
    for topicID in range(topic_num-6):  # self.ntopics
        plt.plot(x, ypercen[topicID], label="Topic" + str(int(topicID) + 1), color=colorList[topicID],linewidth=1)
    plt.ylabel('Percentage of literature number', fontsize=5)
    plt.xlabel('(c)', fontsize=8)
    plt.yticks(fontsize=5)
    plt.xticks(fontsize=5, rotation=30)
    plt.legend(fontsize=5, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.subplots_adjust(right=0.8) 

    plt.subplot(2, 2, 4)
    for topicID in range(topic_num-6, topic_num):
        plt.plot(x, ypercen[topicID], label="Topic" + str(int(topicID) + 1), color=colorList[topicID],linewidth=1)
    plt.ylabel('Percentage of literature number', fontsize=5)
    plt.xlabel('(d)', fontsize=8)
    plt.yticks(fontsize=5)
    plt.xticks(fontsize=5, rotation=30)
    plt.legend(fontsize=5, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.subplots_adjust(right=0.8) 

    plt.rcParams['savefig.dpi'] = 2000  # 图片像素
    plt.style.use('seaborn-white') # 将背景颜色改为。。
    plt.tight_layout()
    plt.savefig('Statistic_graph/各主题论文数量及趋势合并中文.png')



# x轴数据
x1 = ["20-" + k for k in ["01", "03", "05", "07", "09", "11"]]
x2 = ["21-" + k for k in ["01", "03", "05", "07", "09", "11"]]
x3 = ["22-" + k for k in ["01", "03", "05", "07", "09", "11"]]
x4 = ["23-" + k for k in ["01", "03", "05", "07", "09"]]
x = x1 + x2 + x3 + x4

y = []  # y轴数据（频次）
for item in result1.items():  # y轴是频次
    y1 = [value for key, value in item[1]["2020"].items() if key != "#"]
    y2 = [value for key, value in item[1]["2021"].items() if key != "#"]
    y3 = [value for key, value in item[1]["2022"].items() if key != "#"]
    y4 = [value for key, value in item[1]["2023"].items() if key != "#"]
    temp = []
    yy = y1 + y2 + y3 + y4
    for i in range(0, 46, 2):
        if i == 0:
            temp.append(yy[i])
        else:
            temp.append(yy[i] + yy[i-1])
    y.append(temp)
# self.__numberof(x, y)
total=[]
for i in range(0,23):
    tem=0
    for j in range(topic_num):
        tem=tem+y[j][i]
    total.append(tem)
ypercen=[]
for i in range(topic_num):
    temp=[]
    for j in range(0,23):
        temp.append((y[i][j]/total[j]*100) if total[j] != 0 else 0)
    ypercen.append(temp)
percentageof(x=x,y=y,ypercen=ypercen)


