from bioc import BioCAnnotation
from bioc import BioCXMLReader
from bioc import BioCXMLWriter
import os
from tqdm import tqdm
import pandas as pd
import json

def all_exist(avalue, bvalue):
	return any(x.lower() in bvalue.lower() for x in avalue)


text_type_set = ['abstract', 'abstract_title_1', 'front', 'paragraph', 'title', 'title_1', 'title_2', 'title_3', 'title_4', 'title_5']
paper_list = []
list_dir = os.listdir('sample data')
list_dir.remove('.DS_Store')
for dir in list_dir:
    file_list = os.listdir('sample data/'+dir)
    for file in tqdm(file_list):
        if file.endswith('.xml'): 
            bioc_reader = BioCXMLReader(f'sample data/{dir}/{file}', dtd_valid_file='BioC.dtd')
            bioc_reader.read()
            docs = bioc_reader.collection.documents
            paper = {}
            text = ''
            tag_isReview = ''
            for doc in docs:
                for para in doc:
                    text_type = para.infons['type']
                    if text_type in text_type_set:
                        if text_type in ['paragraph']:              #遇到正文
                            if not tag_isReview == '':              #不是综述part就解析
                                text = text + para.text + '\n'
                            else:                                   #否则不管
                                continue
                        else:
                            if tag_isReview == '':                  #遇到非正文，若tag为空
                                if all_exist(['CONFLICT OF INTEREST'.lower(),'related work','literature review', 'intro'], para.text.lower()):   
                                                                    #如果带有综述字眼，就改变tag
                                    tag_isReview = text_type
                                else:                               #不带就直接解析
                                    text = text + para.text + '\n'
                            else:                                   
                                if text_type == tag_isReview:       #遇到相同层次的tag，tag改为空，接着解析
                                    tag_isReview = ''
                                    text = text + para.text + '\n'
                                else:                               #否则不管
                                    continue                                                
            paper['text'] = text
            paper['PMID'] = file.replace('.xml','')
            paper_list.append(paper)


paper_df = pd.DataFrame(paper_list, columns=['PMID', 'text'])
paper_df.to_csv('temp/PMID_fulltext.csv', index=False)

data_full_text = pd.read_csv('temp/PMID_fulltext.csv')
data_abs = pd.read_csv('lda/nobelow=30_topic=13_iteration=2000/output/文本话题归类.csv')

data_merge = pd.merge(data_full_text,data_abs,on='PMID',how='inner')

len(data_full_text),len(data_abs),len(data_merge)


data_merge.to_csv('temp/topic_fulltext.csv',index=False)

