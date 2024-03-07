from bioc import BioCAnnotation
from bioc import BioCXMLReader
from bioc import BioCXMLWriter
import os
from tqdm import tqdm
import pandas as pd

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
            abstract = ''
            for doc in docs:
                for para in doc:
                    text_type = para.infons['type']
                    if text_type in ['abstract', 'abstract_title_1']:
                        abstract = abstract + para.text + '\n'
            paper['abstract'] = abstract
            paper['PMID'] = file.replace('.xml','')
            paper_list.append(paper)
    # break        

paper_df = pd.DataFrame(paper_list, columns=['PMID','abstract'])
paper_df.to_csv('temp/PMID_abs.csv', index=False)

data_full_text = pd.read_csv('temp/PMID_abs.csv')
data_meta = pd.read_csv('meta.csv')

data_merge = pd.merge(data_full_text,data_meta,on='PMID',how='inner')

data_merge['Create Date'] = pd.to_datetime(data_merge['Create Date'])

data_merge['Year'] = data_merge['Create Date'].dt.year
data_merge['Month'] = data_merge['Create Date'].dt.month
data_merge['Quarter'] = data_merge['Create Date'].dt.quarter

data_merge['abstract'].isnull().value_counts()

data_merge = data_merge.dropna(subset=['abstract'])
data_merge['abstract'].isnull().value_counts()

data_merge.to_csv('temp/abs.csv', index=False)
len(data_full_text),len(data_meta),len(data_merge)

