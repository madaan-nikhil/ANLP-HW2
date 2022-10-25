from pathlib import Path
import re

map_labels = {'B-DATASET': 'B-DatasetName', 'B-METRIC': 'B-MetricName','B-TASK': 'B-TaskName',\
              'I-DATASET': 'I-DatasetName', 'I-METRIC': 'I-MetricName','I-TASK': 'I-TaskName',\
              'O':'O'}

# wget https://raw.githubusercontent.com/IBM/science-result-extractor/master/data/TDMSci/conllFormat/train_1500_v2.conll
file_path = Path('train_1500_v2.conll')

raw_text = file_path.read_text().strip()
raw_docs = re.split(r'\n\t?\n', raw_text) # splits each doc
all_docs = ''
for doc in raw_docs:
    tokens = []
    tags = []
    for line in doc.split('\n'): # splits each line
        if '-DOCSTART-' in line:
            continue
        token, _, tag = line.split("\t") #splits text and tag
        all_docs = all_docs + f'{token} -X- _ {map_labels[tag]}\n'
    all_docs = all_docs + '\n'
    
with open('train.conll', 'w') as f:
    f.write(all_docs)