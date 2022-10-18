import glob
import json
import os
import pickle
import random
import re
import scipdf
from spacy.lang.en import English
import tqdm
random.seed(42)


nlp = English()
tokenizer = nlp.tokenizer

os.makedirs('data', exist_ok = True)
os.makedirs('data/filtered', exist_ok = True)

pdf_urls = []
paths = glob.glob('conference_papers/*')

for path in paths:
    with open(path, 'rb') as f:
        urls = pickle.load(f)
        
    urls = random.sample(urls, 20)
    with open('data/filtered/' + path[path.find('/') + 1:], 'wb') as f:
        pickle.dump(urls, f)
        
    pdf_urls = pdf_urls + urls

start_idx = 0

all_papers = []
all_meta = []

for url in tqdm.tqdm(pdf_urls[start_idx:]):
    try:
        article_dict = scipdf.parse_pdf_to_dict(url)
    except:
        print(f'Could not parse {url}')
        continue
    meta = {}
    for key in ['authors', 'pub_date', 'title']:
        meta[key] = article_dict[key]
    meta['url'] = url
        
    text = [article_dict['abstract']]
    for section in article_dict['sections']:
        text.append(section['text'])
        
    url_path = '_'.join(url.split('/'))
    
    full_text = ' '.join(text)
    full_text = re.sub('\n', '', full_text)
    tokenized = tokenizer(full_text)
    full_text = ""
    for token in tokenized:
        full_text = full_text + str(token) + " "
    full_text = full_text[:-1]
    
    all_papers.append({'text': full_text, 'paper_id': url_path})
    all_meta.append(meta)
    
        
with open(f'data/data.json', 'w') as f:
    json.dump(all_papers, f)

with open(f'data/meta.pkl', 'wb') as f:
    pickle.dump(all_meta, f)