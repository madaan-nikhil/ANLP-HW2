import glob
import json
import os
import pickle
import re
import scipdf
from spacy.lang.en import English
import tqdm


nlp = English()
tokenizer = nlp.tokenizer

os.makedirs('data', exist_ok = True)
os.makedirs('data/filtered', exist_ok = True)

pdf_urls = []
paths = glob.glob('old_tokens/filtered/*')

for path in paths:
    with open(path, 'rb') as f:
        urls = pickle.load(f)
    pdf_urls = pdf_urls + urls
    
titles = {}

start_idx = 0
for idx, url in enumerate(tqdm.tqdm(pdf_urls[start_idx:])):
    try:
        article_dict = scipdf.parse_pdf_to_dict(url)
    except:
        print(f'Could not parse {url}')
        continue
        
    titles[article_dict['title']] = idx
    
titles.pop('', None)

with open('conference_papers/topic_wise_papers.pkl', 'rb') as f:
    papers_dict = pickle.load(f)
    
all_papers = []
all_meta = []
topics = papers_dict.keys()
for i in tqdm.tqdm(range(5)):
    for topic in topics:
        url = papers_dict[topic][i]
        try:
            article_dict = scipdf.parse_pdf_to_dict(url)
        except:
            print(f'Could not parse {url}')
            continue
        
        if article_dict['title'] in titles:
            print(f"Found same title {article_dict['title']} - this paper {url}, match {pdf_urls[titles[article_dict['title']]]}")
            continue

        meta = {}
        for key in ['authors', 'pub_date', 'title']:
            meta[key] = article_dict[key]
        meta['url'] = url

        text = [article_dict['abstract']]
        for section in article_dict['sections']:
            text.append(section['text'])

        full_text = ' '.join(text)
        full_text = re.sub('\n', '', full_text)
        tokenized = tokenizer(full_text)
        full_text = ""
        for token in tokenized:
            full_text = full_text + str(token) + " "
        full_text = full_text[:-1]

        all_papers.append({'text': full_text, 'paper_id': topic+' '+url})
        all_meta.append(meta)
        
with open(f'data/data_topic.json', 'w') as f:
    json.dump(all_papers, f)

with open(f'data/meta_topic.pkl', 'wb') as f:
    pickle.dump(all_meta, f)