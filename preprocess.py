import os
import pickle
import scipdf
from spacy.lang.en import English
import tqdm

nlp = English()
tokenizer = nlp.tokenizer


with open('pdf_urls_100.pkl', 'rb') as f:
    pdf_urls = pickle.load(f)
    
os.makedirs('data', exist_ok = True)

for url in tqdm.tqdm(pdf_urls):
    article_dict = scipdf.parse_pdf_to_dict(url + '.pdf')
    
    meta = {}
    for key in ['authors', 'pub_date', 'title']:
        meta[key] = article_dict[key]
        
    text = [article_dict['abstract']]
    for section in article_dict['sections']:
        text.append(section['text'])
        
    doi = '_'.join(article_dict['doi'].split('/'))
    with open(f'data/{doi}.tokens', 'wb') as f:
        pickle.dump(tokenizer(' '.join(text)), f)
        
    with open(f'data/{doi}.meta', 'wb') as f:
        pickle.dump(meta, f)
