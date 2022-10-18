import os
import pickle
import re
import scipdf
from spacy.lang.en import English
import tqdm


nlp = English()
tokenizer = nlp.tokenizer


with open('pdf_urls_100.pkl', 'rb') as f:
    pdf_urls = pickle.load(f)
    
os.makedirs('data', exist_ok = True)
start_idx = 70

for url in tqdm.tqdm(pdf_urls[start_idx:]):
    try:
      article_dict = scipdf.parse_pdf_to_dict(url + '.pdf')
    except:
      print(f'Could not parse {url}')
      continue
    meta = {}
    for key in ['authors', 'pub_date', 'title']:
        meta[key] = article_dict[key]
        
    text = [article_dict['abstract']]
    for section in article_dict['sections']:
        text.append(section['text'])
        
    url_path = '_'.join(url.split('/'))
    with open(f'data/{url_path}.tokens', 'wb') as f:
        full_text = ' '.join(text)
        full_text = re.sub('\n', '', full_text)
        tokenized = tokenizer(full_text)
        pickle.dump(tokenized, f)
        
    with open(f'data/{url_path}.meta', 'wb') as f:
        pickle.dump(meta, f)
