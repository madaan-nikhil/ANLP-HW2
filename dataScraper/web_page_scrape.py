import requests
from bs4 import BeautifulSoup
import re
import os
import pickle
from semanticscholar import SemanticScholar

data_dir = 'DATA_DIR'

def extract_pdfs(url, fileDir, fileName):
  reqs = requests.get(url)
  soup = BeautifulSoup(reqs.text, 'html.parser')
  urls = []
  for link in soup.find_all('a'):
      if len(re.findall("\\.[1-9][0-9]*\\.pdf$", link.get('href'))):
        urls.append(link.get('href'))

  path = os.path.join(fileDir, fileName)
  with open(path+".pkl",'wb') as f:
    pickle.dump(urls, f)

def extract_pdfs_topic_wise(query):
  sch = SemanticScholar()
  urls = []
  cnt = 0
  results = sch.search_paper(query=query, year='2021-2022')
  for item in results.items:
      try:
          ArXiv = item.externalIds['ArXiv']
          paper_url = f'https://arxiv.org/pdf/{ArXiv}.pdf'
          print(paper_url)
          urls.append(paper_url)
          cnt += 1
      except:
          continue
      if cnt==5:
          break
  return urls


if __name__ == "__main__":
  urls = ['https://aclanthology.org/events/acl-2022/', 'https://aclanthology.org/events/acl-2021/', 'https://aclanthology.org/events/emnlp-2021/', 'https://aclanthology.org/events/emnlp-2020/','https://aclanthology.org/events/naacl-2022/','https://aclanthology.org/events/naacl-2021/'] 

  for url in urls:
      fileName = url.split('/')[-2]
      print(fileName)
      extract_pdfs(url, os.path.join(data_dir, fileName))

  queries = ['question answering EMNLP', 'Machine translation ACL','Named entity recognition naacl','Sentiment analysis ACL', 'Natural language inference naacl']

  topic_wise_urls = {}
  for query in queries:
    topic_wise_urls[query] = extract_pdfs_topic_wise(query)
  
  with open(os.path.join(data_dir, 'topic_wise_papers.pkl'),'wb') as f:
    pickle.dump(urls, f)