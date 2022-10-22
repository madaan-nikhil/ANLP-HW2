import urllib
import feedparser

# Base api query url
base_url = 'http://export.arxiv.org/api/query?';

# Search parameters
search_query = 'cat:cs.CL' # search for Computation+and+Language in all fields
start = 0
max_results = 100
sortBy = 'lastUpdatedDate'
sortOrder = 'descending'

# query = 'search_query=%s&start=%i&max_results=%i&sortBy=%s&sortOrder=%s&cat=cs.CL' % (search_query,
#                                                      start,
#                                                      max_results,
#                                                      sortBy,
#                                                      sortOrder)

pdf_urls = []

query = f'search_query={search_query}&start={start}&sortBy={sortBy}&sortOrder={sortOrder}&max_results={max_results}'

print(base_url+query)
# blah

# perform a GET request using the base_url and query
response = urllib.request.urlopen(base_url+query).read()

# parse the response using feedparser
feed = feedparser.parse(response)

# print out feed information
print('Feed title: %s' % feed.feed.title)
print('Feed last updated: %s' % feed.feed.updated)

# print opensearch metadata
print('totalResults for this query: %s' % feed.feed.opensearch_totalresults)
print( 'itemsPerPage for this query: %s' % feed.feed.opensearch_itemsperpage)
print( 'startIndex for this query: %s'   % feed.feed.opensearch_startindex)

# Run through each entry, and print out information
for entry in feed.entries:
    print('arxiv-id: %s' % entry.id.split('/abs/')[-1])
    print('Published: %s' % entry.published)
    print('Title:  %s' % entry.title)
    
    author_string = entry.author
    
    try:
        author_string += ' (%s)' % entry.arxiv_affiliation
    except AttributeError:
        pass
    
    print('Last Author:  %s' % author_string)
    
    # feedparser v5.0.1 correctly handles multiple authors, print them all
    try:
        print('Authors:  %s' % ', '.join(author.name for author in entry.authors))
    except AttributeError:
        pass

    # get the links to the abs page and pdf for this e-print
    for link in entry.links:
        if link.rel == 'alternate':
            print('abs page link: %s' % link.href)
        elif link.title == 'pdf':
            print('pdf link: %s' % link.href)
            pdf_urls.append(str(link.href)[:-2])
    
    # The journal reference, comments and primary_category sections live under 
    # the arxiv namespace
    try:
        journal_ref = entry.arxiv_journal_ref
    except AttributeError:
        journal_ref = 'No journal ref found'
    print('Journal reference: %s' % journal_ref)
    
    try:
        comment = entry.arxiv_comment
    except AttributeError:
        comment = 'No comment found'
    print('Comments: %s' % comment)
    print('Primary Category: %s' % entry.tags[0]['term'])
    
    # Lets get all the categories
    all_categories = [t['term'] for t in entry.tags]
    print('All Categories: %s' % (', ').join(all_categories))
    
    # The abstract is in the <summary> element
    print('Abstract: %s' %  entry.summary)