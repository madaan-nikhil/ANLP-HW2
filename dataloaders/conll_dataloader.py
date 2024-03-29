from asyncore import file_dispatcher
import encodings
from pathlib import Path
import re
from transformers import AutoTokenizer, DistilBertTokenizerFast
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import DataLoader
tag2id_path = "tag2id.txt"
PAD_NUM = -100
np.random.seed(0)

def tokenize_and_align_labels(tags, encodings, tag2id):
        labels = []
        for i, label in enumerate(tags):
            word_ids = encodings.word_ids(batch_index=i)
            # tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
            # print(tokenizer.convert_ids_to_tokens(encodings['input_ids'][i][:20]))
            # print(tokenizer('verify'))
            previous_word_idx = None
            label_ids = []
            # print(word_ids[:20])
            # blah
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to PAD_NUM so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(PAD_NUM)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx and tag2id[label[word_idx]] != 13:
                    label_ids.append(tag2id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or PAD_NUM, depending on
                # the label_all_tokens flag.
                elif tag2id[label[word_idx]] == 13:
                    label_ids.append(tag2id[label[word_idx]])

                else:
                    label_ids.append(PAD_NUM)
                
                previous_word_idx = word_idx

            labels.append(label_ids)

        return labels

def encode_tags(tags, encodings, tag2id):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of PAD_NUM
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * PAD_NUM
        arr_offset = np.array(doc_offset)
        # set labels whose first offset position is 0 and the second is not 0
        # print(len(doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)]), len(doc_labels))
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels

def read_conll(file_path):
    file_path = Path(file_path)

    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text) # splits each doc
    token_docs = []
    tag_docs = []
    print(f"Parsing {len(raw_docs)} docs")
    # print(raw_docs)
    for doc in raw_docs:
        tokens = []
        tags = []
        for i,line in enumerate(doc.split('\n')): # splits each line
            # print(line)
            if '-DOCSTART-' in line:
                continue
            try:
                token, tag = line.split("\t") #splits text and tag
            except:
                print(line,i)
            tokens.append(token)
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)

    return token_docs, tag_docs


def get_loaders(file_path, val_size=0.2, tokenizer=None, batch_size = 10):
    global tag2id, id2tag
    raw_texts, raw_tags = read_conll(file_path)
    texts, tags = [], []
    for i in range(len(raw_tags)):
        if len(np.unique(raw_tags[i])) != 1:
            texts.append(raw_texts[i])
            tags.append(raw_tags[i])

    unique_tags = set(tag for doc in tags for tag in doc)
    tag2id = {}
    with open('dataloaders/tag2id.txt','r') as fp:
        lines = fp.readlines()
    for line in lines:
        k,v = line.split()
        tag2id[k] = int(v)

    id2tag = {id: tag for tag, id in tag2id.items()}
    # tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')

    train_texts, val_texts, train_tags, val_tags = train_test_split(texts, tags, test_size=val_size, random_state=0)

    train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding='max_length',max_length=512, truncation=False)
    val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding='max_length',max_length=512, truncation=False)

    train_labels = tokenize_and_align_labels(train_tags, train_encodings, tag2id)
    val_labels = tokenize_and_align_labels(val_tags, val_encodings, tag2id)

    train_encodings.pop("offset_mapping") # we don't want to pass this to the model
    val_encodings.pop("offset_mapping")
    # print(train_texts[0][:50])

    train_dataset = SciDataset(train_encodings, train_labels,stride=1024)
    val_dataset = SciDataset(val_encodings, val_labels,stride=1024)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=val_dataset.collate_batch)
    return train_loader, val_loader

def get_test_loader( tokenizer=DistilBertTokenizerFast.from_pretrained('distilbert-base-cased'),
                     file_path='dataloaders/test_data.txt', 
                     window_size=512):
    global test_token_data, test_word_ids
    test_file = open(file_path, 'r')
    test_raw_data = " ".join(test_file.readlines()).strip()
    test_texts = []
    # split using docstart to get all papers
    test_texts = re.split('-DOCSTART- ',test_raw_data)
    # split using newlines to get all paragraphs
    test_texts = [text.split('\n') for text in test_texts]
    test_texts = [st.strip().split() for text in test_texts for st in text ]
    # test_texts = [[s for s in text if len(s)] for text in test_texts]
    test_token_data = [] # List[List[str]]
    total_tokens = 0
    for line in test_texts:
        if not len(line):
            # print("error",line)
            continue
        total_tokens += len(line)
        test_token_data.append(line) # List[str]
    # print(test_token_data[0])
    test_encodings = tokenizer(test_token_data, is_split_into_words=True, return_offsets_mapping=True, padding=False, truncation=False)
    test_word_ids = [test_encodings.word_ids(i) for i in range(len(test_encodings['input_ids']))]
    test_encodings.pop("offset_mapping")
    test_dataset = SciDataset(test_encodings,window_size=window_size, stride=window_size)
    test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False, collate_fn=test_dataset.collate_batch)

    return test_loader

def convert_tokens_to_words(pred, batch):
    '''
    each input corresponds to one paragraph
    '''

    words_ids = test_word_ids[batch]
    new_pred = []
    prev_id = None

    for i,word_id in enumerate(words_ids):

        if word_id is None or word_id == prev_id:
            continue
        else:
            new_pred.append(pred[i])
            prev_id = word_id


    return new_pred

def save_output(outputs, tokenizer, file_path):
    '''
    This is for 1 input (paper) at a time
    outputs: Dict containing input_ids and preds
    input_ids: Tensor[N,window_size]
    preds: Tensor[N,window_size]
    '''
    output_file = open(file_path,'a+')
    for i,output in enumerate(outputs):
        # each passage
        input_ids = output['input_ids'].flatten()
        preds = output['preds'].flatten()
        input_mask = input_ids != 0 # things which are not [PAD]

        input_ids = input_ids[input_mask]
        preds = preds[input_mask]
        tokens = tokenizer.batch_decode(input_ids)
        preds = convert_tokens_to_words(preds, i).cpu().numpy().astype(int)

        # did'nt -> did 'nt
        # did 'nt -> did '
        words = test_token_data[i]
        assert len(words) == len(preds), f"{len(words)} and {len(preds)}"
        for word, pred in zip(words, preds):
            output_file.write(f"{word}\t{id2tag[int(pred)]}\n")
        output_file.write('\n')
       
        # for input_idss, pred in zip(input_ids, preds):
        #     tokens = tokenizer.convert_ids_to_tokens(input_idss)
        #     words, word_pred = convert_tokens_to_words(tokens, pred)
        #     for token, pp in zip(tokens, pred):
        #         if token in ['[CLS],[PAD]','[SEP]']:
        #             continue
        #         

class SciDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None, window_size = 512, stride=10, O_fraction=0.9):
        self.encodings = encodings
        self.labels = labels
        self.window_size = window_size
        self.stride = stride
        self.pads ={'input_ids':0, 'attention_mask':0, 'token_type_ids': 0}
        self.O_fraction = O_fraction

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # print(self.encodings)
        # blah
        # item = self.encodings[idx]
        if self.labels is None:
            return (item, None)
        
        # labels = self.sliding_window_overlap(torch.tensor(self.labels[idx]))
        labels = torch.tensor(self.labels[idx])
        # O_nums = int(self.O_fraction)*labels.shape[1]

        # labels_O_nums = (labels==tag2id['O']).sum(dim=1)
        # mask = labels_O_nums <= O_nums
       
        # if not mask.sum(): # if none selected, select atleast 2 
        #     mask = np.random.choice(len(labels),2)
        # print(f"before: {len(labels)}")
        # item = {key : val[mask] for key, val in item.items()}
        # labels = labels[mask]
        # print(f"after: {len(labels)}")
        return (item, labels)

    def __len__(self):
        return len(self.encodings['input_ids'])

    def sliding_window_overlap(self, x, pad_val=PAD_NUM):
        i = 0
        curr_inp = []
        while i < x.shape[0]:
            if i + self.window_size >= x.shape[0]:
                pad_vals = torch.tensor([pad_val]*(self.window_size-len(x[i:])))
                new_tens = torch.cat([x[i:],pad_vals],0)
                curr_inp.append(new_tens)
            else:
                curr_inp.append(x[i:i+self.window_size])
            i += self.stride
        batch = torch.stack(curr_inp,0)
        return batch


    @staticmethod
    def collate_batch(batch):
        

        X = [d[0] for d in batch]
        

        keys = list(X[0].keys())
        inputs = defaultdict(list)
        for key in keys:
            for x in X:
                # print(x[key].shape)
                inputs[key].append(x[key])
            inputs[key] = torch.stack(inputs[key],0)
        if batch[0][1] is None :
            # print(inputs)
            return inputs
        
        Y = torch.stack([torch.tensor(d[1]) for d in batch],0)

        
        
        return inputs, Y


if __name__ == "__main__":
    file_path = "dataloaders/new_train.conll"
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased")
    train_loader, val_loader = get_loaders(file_path=file_path, val_size=0.2, tokenizer=tokenizer)
    example,labels = next(iter(train_loader))
    print(example['input_ids'].shape, labels.shape)
    # print(f"labels: {labels[0][50:]}")
    # print(f"labels: {labels[1][50:]}")
    # for k,v in example.items():
    #     print(v.shape)
    #     if 'input_ids' == k:     
    #         print(tokenizer.convert_ids_to_tokens(v[0])[:50])
    #         print(tokenizer.convert_ids_to_tokens(v[1])[:50])
    #     print(f"{k}: {v[0][:50]}")
    #     print(f"{k}: {v[1][:50]}")
    
    # test_loader = get_test_loader(tokenizer=tokenizer)
    # for example in test_loader:
    #     for k,v in example.items():
    #         if 'input_ids' == k:     
    #             print(tokenizer.convert_ids_to_tokens(v[0])[:50])
    #         print(f"{k}: {v[0][:50]}")
    #     break
