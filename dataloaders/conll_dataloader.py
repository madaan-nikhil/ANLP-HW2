from asyncore import file_dispatcher
import encodings
from pathlib import Path
import re
from transformers import DistilBertTokenizerFast
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import DataLoader
tag2id_path = "tag2id.txt"
PAD_NUM = -100
def tokenize_and_align_labels(tags, encodings, tag2id):
        labels = []
        for i, label in enumerate(tags):
            word_ids = encodings.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to PAD_NUM so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(PAD_NUM)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(tag2id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or PAD_NUM, depending on
                # the label_all_tokens flag.
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
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split('\n'): # splits each line
            if '-DOCSTART-' in line:
                continue
            token, tag = line.split(" -X- _ ") #splits text and tag
            tokens.append(token)
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)

    return token_docs, tag_docs


def get_loaders(file_path, val_size=0.2, tokenizer=None, batch_size = 10):
    global tag2id
    texts, tags = read_conll(file_path)
    unique_tags = set(tag for doc in tags for tag in doc)
    tag2id = {}
    with open('dataloaders/tag2id.txt','r') as fp:
        lines = fp.readlines()
    for line in lines:
        k,v = line.split()
        tag2id[k] = int(v)

    id2tag = {id: tag for tag, id in tag2id.items()}
    # tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')

    train_texts, val_texts, train_tags, val_tags = train_test_split(texts, tags, test_size=val_size)

    train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=False, truncation=False)
    val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=False, truncation=False)

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

class SciDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, window_size = 512, stride=10, O_fraction=0.9):
        self.encodings = encodings
        self.labels = labels
        self.window_size = window_size
        self.stride = stride
        self.pads ={'input_ids':0, 'attention_mask':0, 'token_type_ids': 0}
        self.O_fraction = O_fraction

    def __getitem__(self, idx):
        item = {key: self.sliding_window_overlap(torch.tensor(val[idx]),pad_val=self.pads[key]) for key, val in self.encodings.items()}
        labels = self.sliding_window_overlap(torch.tensor(self.labels[idx]))
        O_nums = int(self.O_fraction)*len(labels)

        labels_O_nums = (labels==tag2id['O']).sum(dim=1)
        mask = labels_O_nums <= O_nums
       
        if not mask.sum(): # if none selected, select atleast 2 
            mask = np.random.choice(len(labels),2)
        
        item = {key : val[mask] for key, val in item.items()}
        labels = labels[mask]
        return (item, labels)

    def __len__(self):
        return len(self.labels)

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
            inputs[key] = torch.cat(inputs[key],0)

        Y = torch.cat([d[1] for d in batch],0)
        return inputs, Y


if __name__ == "__main__":
    file_path = "dataloaders/project-2-at-2022-10-22-19-26-4e2271c2.conll"
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
    train_loader, val_loader = get_loaders(file_path=file_path, val_size=0.2, tokenizer=tokenizer)
    example,labels = next(iter(train_loader))
    print(f"labels: {labels[0][50:]}")
    print(f"labels: {labels[1][50:]}")
    for k,v in example.items():
        print(v.shape)
        if 'input_ids' == k:     
            print(tokenizer.convert_ids_to_tokens(v[0])[:50])
            print(tokenizer.convert_ids_to_tokens(v[1])[:50])
        print(f"{k}: {v[0][:50]}")
        print(f"{k}: {v[1][:50]}")