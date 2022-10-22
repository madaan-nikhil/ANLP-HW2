from asyncore import file_dispatcher
import encodings
from pathlib import Path
import re
from transformers import DistilBertTokenizerFast
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from collections import defaultdict

def tokenize_and_align_labels(tags, encodings, tag2id):
        labels = []
        for i, label in enumerate(tags):
            word_ids = encodings.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(tag2id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        return labels

def encode_tags(tags, encodings, tag2id):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)
        # set labels whose first offset position is 0 and the second is not 0
        print(len(doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)]), len(doc_labels))
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


def get_loaders(file_path, val_size=0.2, tokenizer=None):
    
    texts, tags = read_conll(file_path)
    unique_tags = set(tag for doc in tags for tag in doc)
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}
    # tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')

    train_texts, val_texts, train_tags, val_tags = train_test_split(texts, tags, test_size=val_size)

    train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=False)
    val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=False)

    train_labels = tokenize_and_align_labels(train_tags, train_encodings, tag2id)
    val_labels = tokenize_and_align_labels(val_tags, val_encodings, tag2id)

    train_encodings.pop("offset_mapping") # we don't want to pass this to the model
    val_encodings.pop("offset_mapping")
    # print(train_texts[0][:50])
    train_dataset = SciDataset(train_encodings, train_labels)
    val_dataset = SciDataset(val_encodings, val_labels)

    return train_dataset, val_dataset

class SciDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        labels = torch.tensor(self.labels[idx])
        (return item, labels)

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def collate_batch(batch):
        X = [d[1] for d in batch]
        keys = list(d[0].keys())
        inputs = defaultdict(list)
        for x in X:
            for key in keys:
                inputs[key].append(x[key])
        
        Y = [d[1] for d in batch]
        return inputs, Y


if __name__ == "__main__":
    file_path = "dataloaders/project-2-at-2022-10-22-19-26-4e2271c2.conll"
    train_dataset, val_dataset = get_loaders(file_path=file_path, val_size=0.2)
    example = train_dataset[0]
    for k,v in example.items():
        if 'input_ids' == k:
            tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
            print(tokenizer.convert_ids_to_tokens(v)[:50])
        print(f"{k}: {v[:50]}")