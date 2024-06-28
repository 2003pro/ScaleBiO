from typing import Iterator
from datasets import load_dataset, concatenate_datasets
import datasets
from torch.utils.data import Dataset, IterableDataset, DataLoader
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import os
from transformers import AutoTokenizer
import transformers
from functools import partial
from prompt_maker import PromptMaker
import glob
import copy
import json
import random
from tqdm import tqdm
import jsonlines
import multiprocessing
from time import time

def flat_map_function(element):
    # Replace with your actual logic to return a list
    return [element, element * 2]

# Function to flatten the list
def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]



class JsonDataset(Dataset):
    def __init__(self,
            json_data: Union[os.PathLike, List[Dict]],
            transform: Callable = None, 
            shuffle: bool = True,
            train=False,
            chunk_long_text=False,
        ):
        
        json_filenames = glob.glob(json_data)

        self.source_dict = {fn: idx for fn, idx in zip(json_filenames, range(len(json_filenames)))}
        self.data_files = json_filenames
        data_list = []

        print(self.source_dict)

        for fn, idx in self.source_dict.items():
            if fn.endswith('.json'):
                data = json.load(open(fn))
            elif fn.endswith('.jsonl'):
                data = [i for i in jsonlines.Reader(open(fn))]
            else:
                raise ValueError('Input File Is Either Json or Jsonline')
            for d in data: d["source"] = idx
            data_list.extend(data)
        
        self.data=data_list
        if shuffle:
            random.shuffle(self.data)
        self.transform=transform
        if transform:
            chunked=[]
            if chunk_long_text:
                for i in tqdm(range(len(self.data)), miniters=500):
                    trans=transform(self.data[i])
                    chunked.extend(trans)
                self.data=chunked
            else:
                for i in tqdm(range(len(self.data)), miniters=500):
                    self.data[i] = {**self.data[i], **transform(self.data[i])}
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.transform:
            return {'input_ids': self.data[index]['input_ids'], 
                    'source': self.data[index]['source'], 
                    'labels': self.data[index]['labels']}
        else:
            return self.data[index]


class JsonDataset_deprecate(Dataset):
    def __init__(self,
            json_data: Union[os.PathLike, List[Dict]],
            transform: Callable = None, 
            shuffle: bool = True,
            seed=42,
            train=False,
        ):
        
        json_filenames = glob.glob(json_data)

        self.source_dict = {fn: idx for fn, idx in zip(json_filenames, range(len(json_filenames)))}
        self.data_files = json_filenames
        data_list = []

        print(self.source_dict)

        for fn, idx in self.source_dict.items():
            data = load_dataset(
                "json", 
                data_files=fn, 
                split="train", 
                streaming=False, 
                keep_in_memory=True
            )
            data = data.map(lambda examples: {"source": idx})
            data_list.append(data)
        
        self.data=concatenate_datasets(data_list, split="train")

        if shuffle:
            self.data = self.data.shuffle(seed=seed)
        self.transform=transform
        # if transform:
            # self.data = self.data.map(transform)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.transform:
            source=self.data[index]['source']
            res=self.transform(self.data[index])
            return {'input_ids': res['input_ids'], 
                    'source': source, 
                    'labels': res['labels']}
        else:
            return self.data[index]

class StreamJsonDataset(IterableDataset):
    """
    DO NOT USE
    stream for large scale data
    first shuffle data sources, then shuffle data in the shuffle buffer
    """

    def __init__(self,
            json_data: Union[os.PathLike, List[Dict]],
            transform: Callable = None, 
            shuffle: bool = True,
            shuffle_buffer_size: int = 1000,
            seed=42,
            train=False,
        ):

        if train:
            json_filenames = glob.glob(json_data)
        else:
            json_filenames = [json_data]
        
        self.source_dict = {fn: idx for fn, idx in zip(json_filenames, range(len(json_filenames)))}
        
        self.data_files = json_filenames

        dataiter_list=[]
        for fn, idx in self.source_dict.items():
            dataiter = load_dataset(
                "json", 
                data_files=fn, 
                split="train", 
                streaming=True, 
                keep_in_memory=True
            )
            dataiter = dataiter.map(lambda x, source: {"source": source}, fn_kwargs={"source": idx})
            dataiter_list.append(dataiter)
            
        self.dataiter=concatenate_datasets(dataiter_list)
        print(type(self.dataiter))
        if shuffle:
            self.dataiter = self.dataiter.shuffle(buffer_size=shuffle_buffer_size, seed=seed)

        if transform:
            self.dataiter = self.dataiter.map(transform)

    def __iter__(self):
        return iter(self.dataiter)

def tokenize_prompt(data_point: Dict = None,
    max_length: int = 256,
    tokenizer = None,
    prompt_maker = None,
    response_loss_only: bool = True,
    padding: Union[bool, str] = False,
    truncation: bool = True,
):
    assert prompt_maker is not None, "please provide prompt_maker"
    assert tokenizer is not None, "please provide tokenizer"

    full_text = prompt_maker.get_full(data_point)
    full_tokenized=tokenizer(full_text, max_length=max_length, 
                        truncation=truncation, padding=padding, 
                        add_special_tokens=True, )["input_ids"]
    ## WARNING some tokenizer may not automatically add eos token

    if full_tokenized[-1] != tokenizer.eos_token_id: 
        full_tokenized = full_tokenized + [tokenizer.eos_token_id]
    
    if not response_loss_only:
        return {"input_ids": full_tokenized, "labels": copy.deepcopy(full_tokenized)}
    else:
        input_token=tokenizer(prompt_maker.get_input(data_point), 
                              max_length=max_length, 
                              truncation=truncation, padding=padding, 
                              add_special_tokens=False, )["input_ids"]
        labels = [-100] * len(input_token) + full_tokenized[len(input_token):]
        # attention_mask = [1] * len(full_tokenized)
        return {"input_ids": full_tokenized, "labels": labels}

def tokenize_conversion(data_point: Dict = None,
    max_length: int = 256,
    tokenizer = None,
    prompt_maker = None,
    response_loss_only: bool = True,
    padding: Union[bool, str] = False,
    truncation: bool = True,
):
    assert prompt_maker is None, "no need to use prompt maker"
    assert tokenizer is not None, "please provide tokenizer"

    conversations = data_point['items']

    input_ids = []
    labels = []
    for c in conversations:
        if c['from'] == 'human':
            text = "###Human: " + c['value']
            tokens = tokenizer(text, max_length=999999999999, 
                        truncation=truncation, padding=padding, 
                        add_special_tokens=False, )["input_ids"]
            input_ids += tokens
            labels += [-100] * len(tokens)
        elif c['from'] == 'gpt':
            text = "###Assistant: " + c['value']
            tokens = tokenizer(text, max_length=999999999999, 
                        truncation=truncation, padding=padding, 
                        add_special_tokens=True, )["input_ids"]

            if len(tokens) == 0 or tokens[-1] != tokenizer.eos_token_id: 
                tokens += [tokenizer.eos_token_id]
            input_ids += tokens
            labels += tokens
        else:
            raise NotImplementedError('Wrong from id in share gpt data')
        if len(input_ids) >= max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
            break


    return {"input_ids": input_ids, "labels": copy.deepcopy(labels)}

def tokenize_text_only(data_point: Dict = None,
    max_length: int = 256,
    tokenizer = None,
    prompt_maker = None,
    response_loss_only: bool = True,
    padding: Union[bool, str] = False,
    truncation: bool = True,
):
    assert prompt_maker is None, "no need to use prompt maker"
    assert tokenizer is not None, "please provide tokenizer"

    text=data_point['text']
    input_ids = []
    labels = []

    tokens = tokenizer(text, max_length=999999999999, 
                        truncation=truncation, padding=padding, 
                        add_special_tokens=True, )["input_ids"]
    
    if len(tokens) == 0 or tokens[-1] != tokenizer.eos_token_id: 
        tokens += [tokenizer.eos_token_id]

    res=[]
    for i in range(0, len(tokens), max_length):
        res.append({"input_ids": tokens[i : i + max_length], "labels": copy.deepcopy(tokens[i : i + max_length]), "source": data_point["source"]})

    return res
