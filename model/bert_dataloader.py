import datasets
import torch.utils.data
import transformers

from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split


# 数据集读取
class HotEventDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encoding = encodings
        self.labels = labels

    # 读取单个样本
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encoding.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        # item = {'input_ids': self.encoding[idx], 'labels': torch.tensor(int(self.labels[idx]))}
        return item

    def __len__(self):
        return len(self.labels)


class BertDataLoader(object):
    """
    Bert 模型数据加载类
    """

    def __init__(self, args, bert_tokenizer, logger):
        self.args = args
        self.bert_tokenizer = bert_tokenizer
        self.logger = logger

    # Function to get token ids for a list of texts
    def encode_fn(self, text_list):
        all_input_ids = []
        for text in text_list:
            input_ids = self.bert_tokenizer.encode(
                text,
                add_special_tokens=True,  # CLS & SEP
                max_length=1024,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            all_input_ids.append(input_ids)
        all_input_ids = torch.cat(all_input_ids, dim=0)
        return all_input_ids
    
    def construct_data(self, dataset):
        text = []
        for x in dataset:
            data = ""
            if x["content"]:
                data += "content:" + x["content"]
            if x["user"]:
                data += "user:" + x["user"]
            if x["user_fans"]:
                data += "user_fans:" + x["user_fans"]
            text.append(data)
        return text

    def load_data(self, data_path, batch_size, is_train=False):
        # 加载原始数据
        dataset = datasets.load_dataset("csv", data_files=data_path, split="train")
        # 分词器，词典
        tokenizer = self.bert_tokenizer.from_pretrained("./pretrain_model/bert-base-chinese")

        if is_train:
            # Split data into train and validation
            train_size = int(0.75 * len(dataset))
            dev_size = len(dataset) - train_size
            train_dataset, dev_dataset = random_split(dataset, [train_size, dev_size])

            print(train_dataset[0])
            
            train_text = self.construct_data(train_dataset)
            dev_text = self.construct_data(dev_dataset)
            print(len(train_text))
            print(len(dev_text))
            # train_text = [x["content"] + "[SEP]" + x["user"] + "[SEP]" + x["user_fans"]  for x in train_dataset]
            # dev_text = [x["content"] + "[SEP]" + x["user"] + "[SEP]" + x["user_fans"]  for x in dev_dataset]
            train_label = [x["label"] for x in train_dataset]
            dev_label = [x["label"] for x in dev_dataset]

            train_encoding = tokenizer(train_text, truncation=True, padding=True, max_length=128)
            dev_encoding = tokenizer(dev_text, truncation=True, padding=True, max_length=128)

            # Create train and validation dataloaders
            train_dataset = HotEventDataset(train_encoding, train_label)
            dev_dataset = HotEventDataset(dev_encoding, dev_label)

            # 单个读取到批量读取
            train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            dev_dataloader = DataLoader(dev_dataset, batch_size=64, shuffle=True)

            return train_dataloader, dev_dataloader

        else:
            test_text = self.construct_data(dataset)
            # test_text = [x["content"] + "[SEP]" + x["user"] + "[SEP]" + x["user_fans"] for x in dataset]
            test_label = [x["label"] for x in dataset]

            test_encoding = tokenizer(test_text, truncation=True, padding=True, max_length=128)
            test_dataset = HotEventDataset(test_encoding, test_label)

            test_dataloader = DataLoader(test_dataset, batch_size=64)

            return test_dataloader
