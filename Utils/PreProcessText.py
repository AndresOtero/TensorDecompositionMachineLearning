import torch
from torchtext import data
import Utils.Constant as C
from Utils.DataLoaderUtil import DataLoaderUtil


class PreProcessText:
    _instance = None

    def __init__(self):
        if not self._instance:
            self.text = None
            self.label = None
            self.train_data= None
            self.test_data=None
        else:
            self.text = self._instance.text
            self.label = self._instance.label
            self.train_data = self._instance.train_data
            self.test_data = self._instance.test_data

    @classmethod
    def get_instance(cls):  # Singleton
        if not cls._instance:
            cls._instance = PreProcessText()
        return cls._instance

    def pre_process_imdb_dataset(self, net_params):
        if not self.text:
            self.text = data.Field(tokenize='spacy',tokenizer_language="en_core_web_sm"
                                   ,fix_length=net_params.get_fixed_length(),batch_first = True )
        return self._pre_process_imdb_dataset(net_params)

    def pre_process_imdb_dataset_include_lengths(self, net_params):
        if not self.text:
            self.text = data.Field(tokenize='spacy', include_lengths=True,fix_length=net_params.get_fixed_length())
        return self._pre_process_imdb_dataset(net_params)

    def _pre_process_imdb_dataset(self, net_params):
        if (not self.train_data) and (not self.test_data):
            self.label = data.LabelField(dtype = torch.float)
            train_data, test_data = DataLoaderUtil.get_imdb_data_train_data(self.text, self.label)
            self.text.build_vocab(train_data,
                              max_size=C.MAX_VOCAB_SIZE,
                              vectors="glove.6B.50d",
                              unk_init=torch.Tensor.normal_)
            self.label.build_vocab(train_data)
            self.test_data=test_data
            self.train_data=train_data

        train_iterator, test_iterator = data.BucketIterator.splits(
            (self.train_data, self.test_data),
            batch_size=C.BATCH_SIZE,
            sort_within_batch=True,
            device=net_params.get_device())
        return [train_iterator, test_iterator]

    def configure_embeddings(self, embedding):
        PAD_IDX = self.text.vocab.stoi[self.text.pad_token]

        pretrained_embeddings = self.text.vocab.vectors
        embedding.weight.data.copy_(pretrained_embeddings)

        UNK_IDX = self.text.vocab.stoi[self.text.unk_token]

        EMBEDDING_DIM = C.EMBEDDING_DIM

        embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    def get_vocab_size(self):
        return len(self.text.vocab)

    def get_padding_index(self):
        return self.text.vocab.stoi[self.text.pad_token]

    def get_padding_idx(self):
        return self.text.vocab.stoi[self.text.pad_token]
