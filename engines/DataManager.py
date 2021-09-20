# -*- coding: utf-8 -*-
# @Time : 2019/6/2 上午10:55
# @Author : Scofield Phil
# @FileName: DataManager.py
# @Project: sequence-lableing-vex

import os, logging
import numpy as np
import pandas as pd
from engines.utils import read_csv_, save_csv_
import jieba
from pathlib import PurePosixPath,Path
from datetime import datetime
from tqdm import tqdm

jieba.setLogLevel(logging.INFO)


class DataManager:
    def __init__(self, configs, logger):

        timestamp = datetime.now().strftime("%Y_%m_%d")

        self.configs=configs
        self.logger = logger
        self.hyphen = configs.hyphen

        self.train_file = configs.train_file
        # self.dev_file = configs.dev_file
        self.dev_file = None
        test_file = timestamp + '.csv'
        self.test_file = Path(configs.test).joinpath(test_file).as_posix()
        
        self.UNKNOWN = "<UNK>"
        self.PADDING = "<PAD>"


        output_test_file = timestamp + '.csv'
        self.output_test_file = Path(configs.predict).joinpath(output_test_file).as_posix()
        self.corr_file = Path(configs.update).joinpath(output_test_file).as_posix()

        self.is_output_sentence_entity = configs.is_output_sentence_entity
        
        output_sentence_entity_file = timestamp + configs.output_sentence_entity_file
        self.output_sentence_entity_file = Path(configs.test).joinpath(output_sentence_entity_file).as_posix()

        self.label_scheme = configs.label_scheme
        self.label_level = configs.label_level
        self.suffix = configs.suffix
        self.labeling_level = configs.labeling_level

        self.batch_size = configs.batch_size
        self.max_sequence_length = configs.max_sequence_length
        self.embedding_dim = configs.embedding_dim

        self.vocabs_dir = configs.vocabs_dir
        self.token2id_file = Path(self.vocabs_dir).joinpath("token2id").as_posix()
        self.label2id_file = Path(self.vocabs_dir).joinpath("label2id").as_posix()

        self.token2id, self.id2token, self.label2id, self.id2label = self.loadVocab()

        self.max_token_number = len(self.token2id)
        self.max_label_number = len(self.label2id)

        jieba.load_userdict(self.token2id.keys())

        self.logger.info("dataManager initialized..\n")

    def loadVocab(self):
        if not os.path.isfile(self.token2id_file):
            self.logger.info("vocab files not exist, building vocab..")
            return self.buildVocab(self.train_file)

        self.logger.info("loading vocab..")
        token2id = {}
        id2token = {}
        with open(self.token2id_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.rstrip()
                token = row.split('\t')[0]
                token_id = int(row.split('\t')[1])
                token2id[token] = token_id
                id2token[token_id] = token

        label2id = {}
        id2label = {}
        with open(self.label2id_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.rstrip()
                label = row.split('\t')[0]
                label_id = int(row.split('\t')[1])
                label2id[label] = label_id
                id2label[label_id] = label

        return token2id, id2token, label2id, id2label

    def buildVocab(self, train_path):
        df_train = read_csv_(train_path, names=["token", "label"], delimiter=self.configs.delimiter)
        tokens = list(set(df_train["token"][df_train["token"].notnull()]))
        labels = list(set(df_train["label"][df_train["label"].notnull()]))
        
        token2id = dict(zip(tokens, range(1, len(tokens) + 1)))
        label2id = dict(zip(labels, range(1, len(labels) + 1)))
        id2token = dict(zip(range(1, len(tokens) + 1), tokens))
        id2label = dict(zip(range(1, len(labels) + 1), labels))
        
        id2token[0] = self.PADDING
        id2label[0] = self.PADDING
        token2id[self.PADDING] = 0
        label2id[self.PADDING] = 0
        id2token[len(tokens) + 1] = self.UNKNOWN
        token2id[self.UNKNOWN] = len(tokens) + 1

        self.saveVocab(id2token, id2label)

        return token2id, id2token, label2id, id2label

    def saveVocab(self, id2token, id2label):
        with open(self.token2id_file, "w", encoding='utf-8') as outfile:
            for idx in id2token:
                outfile.write(id2token[idx] + "\t" + str(idx) + "\n")
        with open(self.label2id_file, "w", encoding='utf-8') as outfile:
            for idx in id2label:
                outfile.write(id2label[idx] + "\t" + str(idx) + "\n")

    def getEmbedding(self, embed_file):
        emb_matrix = np.random.normal(loc=0.0, scale=0.08, size=(len(self.token2id.keys()), self.embedding_dim))
        emb_matrix[self.token2id[self.PADDING], :] = np.zeros(shape=(self.embedding_dim))

        with open(embed_file, "r", encoding="utf-8") as infile:
            for row in infile:
                row = row.rstrip()
                items = row.split()
                token = items[0]
                assert self.embedding_dim == len(
                    items[1:]), "embedding dim must be consistent with the one in `token_emb_dir'."
                emb_vec = np.array([float(val) for val in items[1:]])
                if token in self.token2id.keys():
                    emb_matrix[self.token2id[token], :] = emb_vec

        return emb_matrix

    def nextBatch(self, X, y, start_index):
        last_index = start_index + self.batch_size
        X_batch = list(X[start_index:min(last_index, len(X))])
        y_batch = list(y[start_index:min(last_index, len(X))])

        if last_index > len(X):
            left_size = last_index - (len(X))
            for i in range(left_size):
                index = np.random.randint(len(X))
                X_batch.append(X[index])
                y_batch.append(y[index])
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
        return X_batch, y_batch

    def nextRandomBatch(self, X, y):
        X_batch = []
        y_batch = []
        for i in range(self.batch_size):
            index = np.random.randint(len(X))
            X_batch.append(X[index])
            y_batch.append(y[index])
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
        return X_batch, y_batch

    def padding(self, sample):
        for i in range(len(sample)):
            if len(sample[i]) < self.max_sequence_length:
                sample[i] += [self.token2id[self.PADDING] for _ in range(self.max_sequence_length - len(sample[i]))]
        return sample

    def prepare_train(self, tokens, labels, is_padding=True, return_psyduo_label=False):
        X = []
        y = []
        y_psyduo = []
        tmp_x = []
        tmp_y = []
        tmp_y_psyduo = []

        for record in zip(tokens, labels):
            # check the value one by one
            c = record[0]
            l = record[1]

            if c == -1:  # empty line
                # max_sequence_length would be 200-400
                if len(tmp_x) <= self.max_sequence_length:
                    X.append(tmp_x)
                    y.append(tmp_y)
                    if return_psyduo_label: y_psyduo.append(tmp_y_psyduo)
                tmp_x = []
                tmp_y = []
                if return_psyduo_label: tmp_y_psyduo = []
            else:
                #
                tmp_x.append(c)
                tmp_y.append(l)
                if return_psyduo_label: tmp_y_psyduo.append(self.label2id["O"])
        
        if is_padding:
            X = np.array(self.padding(X))
        else:
            X = np.array(X)

        y = np.array(self.padding(y))
        if return_psyduo_label:
            y_psyduo = np.array(self.padding(y_psyduo))
            return X, y_psyduo

        return  X, y

    def prepare(self, ids, tokens, labels, is_padding=True, return_psyduo_label=False):
        X = []
        y = []
        y_psyduo = []
        hash_ids = []

        tmp_x = []
        tmp_y = []
        tmp_y_psyduo = []
        tmp_ids = []
        
        for record in zip(tokens, labels, ids):
            # check the value one by one
            c = record[0]
            l = record[1]
            i = record[2]

            if c == -1:  # empty line
                # max_sequence_length would be 200-400
                if len(tmp_x) <= self.max_sequence_length:
                    X.append(tmp_x)
                    y.append(tmp_y)
                    hash_ids.append(tmp_ids)
                    if return_psyduo_label: y_psyduo.append(tmp_y_psyduo)
                tmp_x = []
                tmp_y = []
                tmp_ids = []
                if return_psyduo_label: tmp_y_psyduo = []
            else:
                #
                tmp_x.append(c)
                tmp_y.append(l)
                tmp_ids.append(i)
                if return_psyduo_label: tmp_y_psyduo.append(self.label2id["O"])
        
        if is_padding:
            X = np.array(self.padding(X))
            # hash_ids = self.padding(hash_ids)
        else:
            X = np.array(X)

        y = np.array(self.padding(y))

        if return_psyduo_label:
            y_psyduo = np.array(self.padding(y_psyduo))
            return  X, y_psyduo

        return hash_ids, X, y

    def getTrainingSet(self, train_val_ratio=0.9):

        # df_train = read_csv_(self.train_file, names=["token", "label"], delimiter=self.configs.delimiter)

        df_train = read_csv_(self.train_file, names=["token", "label"], delimiter=self.configs.delimiter)

        # map the token and label into id
        df_train["token_id"] = df_train.token.map(lambda x: -1 if str(x) == str(np.nan) else self.token2id[x])
        df_train["label_id"] = df_train.label.map(lambda x: -1 if str(x) == str(np.nan) else self.label2id[x])
        # convert the data in maxtrix
        X, y = self.prepare_train(df_train["token_id"], df_train["label_id"])

        # shuffle the samples
        num_samples = len(X)
        indexs = np.arange(num_samples)
        np.random.shuffle(indexs)
        X = X[indexs]
        y = y[indexs]

        if self.dev_file != None:
            X_train = X
            y_train = y
            X_val, y_val = self.getValidingSet()
        else:
            # split the data into train and validation set
            X_train = X[:int(num_samples * train_val_ratio)]
            y_train = y[:int(num_samples * train_val_ratio)]
            X_val = X[int(num_samples * train_val_ratio):]
            y_val = y[int(num_samples * train_val_ratio):]

        self.logger.info("\ntraining set size: %d, validating set size: %d\n" % (len(X_train), len(y_val)))

        return X_train, y_train, X_val, y_val

    def getValidingSet(self):
        df_val = read_csv_(self.dev_file, names=["token", "label"], delimiter=self.configs.delimiter)

        df_val["token_id"] = df_val.token.map(lambda x: self.mapFunc(x, self.token2id))
        df_val["label_id"] = df_val.label.map(lambda x: -1 if str(x) == str(np.nan) else self.label2id[x])

        X_val, y_val = self.prepare_train(df_val["token_id"], df_val["label_id"])
        return X_val, y_val

    def getTestingSet(self):
        df_test = read_csv_(self.test_file, names=None, delimiter=self.configs.delimiter)
        if len(list(df_test.columns)) == 3:
            df_test.columns = ["id", "token", "label"]
            
        elif len(list(df_test.columns)) == 2:
            df_test.columns = ["id", "token"]

        ids = df_test['id'].tolist()
        df_test = df_test[["token"]]
        df_test["token_id"] = df_test.token.map(lambda x: self.mapFunc(x, self.token2id))
        df_test["token"] = df_test.token.map(lambda x: -1 if str(x) == str(np.nan) else x)

        X_test_id, y_test_psyduo_label = self.prepare(ids, df_test["token_id"], df_test["token_id"],
                                                      return_psyduo_label=True)
        hash_ids, X_test_token, _ = self.prepare(ids, df_test["token"], df_test["token"])

        self.logger.info("\ntesting set size: %d\n" % (len(X_test_id)))

        return hash_ids, X_test_id, y_test_psyduo_label, X_test_token


    def getCorrSet(self, df_corr_part, train_val_ratio=0.9):
        '''
            generate data from corrected sentence predictions
        : param corr_part_df: list type

        '''
        if df_corr_part.empty:
            print("There is no data available to update model")
            return
            
        if len(list(df_corr_part.columns)) == 3:
            df_corr_part.columns = ["id", "token", "label"]
        else:
            print("[-] Wrong format csv")
            return

        # whether to update the token dict
        if self.build_new_token():
            self.token2id = self.build_new_token()
        
        # map the token and label into id
        df_corr_part["token_id"] = df_corr_part.token.map(lambda x: -1 if str(x) == str(np.nan) else self.token2id[x])
        df_corr_part["label_id"] = df_corr_part.label.map(lambda x: -1 if str(x) == str(np.nan) else self.label2id[x])
        
        # convert the data in maxtrix
        ids, X, y = self.prepare(df_corr_part['id'].tolist(), df_corr_part["token_id"], df_corr_part["label_id"])

        # shuffle the samples
        num_samples = len(X)
        indexs = np.arange(num_samples)
        np.random.shuffle(indexs)
        X = X[indexs]
        y = y[indexs]

        # split the data into train and validation set
        X_train = X[:int(num_samples * train_val_ratio)]
        y_train = y[:int(num_samples * train_val_ratio)]
        X_val = X[int(num_samples * train_val_ratio):]
        y_val = y[int(num_samples * train_val_ratio):]

        self.logger.info("\ntraining set size: %d, validating set size: %d\n" % (len(X_train), len(y_val)))

        return ids, X_train, y_train, X_val, y_val


    def build_new_token(self,):
        
        df_train = read_csv_(self.corr_file, names=["ord", "token", "label"], delimiter=self.configs.delimiter)
        # the new_tokens should smaller then ori tokens or just a subset of them
        new_tokens = list(set(df_train["token"][df_train["token"].notnull()]))
        
        if len(set(new_tokens) & set(list(self.token2id.keys()))) < len(new_tokens):
            # there are new tokens
            tokens = list(set(list(self.token2id.keys()) + new_tokens))
        else:
            # it is a subset and exit
            return

        token2id = dict(zip(tokens, range(1, len(tokens) + 1)))
        id2token = dict(zip(range(1, len(tokens) + 1), tokens))
        
        id2token[0] = self.PADDING
        token2id[self.PADDING] = 0
        id2token[len(tokens) + 1] = self.UNKNOWN
        token2id[self.UNKNOWN] = len(tokens) + 1

        with open(self.token2id_file, "w", encoding='utf-8') as outfile:
            for idx in id2token:
                outfile.write(id2token[idx] + "\t" + str(idx) + "\n")

        return token2id


    def mapFunc(self, x, token2id):
        if str(x) == str(np.nan):
            return -1
        elif x not in token2id:
            return token2id[self.UNKNOWN]
        else:
            return token2id[x]

    def prepare_single_sentence(self, sentence):
        if self.labeling_level == 'word':
            if self.check_contain_chinese(sentence):
                sentence = list(jieba.cut(sentence))
            else:
                sentence = list(sentence.split())
        elif self.labeling_level == 'char':
            sentence = list(sentence)

        gap = self.batch_size - 1

        x_ = []
        y_ = []

        for token in sentence:
            try:
                x_.append(self.token2id[token])
            except:
                x_.append(self.token2id[self.UNKNOWN])
            y_.append(self.label2id["O"])

        if len(x_) < self.max_sequence_length:
            sentence += ['x' for _ in range(self.max_sequence_length - len(sentence))]
            x_ += [self.token2id[self.PADDING] for _ in range(self.max_sequence_length - len(x_))]
            y_ += [self.label2id["O"] for _ in range(self.max_sequence_length - len(y_))]
        elif len(x_) > self.max_sequence_length:
            sentence = sentence[:self.max_sequence_length]
            x_ = x_[:self.max_sequence_length]
            y_ = y_[:self.max_sequence_length]

        X = [x_]
        Sentence = [sentence]
        Y = [y_]
        X += [[0 for j in range(self.max_sequence_length)] for i in range(gap)]
        Sentence += [['x' for j in range(self.max_sequence_length)] for i in range(gap)]
        Y += [[self.label2id['O'] for j in range(self.max_sequence_length)] for i in range(gap)]
        X = np.array(X)
        Sentence = np.array(Sentence)
        Y = np.array(Y)

        return X, Sentence, Y

    def check_contain_chinese(self, check_str):
        for ch in list(check_str):
            if u'\u4e00' <= ch <= u'\u9fff':
                return True
        return False
    

    def match_corr_diff(self,):
        '''
            match corrected df and predicted df
        : param df: the originally predicted csv
        : param corr_df: the corrected csv
        '''
        # read predicted and corrected results
        df_pred_ori = read_csv_(self.output_test_file, names=['id','token','label'], delimiter=self.configs.delimiter)
        df_corr_ori = read_csv_(self.corr_file, names=['id','token','label'], delimiter=self.configs.delimiter)
        
        df_pred = df_pred_ori.copy()
        df_corr = df_corr_ori.copy()

        df_pred['index1'] = df_pred.index
        df_corr['index1'] = df_corr.index

        # extract sen id in order to match sentences to train
        df_pred['url_sen_id'] = df_pred['id'].map(lambda x: x.rsplit('_', 1)[0] if str(x) != str(np.nan) else x)
        df_corr['url_sen_id'] = df_corr['id'].map(lambda x: x.rsplit('_', 1)[0] if str(x) != str(np.nan) else x)

        # extract the wrong prediction label
        df = df_corr[df_pred.label != df_corr.label]

        mat_id_sta_indexes = df_corr[df_corr.url_sen_id.isin (df.groupby("url_sen_id").first().index)].index

        df_corr_part = df_corr_ori.iloc[mat_id_sta_indexes ].reset_index()[['id','token','label']]
        df_corr_part['url_sen_id'] = df_corr_part['id'].map(lambda x: x.rsplit('_', 1)[0] if str(x) != str(np.nan) else x)

        # Let's create a row which we want to insert
        rows_num = len(df_corr_part)
        df_dict = df_corr_part.to_dict()
        # df_dict_no_index = {key: df_dict[key].values() for key, value in df_dict.items()}

        new_df_dict = {'id': [], 'token': [], 'label': []}

        for i in tqdm(range(rows_num), desc="Inserting sen dilimiter:"):
            if i < rows_num -1:
                # append the current item first
                new_df_dict['id'].append(df_corr_part['id'][i])
                new_df_dict['token'].append(df_corr_part['token'][i])
                new_df_dict['label'].append(df_corr_part['label'][i])
                # append delimiter if id is not equal
                if df_corr_part['url_sen_id'][i] != df_corr_part['url_sen_id'][i+1]:
                    new_df_dict['id'].append(np.nan)
                    new_df_dict['token'].append(np.nan)
                    new_df_dict['label'].append(np.nan)
    
        df_fin = pd.DataFrame.from_dict(new_df_dict)
        return df_fin
