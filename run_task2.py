import os
import time
import numpy as np
import warnings
import argparse
import ipdb
import json
import pathlib
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.nn import CrossEntropyLoss
from torch.nn.functional import relu, tanh
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertModel, modeling
from pytorch_pretrained_bert.modeling import PreTrainedBertModel
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from sklearn.metrics import precision_score, recall_score, f1_score

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--checkdir", type=str, default="checkpoints")
    parser.add_argument("--trainset", type=str, default='../propaganda-detection/data/task2_train.json')
    parser.add_argument("--devset", type=str, default='../propaganda-detection/data/task2_dev.json')
    parser.add_argument("--testset", type=str, default='../propaganda-detection/data/task2_dev_test.json')
    parser.add_argument("--techniques", type=str, default='../propaganda-detection/techniques_list_task1-2.txt')
    parser.add_argument("--plm", type=str, default='bert-base-multilingual-cased')
    parser.add_argument("--name", type=str, default='debug')
    parser.add_argument("--weights", default=None)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--plm_lr", type=float, default=3e-5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ep", type=int, default=30)
    parser.add_argument("--gpu", action='store_false')
    parser.add_argument("--seed", type=int, default=2022)
    args = parser.parse_args()
    return args
 
class PropDataset(data.Dataset):
    def __init__(self, examples, isDev=False, isTest=False):
        self.isTest = isTest
        if self.isTest:
            sents, ids = [], []
            for example in examples:
                id = str(example['id'])
                ids.append([id])
                
                words = example['text'].split(' ')
                sents.append(["[CLS]"] + words + ["[SEP]"])
            
            self.sents, self.ids = sents, ids
            return

        # get the list of each tweet's id, text, and (multiple) labels in it
        ids, texts, labels = [], [], []
        for example in examples:
            ids.append(str(example['id']))
            texts.append(example['text'])
            
            example_labels = example['labels']
            example_labels = [[int(ex_lbl['start']), int(ex_lbl['end']), ex_lbl['technique'], 0, 0] for ex_lbl in example_labels] 
            example_labels = sorted(example_labels)
            if example_labels:
                length = max([ex_lbl[1] for ex_lbl in example_labels]) 
                visit = np.zeros(length)
                upd_example_labels = []
                for ex_lbl in example_labels:
                    if sum(visit[ex_lbl[0]:ex_lbl[1]]):
                        ex_lbl[3] = 1
                        # here not doing 'visit[ex_lbl[0]:ex_lbl[1]] = 1' is wrong since this label can overlap with future labels?
                    else:
                        visit[ex_lbl[0]:ex_lbl[1]] = 1
                    upd_example_labels.append(ex_lbl)
                example_labels = upd_example_labels
            labels.append(example_labels)
        texts = [[[id, sent, 0, len(sent)]] for (id, sent) in zip(ids, texts) if len(sent)!=0]

        # texts_plus_labels[0] is a list containing tweet_text;label_i for i in # labels in tweet
        texts_plus_labels = []
        for ex_sents, ex_labels in zip(texts, labels):
            sent = ex_sents[0] # sen = [id, sentence_text, start_offset, end_offset]
            sent_labels = [] 
            for l in ex_labels:
                if l[1]>sent[3]:
                    l[1]=sent[3]
                sent_labels.append(sent + l)
            if len(ex_labels) == 0:
                dummy = [0, 0, 'O', 0, 0]
                sent_labels.append(sent + dummy)
            texts_plus_labels.append(sent_labels)
        
        # for (sentence; label) data points | here overlapping labels is tackled
        words, tags, ids = [], [], []
        for sent_labels in texts_plus_labels:
            # sent_labels = list of [id, sentence_text, sent_s, sent_e, lbl_s, lbl_e, tech, ovl_lbls, prtl_lbl]
            # for wanlp, the sentence_text will be same since there are multiple labels in the same sentence (which are now separate data points)
            tmp_doc, tmp_label, tmp_id = [], [], []
            
            tmp_sen = sent_labels[0][1]
            tmp_i = sent_labels[0][0]
            label = ['O'] * len(tmp_sen.split(' '))
            
            for sent_label in sent_labels:
                assert tmp_sen == sent_label[1], ipdb.set_trace()
                tokens = sent_label[1].split(' ') # IMPORTANT: Word tokenization using ' '
                token_len = [len(token) for token in tokens]
                
                if (not isDev) and sent_label[7]: # overlapping?
                    tmp_label.append(label)
                    tmp_doc.append(tmp_sen.split(' '))
                    tmp_id.append(tmp_i)
                
                start = sent_label[4] - sent_label[2]
                end = sent_label[5] - sent_label[2]

                if sent_label[6] != 'O':
                    for i in range(1, len(token_len)): 
                        token_len[i] += token_len[i-1] + 1
                    # if not hp.wanlp:
                    #     token_len[-1] += 1 # this is mp for '\n' character
                    token_len = np.asarray(token_len)

                    s_ind = np.min(np.where(token_len > start))
                    tmp = np.where(token_len >= end)
                    e_ind = np.min(tmp) if len(tmp[0]) != 0 else s_ind

                    for i in range(s_ind, e_ind+1):
                        label[i] = sent_label[6]

            tmp_label.append(label)
            tmp_doc.append(tmp_sen.split(' '))
            tmp_id.append(tmp_i)
            # len(tmp_label) need not be == len(article)

            words.append(tmp_doc) 
            tags.append(tmp_label)
            ids.append(tmp_id) 

        flat_words, flat_tags, flat_ids = [], [], []
        for article_w, article_t, article_id in zip(words, tags, ids):
            for sentence, tag, id in zip(article_w, article_t, article_id):
                flat_words.append(sentence) # sentence is list of tokens
                flat_tags.append(tag) # tag is list of labels/techs
                flat_ids.append(id) # id is a string

        sents, ids = [], [] 
        tags_li = [[] for _ in range(2)]
   
        for word, tag, id in zip(flat_words, flat_tags, flat_ids):
            words = word # list of tokens
            tags = tag # list of labels/techs
            # id is a string
            assert len(words)==len(tags), ipdb.set_trace()

            ids.append([id])
            sents.append(["[CLS]"] + words + ["[SEP]"])
            tmp_tags = []
            tmp_tags.append(['O']*len(tags))
            tmp_tags.append(['Non-prop'])

            for j, tag in enumerate(tags):
                assert tag in LABELS[0]
                if tag != 'O' and tag in LABELS[0]:
                    tmp_tags[0][j] = tag
                    tmp_tags[1] = ['Prop']

            for i in range(2):
                tags_li[i].append(["<PAD>"] + tmp_tags[i] + ["<PAD>"])

        self.sents, self.ids, self.tags_li = sents, ids, tags_li
        assert len(sents) == len(ids) == len(tags_li[0])
        for (sent,id,tag) in zip(sents, ids, tags_li[0]):
            assert len(sent) == len(tag), ipdb.set_trace()

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        words = self.sents[idx]
        ids = self.ids[idx]

        if self.isTest:
            x, is_heads = [], []
            for w in words:
                tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
                xx = tokenizer.convert_tokens_to_ids(tokens)

                is_head = [1] + [0]*(len(tokens) - 1)
                if len(xx) < len(is_head): # this happens happen w is empty space
                    xx = xx + [100] * (len(is_head) - len(xx))

                x.extend(xx)
                is_heads.extend(is_head)

            seqlen = len(x)
            words = " ".join(ids + words)
            att_mask = [1] * seqlen
            
            return words, x, is_heads, att_mask, seqlen

        tags = list(list(zip(*self.tags_li))[idx])
        # tags = [['<PAD>', 'O', 'O', 'O', 'Loaded_Language', 'Repetition', '<PAD>'], ['<PAD>', 'Prop', '<PAD>']]

        x, is_heads = [], []
        y = [[] for _ in range(2)]
        tt = [[] for _ in range(2)]
    
        for w, t in zip(words, tags[0]):
            tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = tokenizer.convert_tokens_to_ids(tokens)

            is_head = [1] + [0]*(len(tokens) - 1)
            if len(xx) < len(is_head): # this happens happen w is empty space
                xx = xx + [100] * (len(is_head) - len(xx))

            t = [t] + [t] * (len(tokens) - 1)
            y[0].extend([tag2idx[0][each] for each in t])
            tt[0].extend(t)

            x.extend(xx)
            is_heads.extend(is_head)
        if tags[1][1] == 'Non-prop':
            y[1].extend([1, 0])
            tt[1].extend(['Non-prop'])
        elif tags[1][1] == 'Prop':
            y[1].extend([0, 1])
            tt[1].extend(['Prop'])

        seqlen = len(y[0])
        assert seqlen == len(is_heads) == len(x) and len(words) == len(tags[0]), ipdb.set_trace()
        words = " ".join(ids + words) # "id word1 word2 ..."
        for i in range(2):
            tags[i]= "<S>".join(tags[i]) 
        att_mask = [1] * seqlen
        return words, x, is_heads, att_mask, tags, y, seqlen

def pad_for_test(batch):
    f = lambda x: [sample[x] for sample in batch]
    # batch consists of words, x, is_heads, att_mask, seqlen

    words = f(0)
    is_heads = f(2)
    seqlen = f(-1)
    maxlen = 210
    # maxlen = 420 # this is needed for processing en2ar translated text

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    x = torch.LongTensor(f(1, maxlen))
    att_mask = f(-2, maxlen)

    return words, x, is_heads, att_mask, seqlen

def pad(batch):
    f = lambda x: [sample[x] for sample in batch]
    # batch consists of words, x, is_heads, att_mask, tags, y, seqlen
    words = f(0)
    is_heads = f(2)
    seqlen = f(-1)
    maxlen = 210
    # maxlen = 420 # this is needed for processing en2ar translated text

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    x = torch.LongTensor(f(1, maxlen))

    att_mask = f(-4, maxlen)
    y = []
    tags = []

    y.append(torch.LongTensor([sample[-2][0] + [0] * (maxlen-len(sample[-2][0])) for sample in batch]))
    y.append(torch.LongTensor([sample[-2][1] for sample in batch]))

    for i in range(2):
        tags.append([sample[-3][i] for sample in batch])

    return words, x, is_heads, att_mask, tags, y, seqlen

class BertMultiTaskLearning(PreTrainedBertModel):
    def __init__(self, config):
        super(BertMultiTaskLearning, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.ModuleList([nn.Linear(config.hidden_size, len(LABELS[i])) for i in range(2)])
        self.apply(self.init_bert_weights)
        self.masking_gate = nn.Linear(2, 1)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        token_level = self.classifier[0](sequence_output)
        sen_level = self.classifier[1](pooled_output)

        gate = torch.sigmoid(self.masking_gate(sen_level))

        dup_gate = gate.unsqueeze(1).repeat(1, token_level.size()[1], token_level.size()[2])
        wei_token_level = torch.mul(dup_gate, token_level)

        logits = [wei_token_level, sen_level]
        y_hats = [logits[i].argmax(-1) for i in range(2)]

        return logits, y_hats

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, filepath='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.filepath = filepath

    def __call__(self, val_loss, model, filepath):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, filepath)
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, filepath)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, filepath):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), filepath)
        self.val_loss_min = val_loss

def train(model, iterator, optimizer, criterion, binary_criterion, device='cuda'):
    model.train()

    train_losses = []
    for k, batch in enumerate(iterator):
        words, x, is_heads, att_mask, tags, y, seqlens = batch
        att_mask = torch.Tensor(att_mask)

        optimizer.zero_grad()
        logits, _ = model(x, attention_mask=att_mask)

        loss = []
        for i in range(2):
            logits[i] = logits[i].view(-1, logits[i].shape[-1])
        y[0] = y[0].view(-1).to(device)
        y[1] = y[1].float().to(device)
        loss.append(criterion(logits[0], y[0]))
        loss.append(binary_criterion(logits[1], y[1]))

        joint_loss = hp.alpha*loss[0] + (1-hp.alpha)*loss[1]

        joint_loss.backward()
        optimizer.step()
        train_losses.append(joint_loss.item())

        if k%10==0: # monitoring
            print("step: {}, loss: {}".format(k,joint_loss.item()))

    train_loss = np.average(train_losses)

    return train_loss

def eval(model, iterator, criterion, binary_criterion, device='cuda'):
    print("Evaluating the model...\n")
    model.eval()

    Words, Is_heads = [], []
    Tags, Y, Y_hats = [[] for _ in range(2)], [[] for _ in range(2)], [[] for _ in range(2)]
    valid_losses = []
    with torch.no_grad():
        for _ , batch in enumerate(iterator):
            words, x, is_heads, att_mask, tags, y, seqlens = batch
            att_mask = torch.Tensor(att_mask)
            logits, y_hats = model(x, attention_mask=att_mask) # logits[0].shape=16,210,22; y_hats[0].shape=16,210
            
            loss = []
            loss.append(criterion(logits[0].view(-1, logits[0].shape[-1]), y[0].view(-1).to(device)))
            loss.append(binary_criterion(logits[1].view(-1, logits[1].shape[-1]), y[1].float().to(device)))
            joint_loss = hp.alpha*loss[0] + (1-hp.alpha)*loss[1]
            valid_losses.append(joint_loss.item())
            
            Words.extend(words)
            Is_heads.extend(is_heads)
            for i in range(2):
                Tags[i].extend(tags[i])
                Y[i].extend(y[i].cpu().numpy().tolist())
                Y_hats[i].extend(y_hats[i].cpu().numpy().tolist())

    valid_loss = np.average(valid_losses)

    for idx, is_heads in enumerate(Is_heads):
        Y[0][idx] = Y[0][idx][:len(is_heads)]
        Y[0][idx] = [y for head, y in zip(is_heads, Y[0][idx]) if head == 1]
        
        Y_hats[0][idx] = Y_hats[0][idx][:len(is_heads)]
        Y_hats[0][idx] = [hat for head, hat in zip(is_heads, Y_hats[0][idx]) if head == 1]

        Y[0][idx] = Y[0][idx][1:-1]
        Y_hats[0][idx] = Y_hats[0][idx][1:-1]
        
        assert len(Y_hats[0][idx]) == len(Y[0][idx]) == len(Tags[0][idx].split('<S>')[1:-1]) == len(Words[idx].split(' ')[2:-1]), ipdb.set_trace()

    print("\nSentence Classification performance")
    sc_y_pred = Y_hats[1]
    sc_y_true = [gold[1] for gold in Y[1]]
    print("y_pred", np.bincount(sc_y_pred))
    print("y_true", np.bincount(sc_y_true))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision = precision_score(y_true=sc_y_true, y_pred=sc_y_pred)
        recall = recall_score(y_true=sc_y_true, y_pred=sc_y_pred)
        f1 = f1_score(y_true=sc_y_true, y_pred=sc_y_pred)
    print("(p, r, f1):", precision, recall, f1)

    print("\nSpan Identification performance")
    si_y_pred, si_y_true = [], []
    for y_true, y_pred in zip(Y[0], Y_hats[0]):
        si_y_pred += y_pred
        si_y_true += y_true
    si_y_pred, si_y_true = np.array(si_y_pred), np.array(si_y_true)

    num_predicted, num_correct, num_gold = 0, 0, 0
    num_predicted += len(si_y_pred[si_y_pred>1])
    num_correct += (np.logical_and(si_y_true==si_y_pred, si_y_true>1)).astype(np.int).sum()
    num_gold += len(si_y_true[si_y_true>1])
    
    print("y_pred", np.bincount(si_y_pred))
    print("y_true", np.bincount(si_y_true))
    try:
        precision = num_correct / num_predicted
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2*precision*recall / (precision + recall)
    except ZeroDivisionError:
        if precision*recall==0:
            f1=1.0
        else:
            f1=0
    
    print("(p, r, f1):", precision, recall, f1)
    return precision, recall, f1, valid_loss

def test(model, iterator, device='cuda'):
    print("Testing the model...\n")
    model.eval()

    Words, Is_heads = [], []
    Y_hats = [[] for _ in range(2)]
    with torch.no_grad():
        for _ , batch in enumerate(iterator):
            words, x, is_heads, att_mask, seqlens = batch
            att_mask = torch.Tensor(att_mask)
            logits, y_hats = model(x, attention_mask=att_mask) # logits[0].shape=16,210,22; y_hats[0].shape=16,210
            Words.extend(words)
            Is_heads.extend(is_heads)
            for i in range(2):
                Y_hats[i].extend(y_hats[i].cpu().numpy().tolist())

    all_preds = []
    for idx, is_heads in enumerate(Is_heads):
        Y_hats[0][idx] = Y_hats[0][idx][:len(is_heads)]
        Y_hats[0][idx] = [hat for head, hat in zip(is_heads, Y_hats[0][idx]) if head == 1]
        Y_hats[0][idx] = Y_hats[0][idx][1:-1]
        assert len(Y_hats[0][idx]) == len(Words[idx].split(' ')[2:-1]), ipdb.set_trace()

        all_preds.append(Y_hats[0][idx])
    
    return all_preds

if __name__=="__main__":
    global hp
    hp = get_arg_parser()

    # pl.seed_everything(args.seed)
    device = "cuda:0" if hp.gpu else "cpu"
    num_epochs = hp.ep
    batch_size = hp.bs
    num_layers = hp.num_layers
    
    all_ptechs_path = hp.techniques
    with open(all_ptechs_path, "r", encoding='utf-8') as f:
        all_ptechs = f.readlines()
    all_ptechs = [line.strip() for line in all_ptechs]
    all_ptechs = [line for line in all_ptechs if len(line) != 0]
    all_ptechs.remove("no technique")
    global num_ptechs
    num_ptechs = len(all_ptechs)
    print("%d propaganda techniques, found in %s, will be used in classification layers of the model"%(num_ptechs, all_ptechs_path))

    global LABELS
    LABELS = []
    LABELS.append(("<PAD>", "O"))
    for lbl in all_ptechs:
        LABELS[0] += (lbl,)
    LABELS.append(("Non-prop", "Prop"))

    global tag2idx
    global idx2tag
    tag2idx, idx2tag = [], []
    for i in range(2):
        tag2idx.append({tag:idx for idx, tag in enumerate(LABELS[i])})
        idx2tag.append({idx:tag for idx, tag in enumerate(LABELS[i])})

    global tokenizer
    tokenizer = BertTokenizer.from_pretrained(hp.plm, do_lower_case=False)

    # model
    model = BertMultiTaskLearning.from_pretrained(hp.plm)
    model = nn.DataParallel(model)
    model.to(device)

    # dataset
    if hp.train:
        with open(hp.trainset, 'r') as f:
            train_examples = json.load(f)
        with open(hp.devset, 'r') as f:
            dev_examples = json.load(f)
    
        train_dataset = PropDataset(train_examples, isDev=False, isTest=False)
        dev_dataset = PropDataset(dev_examples, isDev=True, isTest=False)

        train_iter = data.DataLoader(dataset=train_dataset,
                                batch_size=hp.bs,
                                shuffle=True,
                                num_workers=1,
                                collate_fn=pad)
        dev_iter = data.DataLoader(dataset=dev_dataset,
                                batch_size=hp.bs,
                                shuffle=False,
                                num_workers=1,
                                collate_fn=pad)
    else:
        with open(hp.testset, 'r') as f:
            test_examples = json.load(f)
        test_dataset = PropDataset(test_examples, isDev=False, isTest=True)
        test_iter = data.DataLoader(dataset=test_dataset,
                                    batch_size=hp.bs,
                                    shuffle=False,
                                    num_workers=1,
                                    collate_fn=pad_for_test)

    if hp.weights:
        model.load_state_dict(torch.load(hp.weights))

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    binary_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([3932/14263]).cuda())

    if hp.train:
        warmup_proportion = 0.1
        num_train_optimization_steps = int(len(train_dataset) / hp.bs ) * hp.ep
        param_optimizer = list(model.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = BertAdam(optimizer_grouped_parameters,
                            lr=hp.plm_lr,
                            warmup=warmup_proportion,
                            t_total=num_train_optimization_steps)

        save_path = os.path.join(hp.checkdir, hp.name + '.pt')
        early_stopping = EarlyStopping(patience=hp.patience, verbose=True)
        avg_train_losses, avg_valid_losses = [], []
        for epoch in range(1, hp.ep+1):
            print(f"=========training at epoch={epoch}=========")
            
            train_loss = train(model, train_iter, optimizer, criterion, binary_criterion, device)
            avg_train_losses.append(train_loss.item())

            precision, recall, f1, valid_loss = eval(model, dev_iter, criterion, binary_criterion, device)
            avg_valid_losses.append(valid_loss.item())

            epoch_len = len(str(hp.ep))
            print_msg = (f'[{epoch:>{epoch_len}}/{hp.ep:>{epoch_len}}]     ' +
                        f'train_loss: {train_loss:.5f} ' +
                        f'valid_loss: {valid_loss:.5f}')
            print(print_msg)

            early_stopping(val_loss=-1*f1, model=model, filepath=save_path)

            if early_stopping.early_stop:
                print("Early stopping")
                break
        
    else:
        all_preds = test(model, test_iter, device)
        assert len(all_preds) == len(test_examples), ipdb.set_trace()

        pred_json = []
        for idx, (sent, example, pred) in enumerate(zip(test_dataset.sents, test_examples, all_preds)):
            sent = sent[1:-1]
            assert example['text'] == ' '.join(sent), ipdb.set_trace()
            example['labels'] = []

            pred = [idx2tag[0][hat] for hat in pred]

            curr_lbl, s, e = None, None, None
            for idx, (word, tag) in enumerate(zip(sent, pred)):
                if tag == 'O':
                    if curr_lbl:
                        lbl = {
                            'start': s,
                            'end': e,
                            'technique': curr_lbl,
                            'text_fragment': example['text'][s:e]
                        }
                        example['labels'].append(lbl)
                        curr_lbl, s, e = None, None, None
                else:
                    if curr_lbl is None:
                        curr_lbl = tag
                        s = len(" ".join(sent[0:idx+1])) - len(sent[idx])
                        e = len(" ".join(sent[0:idx+1]))
                    else:
                        if tag == curr_lbl:
                            e = len(" ".join(sent[0:idx+1]))
                        else:
                            lbl = {
                                'start': s,
                                'end': e,
                                'technique': curr_lbl,
                                'text_fragment': example['text'][s:e]
                            }
                            example['labels'].append(lbl)
                            
                            curr_lbl = tag
                            s = len(" ".join(sent[0:idx+1])) - len(sent[idx])
                            e = len(" ".join(sent[0:idx+1]))

            pred_json.append(example)
        
        save_path = os.path.join(hp.checkdir, hp.name + '.json')
        with open(save_path, "w") as fout:
            json.dump(pred_json, fout, indent=4, ensure_ascii=False)