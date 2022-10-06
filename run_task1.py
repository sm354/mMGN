import argparse
import os
import json
import numpy as np
import sklearn
import ipdb
from tqdm import tqdm
import torch
import torch.nn as nn
import transformers
from transformers import AutoModel, AutoTokenizer
import pytorch_lightning as pl
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--checkdir", type=str, default="checkpoints")
    parser.add_argument("--trainset", type=str, default='../propaganda-detection/data/task1_train.json')
    parser.add_argument("--devset", type=str, default='../propaganda-detection/data/task1_dev.json')
    parser.add_argument("--testset", type=str, default='../propaganda-detection/data/task1_dev_test.json')
    parser.add_argument("--techniques", type=str, default='../propaganda-detection/techniques_list_task1-2.txt')
    parser.add_argument("--plm", type=str, default='xlm-roberta-large')
    parser.add_argument("--name", type=str, default='debug')
    parser.add_argument("--weights", default=None)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--plm_lr", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ep", type=int, default=20)
    parser.add_argument("--gpu", default=False)
    parser.add_argument("--seed", type=int, default=2022)
    args = parser.parse_args()
    return args

def get_xy(data, num_techs, tech2idx, idx2tech, test=False):
    if test:
        x = []
        for i, example in enumerate(data):
            id, text = example['id'], example['text']            
            x.append(text)
        return x, None

    x, y = [], []
    for i, example in enumerate(data):
        id, text, labels = example['id'], example['text'], example['labels']
        assert len(labels)!=0, ipdb.set_trace()
        
        x.append(text)
        y_i = np.zeros(num_techs)
        for label in labels:
            if label == "no technique":
                continue
            y_i[tech2idx[label]]=1
        y_i = y_i.tolist()
        y.append(y_i)
    return x, y

class myDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, tokenizer, test=False):
        super(myDataset, self).__init__()
        self.test = test
        self.x_tok = tokenizer(
            x,
            max_length=256, 
            padding="max_length", # truncation=True, 
            return_tensors='pt'
        )
        if not self.test:
            self.y = torch.tensor(y)
            assert len(self.x_tok['input_ids'])==len(self.y), ipdb.set_trace()
    
    def __getitem__(self, idx):
        if self.test:
            return self.x_tok['input_ids'][idx], self.x_tok['attention_mask'][idx]
        else:
            return self.x_tok['input_ids'][idx], self.x_tok['attention_mask'][idx], self.y[idx]
    
    def __len__(self):
        return len(self.x_tok['input_ids'])

class myModel(nn.Module):
    def __init__(self, plm, hidden_dim=768, dropout=0.3, n_classes=20):
        super(myModel, self).__init__()
        self.n_classes = n_classes
        self.plm = plm
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, inp):
        out = self.plm(**inp)
        sequence_output, pooled_output = out[0], out[1]
        logits = self.classifier(pooled_output) # [bs, n_classes]
        preds = (logits>=0).long()
        return logits, preds

def train(model, train_loader, device, criterion, optimizers, schedulers, ep):
    model.train()
    train_loss, y_true, y_pred = [], [], []
    for batch_idx, (inp_id, attn, y) in enumerate(train_loader):
        batch = {
            "input_ids": inp_id.to(device),
            "attention_mask": attn.to(device),
        }
        y = y.to(device)
        
        for optim in optimizers.values():
            optim.zero_grad()

        logits, preds = model(batch)
        loss = criterion(logits, y.float())
    
        loss.backward()
        for optim in optimizers.values():
            optim.step()
        for scheduler in schedulers.values():
            scheduler.step()

        y = y.cpu().detach().numpy().tolist()
        preds = preds.cpu().detach().tolist()
        y_true.extend(y)
        y_pred.extend(preds)

        train_loss.append(loss.item())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision = precision_score(y_true=y_true, y_pred=y_pred, labels=None, average="micro")
        recall = recall_score(y_true=y_true, y_pred=y_pred, labels=None, average="micro")
        f1 = f1_score(y_true=y_true, y_pred=y_pred, labels=None, average="micro")
    
    return np.mean(train_loss), precision, recall, f1

def eval(model, dev_loader, device, criterion, ep):
    model.eval() 
    dev_loss, y_true, y_pred = [], [], []
    with torch.no_grad():
        for batch_idx, (inp_id, attn, y) in enumerate(dev_loader):
            batch = {
                "input_ids": inp_id.to(device),
                "attention_mask": attn.to(device),
            }
            y = y.to(device)

            logits, preds = model(batch)
            loss = criterion(logits, y.float())

            y = y.cpu().detach().numpy().tolist()
            preds = preds.cpu().detach().tolist()
            y_true.extend(y)
            y_pred.extend(preds)

            dev_loss.append(loss.item())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision = precision_score(y_true=y_true, y_pred=y_pred, labels=None, average="micro")
        recall = recall_score(y_true=y_true, y_pred=y_pred, labels=None, average="micro")
        f1 = f1_score(y_true=y_true, y_pred=y_pred, labels=None, average="micro")

    return np.mean(dev_loss), precision, recall, f1

def test(model, test_loader, device):
    model.eval() 
    y_pred = []
    
    with torch.no_grad():
        for batch_idx, (inp_id, attn) in enumerate(test_loader):
            batch = {
                "input_ids": inp_id.to(device),
                "attention_mask": attn.to(device),
            }
            logits, preds = model(batch)
            preds = preds.cpu().detach().tolist()
            y_pred.extend(preds)
    return y_pred

if __name__ == "__main__":
    args = get_arg_parser()
    pl.seed_everything(args.seed)
    device = "cuda:0" if args.gpu else "cpu"
    num_epochs = args.ep
    batch_size = args.bs

    if not os.path.exists(args.checkdir):
        os.mkdir(args.checkdir)
    
    with open(args.techniques, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    all_techniques = [line.strip() for line in lines if len(line.strip())!=0]
    all_techniques.remove("no technique")
    num_techs = len(all_techniques)
    print("%d techniques found in %s"%(num_techs, args.techniques))
    
    tech2idx, idx2tech = {}, {}
    for idx, tech in enumerate(all_techniques):
        tech2idx[tech] = idx
        idx2tech[idx] = tech

    # model
    plm = AutoModel.from_pretrained(args.plm)
    plm_tokenizer = AutoTokenizer.from_pretrained(args.plm)
    model = myModel(plm=plm, hidden_dim=plm.config.hidden_size, dropout=0.3, n_classes=num_techs).to(device)

    # dataset
    kwargs= {}

    if args.train:
        with open(args.trainset, 'r') as f:
            train_data = json.load(f)
        with open(args.devset, 'r') as f:
            dev_data = json.load(f)
        x_train, y_train = get_xy(train_data, num_techs, tech2idx, idx2tech)
        x_dev, y_dev = get_xy(dev_data, num_techs, tech2idx, idx2tech)
        
        trainset, devset = myDataset(x_train, y_train, plm_tokenizer), myDataset(x_dev, y_dev, plm_tokenizer)
        
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
        dev_loader = torch.utils.data.DataLoader(devset, batch_size=batch_size, shuffle=False, **kwargs)
        
        print("# examples in train = %d and in dev = %d"%(len(x_train), len(x_dev)))
    else:
        with open(args.testset, 'r') as f:
            test_data = json.load(f)
        x_test, y_test = get_xy(test_data, num_techs, tech2idx, idx2tech, test=True)
        testset = myDataset(x_test, y_test, plm_tokenizer, test=True)
        test_loader=torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, **kwargs)
        print("# examples in test = %d"%(len(x_test)))

    if args.weights:
        print("loading model from %s"%(args.weights))
        model.load_state_dict(torch.load(args.weights, map_location=device))
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([158/20]).to(device))

    if args.train:
        optimizers, schedulers = {}, {}
        optimizers["plm_optimizer"] = torch.optim.Adam(
            model.plm.parameters(), lr=args.plm_lr
        )
        schedulers["plm_scheduler"] = transformers.get_linear_schedule_with_warmup(
            optimizers["plm_optimizer"],
            0, len(x_train) * num_epochs,
            # len(x_train), len(x_train) * num_epochs
        )

        optimizers["general_optimizer"] = torch.optim.Adam(
            model.classifier.parameters(), lr=args.lr
        )
        schedulers["general_scheduler"] = transformers.get_linear_schedule_with_warmup(
            optimizers["general_optimizer"],
            0, len(x_train) * num_epochs
        )

        best_f1 = 0
        for ep in range(num_epochs):
            train_l, train_p, train_r, train_f = train(model, train_loader, device, criterion, optimizers, schedulers, ep)
            dev_l, dev_p, dev_r, dev_f = eval(model, dev_loader, device, criterion, ep)

            if True: # ep%5==0:
                print('epoch:%d (loss, precision, recall, f1) train=(%.2f, %.2f, %.2f, %.2f) dev=(%.2f, %.2f, %.2f, %.2f)'\
                    %(ep, train_l, train_p, train_r, train_f, dev_l, dev_p, dev_r, dev_f))
            
            if dev_f > best_f1:
                best_f1 = dev_f
                print(f"saving model") #  at {best_f1:.5f} f1")
                torch.save(model.state_dict(), os.path.join(args.checkdir, args.name+'.pt'))

    else:
        test_pred = test(model, test_loader, device)
        assert len(test_pred)==len(test_data), ipdb.set_trace()
        pred_json = []
        for (example, pred) in zip(test_data, test_pred):
            pred = [idx2tech[idx] for idx, lbl in enumerate(pred) if lbl==1]
            example['labels'] = pred if len(pred) != 0 else ["no technique"]
            pred_json.append(example)
        
        with open(os.path.join(args.checkdir, args.name + '.json'), "w") as fout:
            json.dump(pred_json, fout, indent=4, ensure_ascii=False)
