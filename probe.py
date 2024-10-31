import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertConfig
from tqdm import tqdm
from torchmetrics.classification import MulticlassF1Score
import utils
import logging
import os
import data


emb_dim = 768
num_labels = 9
class ProbingClassifier(nn.Module):
    def __init__(
            self,
            input_dim,
            num_labels
    ):
        super(ProbingClassifier, self).__init__()
        self.input_dim = input_dim
        self.num_labels = num_labels
        self.mlp = nn.Linear(self.input_dim, self.num_labels, bias = False)

        # self.classifier = nn.Sequential(
        #     self.hidden,
        #     F.relu,
        #     self.out
        # )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, sentences, sent_logits, labels = None):
        logits = self.mlp(sent_logits)
        # print(logits.shape)
        # print('lbls shape')
        # print(labels.shape)
        # print(labels)
        logits_aligned = align_pred_to_word(sentences, logits, labels)
        pred_label_logits = torch.flatten(logits_aligned, start_dim=0, end_dim=1)
        if labels is None:
            flat_trgt_labels = None
        else:
            flat_trgt_labels = torch.flatten(labels)
            if pred_label_logits.size(dim=0) != flat_trgt_labels.size(dim=0):
                return None, None
        loss = self.loss(pred_label_logits, flat_trgt_labels) if flat_trgt_labels is not None else None
        return logits_aligned, loss


def calculate_f1(pred, lbls):
    f1 = 0
    for i in range(9):
        TP = 0
        FP = 0
        FN = 0
        for i in range(len(pred)):
            p = pred[i]
            l = pred[i]
            temp_true = [1 if j == i else 0 for j in l]
            temp_pred = [1 if j == i else 0 for j in p]
            for t, j in zip(temp_true, temp_pred):
                if t == 1 and j == 1:
                    TP +=1
                if t == 0 and j == 1:
                    FP += 1
                if t ==1  and j == 0:
                    FN += 1
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        f1 += (2 * precision * recall)/(precision+recall)
    f1 = f1/9

def align_pred_to_word(sentences, pred_label_logits, labels):
    preds_avg = []
    # print(pred_label_logits.shape)
    # print(sentences.shape)
    pred_label_logits = F.softmax(pred_label_logits, dim=-1)
    for sentence, logit, lbl in zip(sentences, pred_label_logits, labels):
        # print('label shape')
        # print(lbl.shape)
        # print(lbl)
        sum = torch.zeros_like(logit[-1])
        total = 0
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        words = tokenizer.convert_ids_to_tokens(sentence)
        # print("word len")
        # print(words)
        pred_avg = []
        for i in range(0, len(words)):
            if '##' in words[i]:
                sum += logit[i]
                total += 1
                # print(words[i])
            else:
                if total != 0:
                    pred_avg.append(sum/total)
                if words[i] == '[PAD]':
                    break
                    # print(sum/total)
                sum = 0
                total = 0
                sum += logit[i]
                total += 1
            if i == len(words)-1:
                pred_avg.append(sum/total)
                # print(total)
        a = torch.stack(pred_avg, dim=0)
        preds_avg.append(a)
        # print('a shape')
        # print(a.shape)


    preds_avg = nn.utils.rnn.pad_sequence(preds_avg, batch_first = True, padding_value=data.label2idx['<pad>'])
    # print(preds_avg.shape)
    # preds_avg = torch.stack(preds_avg, dim=0)
    return preds_avg

def train(train_loader, dev_loader, mode, layer, logdir, probe_path, device, test_sents):
    model = BertModel.from_pretrained("bert-base-multilingual-cased", output_hidden_states=True)
    model.to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=0)
    best_dev_loss = None
    no_improvement = 0
    stop_early = False
    global_iter = 0
    # running_loss, running_iter = 0.0, 0
    utils.setup_log("train.log")
    logging.info("Training Layer {}".format(layer))
    best_f1 = None


    probe = ProbingClassifier(emb_dim, num_labels)


    path = "WordLID/layer{}.pt".format(layer)
    probe = ProbingClassifier(emb_dim, num_labels).to(device)
    if os.path.exists(path):
        checkpoint = torch.load(path)
        # print(checkpoint)
        probe.load_state_dict(checkpoint['probe_state'])
        probe.eval()
    else:
        return

    probe.to(device)
    # evaluate(test_sents, layer, device, probe)

    # print(layer)

    for epoch in range(50):
        train_loss = 0.0
        train_iter = 0
        probe.train()
        if stop_early:
            break
        for sents, lbls in tqdm(train_loader):
            # print(sents.shape)
            # print(lbls.shape)
            optimizer.zero_grad()
            # print(batch)
            with torch.no_grad():
                logits = model(sents.to(device))
            # print(logits[2][layer].shape)
            layer_logits = logits[2][layer]
            out, loss = probe(sents, layer_logits, lbls.to(device))
            if out is None and loss is None:
                print("Skipped batch")
            else:
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_iter += len(lbls)

            # print(loss)
        curr_loss = train_loss/train_iter
        tqdm.write('Layer={:>2} epoch={:,} Loss={:.4f}'.format(
                    layer, epoch, curr_loss))
        logging.info('Layer={:>2} epoch={:,} Loss={:.4f}'.format(
                    layer, epoch, curr_loss))
        
        dev_loss, dev_correct, dev_iter = 0.0, 0,0
        metric = MulticlassF1Score(num_classes=9).to(device)
        
        dev_out = []
        dev_lbls = []
        probe.eval()
        for sents, lbls in tqdm(dev_loader):
            logits = model(sents.to(device))
            # print(logits[2][layer].shape)
            layer_logits = logits[2][layer]
            out, loss = probe(sents, layer_logits, lbls.to(device))
            dev_loss += loss.item()
            pred = torch.argmax(out, dim=-1)
            dev_correct += ( pred == lbls).sum().cpu().item()
            tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
            # for i in range(sents.size(dim=0)):
            #     words = tokenizer.convert_ids_to_tokens(sents[i])
            #     print(words)
            #     print("pred")
            #     print(pred[i])
            #     print("lbls")
            #     print(lbls[i])

            dev_iter += len(lbls)
            dev_out.append(torch.flatten(torch.argmax(out, dim=-1)))
            dev_lbls.append(torch.flatten(lbls))
        avg_dev_loss = dev_loss/dev_iter
        dev_acc = dev_correct/dev_iter
        # f1 = calculate_f1(dev_out, dev_lbls)
        dev_out_t = nn.utils.rnn.pad_sequence(dev_out, batch_first = True)
        dev_lbl_t = nn.utils.rnn.pad_sequence(dev_lbls, batch_first = True)
        f1 = metric(dev_out_t.to(device), dev_lbl_t.to(device))
        tqdm.write('Layer={:>2} Epoch={} Valid Loss={:.4f} Acc={:.3f} F1:={:.3f}'.format(layer, epoch, avg_dev_loss, dev_acc, f1))
        logging.info('Layer={:>2} Epoch={} Valid Loss={:.4f} Acc={:.3f} F1:={:.3f}'.format(layer, epoch, avg_dev_loss, dev_acc, f1))
        
        if best_dev_loss is None or dev_loss < best_dev_loss:
            no_improvement = 0
            best_dev_loss = dev_loss

            if best_f1 is None or best_f1 < f1:
                best_f1 = f1
            out_path = "WordLID/layer{}.pt".format(layer)
            stats = 'dev_acc: {}, dev_f1: {}'.format(dev_acc, f1)
            utils.save_checkpoint(probe=probe, optimizer=optimizer, epoch=epoch, stats=stats, path=out_path)
            # torch.save(probe.state_dict, outpath)
            # save model
        else:
            no_improvement +=1

        scheduler.step(avg_dev_loss)

        if no_improvement == 3:
            stop_early = True
            logging.info('Best F1 score: {}'.format(best_f1))
            # evaluate(test_sents, layer, device, probe)
            break


def evaluate(test_sents, layer, device, probe):

    model = BertModel.from_pretrained("bert-base-multilingual-cased", output_hidden_states=True)
    model.to(device)
    # running_loss, running_iter = 0.0, 0
    # utils.setup_log("train.log")
    logging.info("Testing Layer {}".format(layer))
    path = "WordLID/layer{}.pt".format(layer)
    probe = ProbingClassifier(emb_dim, num_labels).to(device)
    if os.path.exists(path):
        checkpoint = torch.load(path)
        print(checkpoint)
        probe.load_state_dict(checkpoint['probe_state'])
        probe.eval()
    else:
        return
    # probe = ProbingClassifier(emb_dim, num_labels)
    probe.to(device)
    # preds = []

    pred_file = open("WordLID/layer{}.txt".format(layer), 'w')

    for sents in tqdm(test_sents):
        # print(torch.unsqueeze(sents, 0))
        probe.eval()
        logits = model(**sents)
        # print(logits[2][layer].shape)
        layer_logits = logits[2][layer]
        out, _ = probe(layer_logits)
        # preds.append(torch.argmax(out, dim=-1))
        lbls = torch.flatten(torch.argmax(out, dim=-1))
        for i in range(1,lbls.size(0)-1): 
            l = lbls[i].item()
            p_lbl = data.idx2label.get(l)
            pred_file.write("%s\n" % p_lbl)
        pred_file.write('\n')
           
        # f1 = calculate_f1(dev_out, dev_lbls)

    # for pred in preds:
    #     print(pred)
    #     for p in pred:
    #         p_lbl = data.idx2label.get(p)
    #         pred_file.write("%s\n" % p_lbl)
    #     pred_file.write("\n")
    pred_file.close()
    # write preds to file
    return
