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
import numpy as np


emb_dim = 768
num_labels = 8
# weights = torch.tensor([3,3,18,18,18,18,18,18])
criterion = nn.CrossEntropyLoss()
class ProbingClassifier(nn.Module):
    def __init__(
            self,
            input_dim,
            num_labels
    ):
        super(ProbingClassifier, self).__init__()
        self.input_dim = input_dim
        self.num_labels = num_labels
        self.mlp = nn.Linear(self.input_dim, self.num_labels, bias = True)

        # self.classifier = nn.Sequential(
        #     self.hidden,
        #     F.relu,
        #     self.out
        # )

    def forward(self, sentences, sent_logits, train = False):
        logits = self.mlp(sent_logits)
        return logits


def align_pred_to_word(sentences, pred_label_logits):
    preds_avg = []
    print(pred_label_logits.shape)
    print(sentences.shape)
    pred_label_logits = F.log_softmax(pred_label_logits, dim=-1)
    pred_label_logits = torch.squeeze(pred_label_logits)


    for sentence, logit in zip(sentences.to('mps'), pred_label_logits.to('mps')):
    #     print("0")
    #     # print('label shape')
    #     # print(lbl.shape)
    #     # print(lbl)
        sum = torch.zeros_like(logit[-1])
        total = 0
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        print(sentence)
        print(logit)
        words = tokenizer.convert_ids_to_tokens(sentence, dim=0)
        # print("word len")
        # print(words)
        pred_avg = []
        for i in range(0, len(words)):
            # print(words[i])
            if '##' in words[i]:
                sum += logit[i]
                total += 1
                # print(words[i])
            else:
                if total != 0:
                    pred_avg.append(sum/total)
                    # print(pred_avg)
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
        print(preds_avg)


        # print('a shape')
        # print(a.shape)


    preds_avg = nn.utils.rnn.pad_sequence(preds_avg, batch_first = True, padding_value=data.label2idx['<pad>'])
    # print(preds_avg.shape)
    # preds_avg = torch.stack(preds_avg, dim=0)
    return preds_avg
    # return 0

def train(train_loader, dev_loader, mode, layer, logdir, probe_path, device):
    model = BertModel.from_pretrained("bert-base-multilingual-cased", output_hidden_states=True)
    model.to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)
    best_dev_loss = None
    no_improvement = 0
    stop_early = False
    global_iter = 0
    utils.setup_log("train.log")
    logging.info("Training Layer {}".format(layer))
    best_f1 = None


    # probe = ProbingClassifier(emb_dim, num_labels)


    path = "WordLID/layer{}.pt".format(layer)
    probe = ProbingClassifier(emb_dim, num_labels).to(device)

    probe.to(device)

    for epoch in range(50):
        training_losses = []
        training_accuracies = []
        probe.train()
        if stop_early:
            break

        for sents, attention_mask, aligned_lbls, lbls in tqdm(train_loader):
            # print(sents.shape)
            # print(aligned_lbls.shape)
            # print(lbls.shape)
            optimizer.zero_grad()
            # print(batch)
            logits = model(sents.to(device), attention_mask = attention_mask.to(device))
            # print(logits[2][layer].shape)
            layer_logits = logits[2][layer]
            out = probe(sents, layer_logits, train=True)
            if out is None :
                print("Skipped batch")
            else:
                # print(out.shape)
                out_masked = out[attention_mask.bool()]  # Make sure to use .bool() if attention_mask is an integer tensor
                aligned_lbls_masked = aligned_lbls[attention_mask.bool()]

                # print(out.shape)
                # print(aligned_lbls.shape)

                # pred_label_logits = torch.flatten(out, start_dim=0, end_dim=2)
                # flat_trgt_labels = torch.flatten(aligned_lbls)
                pred_label_logits = out_masked.view(-1, out_masked.size(-1))  # Adjust shape for logits
                flat_trgt_labels = aligned_lbls_masked.view(-1) 
                loss = criterion(pred_label_logits, flat_trgt_labels)
                # predictions = torch.argmax(out, axis=-1)
                # print(predictions[attention_mask].shape)
                # print(predictions)
                # accuracy = (predictions == aligned_lbls).sum() / len(predictions)
                predictions = torch.argmax(pred_label_logits, dim=-1)

                # Calculate accuracy
                accuracy = (predictions == flat_trgt_labels).sum() / flat_trgt_labels.numel()  #
                # accuracy = (predictions == aligned_lbls_masked).sum() / aligned_lbls_masked.numel()  # Use numel for total count

                training_losses.append(loss.item())
                training_accuracies.append(accuracy.item())

                loss.backward()
                optimizer.step()

            # print(loss)
        training_loss_mean, training_acc_mean = np.mean(training_losses), np.mean(training_accuracies)
        tqdm.write('Layer={:>2} epoch={:,} Loss={:.4f} Acc={:.3f}'.format(
                    layer, epoch, training_loss_mean, training_acc_mean))
        logging.info('Layer={:>2} epoch={:,} Loss={:.4f} Acc={:.3f}'.format(
                    layer, epoch, training_loss_mean, training_acc_mean))
        
        metric = MulticlassF1Score(num_classes=9, average = 'weighted').to(device)
        
        dev_out = []
        dev_lbls = []

        validation_losses = []
        validation_accuracies = []

        probe.eval()
        for sents, attention_mask, aligned_lbls, lbls in tqdm(dev_loader):
            with torch.no_grad():
                logits = model(sents.to(device), attention_mask = attention_mask.to(device))
                # print(logits[2][layer].shape)
                layer_logits = logits[2][layer]
                out = probe(sents, layer_logits, train=False)
                #out: B x max_sent_len x lbl_len
                out_masked = out[attention_mask.bool()]
                aligned_lbls_masked = aligned_lbls[attention_mask.bool()]

                # pred_label_logits = torch.flatten(out, start_dim=0, end_dim=2)
                # flat_trgt_labels = torch.flatten(aligned_lbls)
                # loss = criterion(pred_label_logits, flat_trgt_labels)
                pred_label_logits = out_masked.view(-1, out_masked.size(-1))  # Adjust shape for logits
                flat_trgt_labels = aligned_lbls_masked.view(-1) 
                loss = criterion(pred_label_logits, flat_trgt_labels)

                # out, loss = probe(sents, layer_logits, train=False, labels = lbls.to(device))
                validation_losses.append(loss.item())
                # pred = torch.argmax(out, dim=-1)
                # accuracy = (pred == aligned_lbls).sum() / len(pred)
                predictions = torch.argmax(pred_label_logits, dim=-1)
                accuracy = (predictions == flat_trgt_labels).sum() / flat_trgt_labels.numel()
                validation_accuracies.append(accuracy.item())

                # dev_out.append(torch.flatten(torch.argmax(out, dim=-1)))
                # dev_lbls.append(torch.flatten(aligned_lbls))
                dev_out.append(predictions)
                dev_lbls.append(flat_trgt_labels)

        valid_loss_mean, valid_acc_mean = np.mean(validation_losses), np.mean(validation_accuracies)

        dev_out_t = nn.utils.rnn.pad_sequence(dev_out, batch_first = True)
        dev_lbl_t = nn.utils.rnn.pad_sequence(dev_lbls, batch_first = True)
        f1 = metric(dev_out_t.to(device), dev_lbl_t.to(device))
        tqdm.write('Layer={:>2} Epoch={} Valid Loss={:.4f} Acc={:.3f} F1:={:.3f}'.format(layer, epoch, valid_loss_mean, valid_acc_mean, f1))
        logging.info('Layer={:>2} Epoch={} Valid Loss={:.4f} Acc={:.3f} F1:={:.3f}'.format(layer, epoch, valid_loss_mean, valid_acc_mean, f1))
        scheduler.step(valid_loss_mean)
        
        if best_dev_loss is None or valid_loss_mean < best_dev_loss:
            no_improvement = 0
            best_dev_loss = valid_loss_mean

            if best_f1 is None or best_f1 < f1:
                best_f1 = f1
            out_path = "WordLID/layer{}.pt".format(layer)
            stats = 'dev_acc: {}, dev_f1: {}'.format(valid_acc_mean, f1)
            utils.save_checkpoint(probe=probe, optimizer=optimizer, epoch=epoch, stats=stats, path=out_path)
            # torch.save(probe.state_dict, outpath)
            # save model
        else:
            no_improvement +=1

        if no_improvement == 3:
            stop_early = True
            logging.info('Best F1 score: {}'.format(best_f1))
            # evaluate(test_sents, layer, device, probe)
            break


def evaluate(test_sentences_tok, test_sentences_attn, test_sentences, layer, device):

    model = BertModel.from_pretrained("bert-base-multilingual-cased", output_hidden_states=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model.to(device)
    # running_loss, running_iter = 0.0, 0
    # utils.setup_log("train.log")
    logging.info("Testing Layer {}".format(layer))
    path = "WordLID/layer{}.pt".format(layer)
    probe = ProbingClassifier(emb_dim, num_labels).to(device)
    if os.path.exists(path):
        checkpoint = torch.load(path)
        # print(checkpoint)
        probe.load_state_dict(checkpoint['probe_state'])
        probe.eval()
    else:
        return
    # probe = ProbingClassifier(emb_dim, num_labels)
    probe.to(device)
    # preds = []

    pred_file = open("WordLID/layer{}.txt".format(layer), 'w')

    for sent, attention_mask, test_sent in tqdm(zip(test_sentences_tok, test_sentences_attn, test_sentences), total=test_sentences_tok.size(dim=0)):
        # print(torch.unsqueeze(sents, 0))
        # print(sent.shape)
        probe.eval()
        with torch.no_grad():
            # print(sent)
            s = torch.unsqueeze(sent,0).to(device)
            a = torch.unsqueeze(attention_mask,0).to(device)
            logits = model(s, attention_mask = a)
            # logits = model(torch.unsqueeze(sent,0))
            # print(logits[2][layer].shape)
            layer_logits = logits[2][layer]
            out = probe(s, layer_logits)
            # print(out.shape)
            # [1, max_sent_lenn, lbls]
            out_masked = out[a.bool()]
            # print(out_masked.shape)
            # [token_len, lbls]
            pred_label_logits = out_masked.view(-1, out_masked.size(-1))  # Adjust shape for logits
            # print(pred_label_logits.shape)

        # preds.append(torch.argmax(out, dim=-1))

        #lbls: Bxmax_len

        # lbls = torch.argmax(out, dim=-1)
        preds_avg = []
        pred_label_logits = F.softmax(out, dim=-1)
        logit = torch.squeeze(pred_label_logits)


        # for sentence, logit in zip(sentences.to('mps'), pred_label_logits.to('mps')):
        #     print("0")
        #     # print('label shape')
        #     # print(lbl.shape)
        #     # print(lbl)
        sum = torch.zeros_like(logit[-1])
        total = 0
        
        # print(torch.unsqueeze(sent,1).shape)
        # print(pred_label_logits)
        words = tokenizer.convert_ids_to_tokens(torch.unsqueeze(sent,1))
            # print("word len")
        # print(words)
        pred_avg = []
        for i in range(0, len(words)):
            # print(words[i])
            if '##' in words[i]:
                sum += logit[i]
                total += 1
                # print(words[i])
            else:
                if total != 0:
                    pred_avg.append(sum/total)
                    # print(pred_avg)
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
        # preds_avg.append(a)
        # print('pred_avg')
        preds_avg = torch.argmax(a, dim=-1)
        # print(preds_avg)

        # print(preds_avg[0])
        # print(lbls.shape)
        # print(lbls)

        if(preds_avg.size(dim=0) != len(test_sent)+2):
            print("DIFF")
            print(preds_avg)
            print(preds_avg.size(dim=0))
            print(test_sent)
            print(len(test_sent))
            

        # for i in range(preds_avg.size(0)):
        for j in range(1, preds_avg.size(0)-1):
            l = preds_avg[j].item()
            p_lbl = data.idx2label.get(l)
            pred_file.write("%s\n" % p_lbl)
        pred_file.write('\n')
           
        # f1 = calculate_f1(dev_out, dev_lbls)
        
    pred_file.close()
    # write preds to file
    return
