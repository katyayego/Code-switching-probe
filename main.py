import logging
import utils
import models
import data
import torch
import os
import probe

train_path = "lid_spaeng/train.conll"
test_path = 'lid_spaeng/test.conll'
valid_path = 'lid_spaeng/dev.conll'
# BERT-Base, Multilingual Cased (New, recommended): 104 languages, 12-layer, 768-hidden, 12-heads, 110M parameters
layer_num = 12

def run_layer(train_loader, dev_loader, test_sentences, mode, layer, device):
    if mode == 'train':
        print('\nstart training layer {} of {} with task {}'.format(
            layer, model_name, task_name))
        logdir = ''
        probe_path = ''
        probe.train(train_loader, dev_loader, mode, layer, logdir, probe_path + '.tmp', device, test_sentences)
    elif mode == 'test':
        print('\nstart evaluating layer {} of {} with task {}'.format(
            layer, model_name, task_name))
        summary, labels, preds = probe.test(mode, layer, probe_path, test_sentences)
        return labels, preds
    return None, None

def run_probe(mode = "train", device="cpu"):
    
    train_loader, valid_loader, test_sentences = data.load_data(train_file=train_path, valid_file=valid_path, test_file=test_path)
        # logging.info(train_loader.__len__)
    logging.info("Loaded data...")
    # train_features, train_labels = next(iter(train_loader))
    # print(train_features.shape)
    # print(train_labels.shape)
    layers_labels = []
    for layer in range(1):
        layer_labels, layer_preds = run_layer(train_loader, valid_loader, test_sentences, mode, layer, device)
        if layer_labels is not None:
            if len(layers_labels) == 0:
                layers_labels.append(layer_labels)
            layers_labels.append(layer_preds)

    if len(layers_labels) > 0:
        preds_dir = os.path.join("lid_spaeng", mode, task_name, 'predictions')
        # preds_path = data_path(preds_dir, mode, 'json', model_name)

        with open(preds_dir, 'w') as f:
            for labels in zip(*layers_labels):
                labels = [data.idx2label.get(lab) for lab in labels]
                f.write('\t'.join(labels) + '\n')

        print(f'Saved layer-wise predictions to {preds_dir}')

    # if config.report:
    # summaries = []

    # for layer in range(layer_num):
    #     summary_dir = os.path.join(task_name, 'summaries', config.name, mode)
    #     summary_path = data_path(summary_dir, mode, 'json', model_name, layer)

    #     if not os.path.exists(summary_path):
    #         print('skipping, {} does not exist'.format(summary_path))
    #     else:
    #         with open(summary_path) as f:
    #             summary = json.load(f)
    # #         summaries.append((layer, summary))

    # #     report(summaries)

    # if run_type == "train":
    #     return
    # elif run_type == "eval":
    #     return 
def run_evaluation(device):
    train_loader, valid_loader, test_sentences = data.load_data(train_file=train_path, valid_file=valid_path, test_file=test_path)
    for layer in range(layer_num):
        # break
        probe.evaluate(test_sentences, layer, device)

if __name__ == "__main__":
    utils.setup_log("main.log")
    logging.info("Runnning...")
    

    task_name ="WLID"
    model_name = 'mBERT'
    # print('CUDA available' if torch.cuda.is_available() else 'Running on CPU')
    print('MPS availiable' if torch.backends.mps.is_available() else 'Running on CPU')
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print('Starting task {} with model {}'.format(task_name, model_name))
    run_probe(mode="train", device = device)
    run_evaluation(device=device)

    # trainer = models.ProbeTrainer()
    # bert = models.Bert(0, 5)
    # bert.get_embeddings("i like chicken")