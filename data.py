from transformers import BertTokenizer, BertModel, BertConfig
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import nlp
import config
import re
import string
from emoji import UNICODE_EMOJI

# device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# 8 labels: lang1, lang2, fw, mixed, unk, ambiguous, other and named entities (ne).
num_labels = 8
label2idx = {"<pad>":0, "lang1":1, "lang2":2, "fw":3, "mixed":4, "unk":5, "ambiguous":6, "other":7, "ne":8} #make this dynamic
idx2label = {0:'<pad>', 1:'lang1', 2:"lang2", 3:"fw", 4:"mixed", 5:"unk", 6:"ambiguous", 7:"other", 8:"ne"}
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

class LIDSentenceDataset(Dataset):
  def __init__(self, sentences, aligned_label, label):
    # assert len(sentences) == len(aligned_label)
    self.sentences = sentences
    # if aligned_label is not None and label is not None:
    self.aligned_label = aligned_label
    self.label = label
    # else:
    #   self.aligned_label = None
    #   self.label = None

  def __getitem__(self, i):
    if self.aligned_label is not None and self.label is not None:
      return self.sentences[i], self.aligned_label[i], self.label[i]
    else: 
      return self.sentences[i]

  def __len__(self):
    return len(self.sentences)

def load_conllu(filename):
  f = open(filename)
  sentences = []
  sentence = []
  word_labels = []
  word_label = []
  sent_labels = []
  for line in f:
    if len(line)==0 or line.startswith('# sent_enum =') or line[0]=="\n":
      if len(sentence) > 0:
        sentences.append(sentence)
        word_labels.append(word_label)
        if "lang1" in word_label and "lang2" in word_label:
          sent_labels.append("cs")
        elif "lang1" in word_label and "lang2" not in word_label:
          sent_labels.append("eng")
        elif "lang1" not in word_label and "lang2" in word_label:
          sent_labels.append("es")
        else:
          sent_labels.append("other")
        sentence = []
        word_label = []
      continue
    splits = line.split()
    # print(splits)
    sentence.append(splits[0])
    word_label.append(splits[-1])
  return sentences, word_labels, sent_labels

def load_dataset():
    data = nlp.load_dataset('lince', 'lid_spaeng')
    # need to add individual sent labels
    # for split in data:
    #     print(split, len(data[split]))
    return data['train'], data['validation'], data['test']



def align_tokenizations(tokenizer, sentences, labels):
  bert_tokenized_sentences = []
  aligned_labels = []

  for sentence, tagging in zip(sentences, labels):
    # first generate BERT-tokenization
    # bert_tokenized_sentence = tokenizer.tokenize(' '.join(sentence))
    # s = sentence
    for i in range(len(sentence)):
      word = sentence[i]
      word_s = word.translate(str.maketrans('', '',
                                    string.punctuation))
      if word_s == "":
        sentence[i] = '[UNK]'
      elif word_s[0] in UNICODE_EMOJI['en']:
        sentence[i] = '[UNK]'
      else:
        sentence[i] = word_s

    # print(sentence)
      
    bert_tokenized_sentence = tokenizer(' '.join(sentence), max_length=512, truncation=True)
    # print(bert_tokenized_sentence)
    aligned_label = []
    current_word = ''
    index = -1 # index of current word in sentence and tagging
    tokenized_sent = tokenizer.convert_ids_to_tokens(bert_tokenized_sentence['input_ids'])
    for token in tokenized_sent:
        current_word += re.sub(r'^##', '', token) # recompose word with subtoken
    #   sentence[index] = sentence[index].replace('\xad', '') # fix bug in data

      # note that some word factors correspond to unknown words in BERT
    #   assert token == '[UNK]' or sentence[index].startswith(current_word)
        if(index < len(tagging)):
          if token == '[UNK]' or sentence[index] == current_word: # if we completed a word
            current_word = ''
            aligned_label.append(tagging[index])
            index += 1
          else:
            if token == '[CLS]' or token == '[SEP]':
              current_word = ''
              aligned_label.append("<pad>")
              index += 1
            else:
              aligned_label.append(tagging[index])
        else:
          if token == '[CLS]' or token == '[SEP]':
            current_word = ''
            aligned_label.append("<pad>")
            index += 1
    # print(tokenized_sent)
    # print(aligned_label)
    # if tokenized_sent != aligned_labels
    assert len(tokenized_sent) == len(aligned_label)
    bert_tokenized_sentences.append(bert_tokenized_sentence)
    aligned_labels.append(aligned_label)

  return bert_tokenized_sentences, aligned_labels

def convert_to_ids(aligned_labels, labels):
  aligned_label_ids = []
  labels_ids = []
  for aligned_label, label in zip(aligned_labels, labels):
    # sentence_tensor = torch.tensor(tokenizer.convert_tokens_to_ids(['[CLS]'] + sentence + ['[SEP]'])).long()
    label_list = [label2idx[l] for l in label]
    label_tensor = torch.tensor([0] + label_list + [0]).long()
    # print(aligned_label)
    aligned_label_list = [label2idx[l] for l in aligned_label]
    aligned_label_tensor = torch.tensor(aligned_label_list).long()
    # sentences_ids.append(sentence_tensor.to(device))
    labels_ids.append(label_tensor.to(device))
    aligned_label_ids.append(aligned_label_tensor.to(device))
  return aligned_label_ids, labels_ids

def tokenize_labeled_data( sentences, labels):
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    # model =  BertModel.from_pretrained("bert-base-multilingual-cased", output_hidden_states=True)
    sentence_ids, aligned_labels = align_tokenizations(tokenizer, sentences, labels)
    aligned_labels_ids, labels_ids = convert_to_ids(aligned_labels, labels)
    # print(sentence_ids[0])
    # print(aligned_labels_ids[0])
    # print(aligned_labels[0])
    # print(labels_ids[0])
    return sentence_ids, aligned_labels_ids, labels_ids
    # tokenizer.tokenize('This tokenizer is sooooo awesome.')
    # tokenizer.tokenize(' '.join(train_sentences[0]))

def tokenization(example):
    print(example['tokens'][0])
    # print(tokenizer(example['tokens']))
    return tokenizer(example["tokens"], max_length=512, truncation=True)

def tokenize_labeled_hfdf(df):
    print(df.column_names)
    # print(df["tokens"])
    encoded_dataset = df.map(tokenization, batched=True)
#   hfdf['tokenized'] = tokenizer.tokenize(' '.join(hfdf['tokens']))
#   hfdf['tokenized'] = hfdf['tokens'].apply(tokenizer)
    print(encoded_dataset[0])

def tokenize_unlabeled_data(sentences):
  tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
  sentences_ids = []
  # attention_masks = []
#   print(len(sentences))
#   print(len(sentences[1]))
#   print(len(sentences[2]))
#   figure out why this is happening: sentence = [3][8289][sent len]
  for sentence in sentences[0]:
    # print(len(sentence))
    # print(len(sentence[0]))
    # print(sentences[1][0])
    # print(' '.join(sentence))
    bert_tokenized_sentence = tokenizer(' '.join(sentence), max_length=512, truncation=True)
    # sentence_tensor = torch.tensor(tokenizer.convert_tokens_to_ids(['[CLS]'] + bert_tokenized_sentence + ['SEP'])).long()
    sentences_ids.append(bert_tokenized_sentence)
    # attention_masks.append(torch.tensor(bert_tokenized_sentence['attention_mask']))
  # pad_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('[PAD]'))[0]

  # sentences = nn.utils.rnn.pad_sequence(sentences_ids, batch_first=True,padding_value=pad_token)
  # attention_mask = nn.utils.rnn.pad_sequence(attention_masks, batch_first=True,padding_value=0)
  return sentences_ids


  
def collate_fn(items):
    max_len = max(len(item[0]) for item in items)

#   sentences = torch.zeros((len(items), max_len), device=items[0][0].device).long().to(device)
#   labels = torch.zeros((len(items), max_len)).long().to(device)
    sentences = []
    attention_mask = []
    aligned_labels=[]
    labels = []
    # print(items)
    for sentence, aligned_label, label in items:
    # sentences[i][0:len(sentence)] = sentence
    # labels[i][0:len(label)] = label
        # print(sentence)
        # print(label)
        sentences.append(torch.tensor(sentence['input_ids']).to(device))
        attention_mask.append(torch.tensor(sentence['attention_mask']))
        labels.append(label)
        aligned_labels.append(aligned_label)
        # aligned_labels.append(aligned_label)
    # print(labels)
    # print(len(sentences))
    # print(len(labels))
    labels = nn.utils.rnn.pad_sequence(labels, batch_first = True, padding_value=label2idx['<pad>'])
    aligned_labels = nn.utils.rnn.pad_sequence(aligned_labels, batch_first=True, padding_value=label2idx['<pad>'])
    pad_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('[PAD]'))[0]


    # sentences = nn.utils.rnn.pad_sequence(sentences, batch_first = True, padding_value=tokenizer.token)
    sentences = nn.utils.rnn.pad_sequence(sentences, batch_first=True,padding_value=pad_token)
    attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True,padding_value=0)
    return sentences, attention_mask, aligned_labels,labels

def collate_fn_unlabeled(items):
    # max_len = max(len(item[0]) for item in items)

#   sentences = torch.zeros((len(items), max_len), device=items[0][0].device).long().to(device)
#   labels = torch.zeros((len(items), max_len)).long().to(device)
    sentences = []
    attention_mask = []
    # print(items)
    for sentence in items:
    # sentences[i][0:len(sentence)] = sentence
    # labels[i][0:len(label)] = label
        # print(sentence)
        # print(label)
        sentences.append(torch.tensor(sentence['input_ids']).to(device))
        attention_mask.append(torch.tensor(sentence['attention_mask']))
        # labels.append(label)
        # aligned_labels.append(aligned_label)
        # aligned_labels.append(aligned_label)
    # print(labels)
    # print(len(sentences))
    # print(len(labels))
    # labels = nn.utils.rnn.pad_sequence(labels, batch_first = True, padding_value=label2idx['<pad>'])
    # aligned_labels = nn.utils.rnn.pad_sequence(aligned_labels, batch_first=True, padding_value=label2idx['<pad>'])
    pad_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('[PAD]'))[0]


    # sentences = nn.utils.rnn.pad_sequence(sentences, batch_first = True, padding_value=tokenizer.token)
    sentences = nn.utils.rnn.pad_sequence(sentences, batch_first=True,padding_value=pad_token)
    attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True,padding_value=0)
    return sentences, attention_mask


# def get_embeddings(model, sentences):
#     tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
#     for sent in sentences:
#         sent = tokenizer.convert_tokens_to_ids(sent)
#         # print(sent)
#     return sentences

def convert_ids_to_tokens(sent_ids, lbls):
    assert len(sent_ids) == len(lbls)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    for sent, lbl in zip(sent_ids, lbls):
        s =  tokenizer.convert_ids_to_tokens(sent_ids)

def load_data(train_file, valid_file, test_file):

    train_sentences, train_labels, train_sent_labels = load_conllu(train_file)
    # train, val, test = load_dataset()
    train_sentences_tok, train_aligned_labels_tok, train_labels_tok = tokenize_labeled_data(train_sentences, train_labels)
    # print(train_sentences_tok)
    # train_sentences_emb = get_embeddings(None, train_sentences_tok)
    # print(train_sentences_emb)
    valid_sentences, valid_labels, valid_sent_labels = load_conllu(valid_file)
    valid_sentences_tok, valid_aligned_labels_tok, valid_labels_tok = tokenize_labeled_data(valid_sentences, valid_labels)
    # valid_sentences_emb = get_embeddings(None, valid_sentences_tok)


    # ignore test taggings
    test_sentences = load_conllu(test_file)
    test_sentences_tok = tokenize_unlabeled_data(test_sentences)
    # print(len(train_sentences))
    # print(len(train_labels))
    # print(len(train_sent_labels))
    # print(train_sentences[0])
    # print(train_labels[0])
    # print(train_sent_labels[0])

    train_loader = DataLoader(LIDSentenceDataset(train_sentences_tok, train_aligned_labels_tok, train_labels_tok), batch_size=config.batch_size, collate_fn=collate_fn, shuffle=True)
    valid_loader = DataLoader(LIDSentenceDataset(valid_sentences_tok, valid_aligned_labels_tok, valid_labels_tok), batch_size=config.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(LIDSentenceDataset(test_sentences_tok, None, None), batch_size=12, collate_fn=collate_fn_unlabeled)

    return train_loader, valid_loader, test_loader

