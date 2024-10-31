from transformers import BertTokenizer, BertModel, BertConfig
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import nlp
import config
import re
import string

# device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# 8 labels: lang1, lang2, fw, mixed, unk, ambiguous, other and named entities (ne).
num_labels = 8
label2idx = {"<pad>":0, "lang1":1, "lang2":2, "fw":3, "mixed":4, "unk":5, "ambiguous":6, "other":7, "ne":8} #make this dynamic
idx2label = {0:'<pad>', 1:'lang1', 2:"lang2", 3:"fw", 4:"mixed", 5:"unk", 6:"ambiguous", 7:"other", 8:"ne"}
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

class LIDSentenceDataset(Dataset):
  def __init__(self, sentences, label):
    # assert len(sentences) == len(aligned_label)
    self.sentences = sentences
    # self.aligned_label = aligned_label
    self.label = label

  def __getitem__(self, i):
    return self.sentences[i], self.label[i]

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
      # word_s = word.translate(str.maketrans('', '',
      #                               string.punctuation))
      # if "\’" in word:
      #    word.replace("\’", '')
      
      # if(word == '\'ve'): 
      #    print(word)
      #    print(word_s)
      # if 'https' in word_s:
      #   sentence[i] = "[UNK]"
      # if 'http' in word_s:
      #   sentence[i] = "[UNK]"
      # if '@' in word:
      #   sentence[i] = word_s
      if word_s == "":
         sentence[i] = '[UNK]'
      else:
        sentence[i] = word_s

    # print(sentence)
      
    bert_tokenized_sentence = tokenizer(' '.join(sentence), max_length=512, truncation=True)
    # print(bert_tokenized_sentence)
    aligned_label = []
    current_word = ''
    index = 0 # index of current word in sentence and tagging
    for token in bert_tokenized_sentence:
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
            aligned_label.append(tagging[index])
        else:
           aligned_label.append("other")
        
    assert len(bert_tokenized_sentence) == len(aligned_label)

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
    aligned_label_list = [label2idx[l] for l in aligned_label]
    aligned_label_tensor = torch.tensor([0] + aligned_label_list + [0]).long()
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
#   print(len(sentences))
#   print(len(sentences[1]))
#   print(len(sentences[2]))
#   figure out why this is happening: sentence = [3][8289][sent len]
  for sentence in sentences[0]:
    # print(len(sentence))
    # print(len(sentence[0]))
    # print(sentences[1][0])
    # print(' '.join(sentence))
    bert_tokenized_sentence = tokenizer(' '.join(sentence), return_tensors='pt')
    # sentence_tensor = torch.tensor(tokenizer.convert_tokens_to_ids(['[CLS]'] + bert_tokenized_sentence + ['SEP'])).long()
    sentences_ids.append(bert_tokenized_sentence.to(device))
  return sentences_ids


  
def collate_fn(items):
    max_len = max(len(item[0]) for item in items)

#   sentences = torch.zeros((len(items), max_len), device=items[0][0].device).long().to(device)
#   labels = torch.zeros((len(items), max_len)).long().to(device)
    sentences = []
    aligned_labels=[]
    labels = []
    # print(items)
    for sentence, label in items:
    # sentences[i][0:len(sentence)] = sentence
    # labels[i][0:len(label)] = label
        # print(sentence)
        # print(label)
        sentences.append(torch.tensor(sentence['input_ids']).to(device))
        labels.append(label)
        # aligned_labels.append(aligned_label)
    # print(labels)
    # print(len(sentences))
    # print(len(labels))
    labels = nn.utils.rnn.pad_sequence(labels, batch_first = True, padding_value=label2idx['<pad>'])
    pad_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('[PAD]'))[0]


    # sentences = nn.utils.rnn.pad_sequence(sentences, batch_first = True, padding_value=tokenizer.token)
    sentences = nn.utils.rnn.pad_sequence(sentences, batch_first=True,padding_value=pad_token)
    return sentences,labels

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

    train_loader = DataLoader(LIDSentenceDataset(train_sentences_tok, train_labels_tok), batch_size=config.batch_size, collate_fn=collate_fn, shuffle=True)
    valid_loader = DataLoader(LIDSentenceDataset(valid_sentences_tok, valid_labels_tok), batch_size=config.batch_size, collate_fn=collate_fn)

    return train_loader, valid_loader, test_sentences_tok

