from transformers import BertTokenizer, BertModel, BertConfig
import config
import logging
import torch

class Bert:
    def __init__(self, layer, hidden_layer):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.model =  BertModel.from_pretrained("bert-base-multilingual-cased", output_hidden_states=True)
        self.layer = layer
        self.hidden_layer_size = hidden_layer

    def get_embeddings(self, text):
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        logging.info(tokenized_text)
        logging.info(indexed_tokens)
        tokens_t = torch.tensor([indexed_tokens])

        with torch.no_grad():
            logits = self.model(tokens_t)
            # logging.info(logits)
            layer_logits = logits[2][self.layer]
            # logging.info(layer_logits.shape)
        return layer_logits