from transformers import BertTokenizer, BertConfig, BertModel
import torch.nn as nn
import torch.nn.functional as F
import torch


class BERTClassifierModel(nn.Module):

    def __init__(self, config, num_classes=6, hidden_size=768):
        super(BERTClassifierModel, self).__init__()
        self.number_of_classes = num_classes
        self.dropout = nn.Dropout(0.01)
        self.hidden_size = hidden_size
        self.bert = BertModel.from_pretrained('model/pytorch_model.bin', config=config)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs):
        _, embedding = self.bert(inputs[0], token_type_ids=None, attention_mask=inputs[1])
        output = self.classifier(self.dropout(embedding))
        return F.sigmoid(output)


def prediction(sentence):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = BertConfig.from_json_file('model/bert_config.json')
    model = BERTClassifierModel(config).to(device)

    model.load_state_dict(torch.load('model/model_bert.bin'))
    tokenizer = BertTokenizer('model/vocab.txt')

    input_id = torch.tensor(
        tokenizer.encode(sentence, add_special_tokens=True, truncation=True, max_length=256, pad_to_max_length=True))
    attention_mask = torch.tensor([float(i > 0) for i in input_id])

    input_id = input_id.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)

    result = model((input_id, attention_mask)).detach()

    return result
