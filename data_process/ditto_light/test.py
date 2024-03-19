# from transformers import DistilBertTokenizer, DistilBertModel

# model = DistilBertModel.from_pretrained("distilbert-base-uncased")
# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)
# print(output)
# print('finish')


import torch
import os


# os.environ["CUDA_VISIBLE_DEVICES"] = '2,1'

print(torch.cuda.is_available())
print(torch.cuda.device_count())

print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_name(1))

print(torch.cuda.get_device_name(2))
