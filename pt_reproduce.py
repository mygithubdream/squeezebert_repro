from transformers import SqueezeBertTokenizer, SqueezeBertModel
tokenizer = SqueezeBertTokenizer.from_pretrained('squeezebert/squeezebert-uncased')
model = SqueezeBertModel.from_pretrained("squeezebert/squeezebert-uncased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print(output['last_hidden_state'])
print("end")
