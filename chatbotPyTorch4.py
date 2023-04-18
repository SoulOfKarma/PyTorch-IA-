import torch
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Carga del modelo previamente entrenado
model_path = 'chatbot_model4.pth'
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Configuración del tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

model.resize_token_embeddings(len(tokenizer))
model.load_state_dict(torch.load(model_path), strict=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definición de la función para generar respuesta del modelo
def generate_response(prompt, model, tokenizer):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)
    response_ids = model.generate(input_ids=input_ids, max_length=500, num_return_sequences=1, no_repeat_ngram_size=2)
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return response

# Carga del archivo JSON con los datos de entrenamiento
with open('jsonDatas7.json') as f:
    data = json.load(f)

# Preparación de los datos
all_text = []
all_responses = []
for entry in data['data']:
    all_text.append(entry['text'])
    all_responses.append(entry['response'])

text_set = list(set(all_text))
response_set = list(set(all_responses))

# Bucle para conversar con el bot
while True:
    #prompt = input("You: ") + " "
    prompt = input("You: ") + " ."
    response = generate_response(prompt, model, tokenizer)
    print(f"Ophelia: {response}")
