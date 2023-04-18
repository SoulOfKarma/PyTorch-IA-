import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class Chatbot(nn.Module):
    def __init__(self, model):
        super(Chatbot, self).__init__()
        self.model = model
    
    def forward(self, input_ids):
        outputs = self.model(input_ids=input_ids, return_dict=True)
        logits = outputs.logits
        return logits

# Lectura del archivo JSON
with open('jsonDatas8.json') as f:
    data = json.load(f)

# Preparación de los datos
all_text = []
all_responses = []
for entry in data['data']:
    all_text.append(entry['text'])
    all_responses.append(entry['response'])
    
text_set = list(set(all_text))
response_set = list(set(all_responses))

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Agregamos nuestro tokenizer al modelo
model.resize_token_embeddings(len(tokenizer))

# Creación del modelo y envío a la GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Chatbot(model).to(device)

# Función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

# Entrenamiento del modelo
n_epochs = 100
batch_size = 8

n_batches = len(text_set) // batch_size

for epoch in range(n_epochs):
    epoch_loss = 0
    for i in range(0, len(text_set), batch_size):
        batch_text = all_text[i:i+batch_size]
        batch_responses = all_responses[i:i+batch_size]
        batch_size = len(batch_text)  # Obtener el tamaño real del lote
        inputs = torch.zeros(batch_size, dtype=torch.long).to(device)
        targets = torch.zeros(batch_size, dtype=torch.long).to(device)
        for j in range(batch_size):
            inputs[j] = text_set.index(batch_text[j])
            targets[j] = response_set.index(batch_responses[j])
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Impresión del progreso
        if (i//batch_size) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} Batch {i//batch_size+1}/{n_batches} Loss: {loss.item():.4f}")

    epoch_loss /= n_batches
    print(f"Epoch {epoch+1}/{n_epochs} Loss: {epoch_loss:.4f}")

# Guardado del modelo
torch.save(model.state_dict(), 'chatbot_model4.pth')
