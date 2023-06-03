import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

torch.cuda.empty_cache()

# model and tokenizer
model_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
model, tokenizer = AutoModelForSequenceClassification.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name)
# question, answer = "Explain nuclear fusion like I am five", "Nuclear fusion is the process by which two or more protons and neutrons combine to form a single nucleus. It is a very important process in the universe, as it is the source of energy for stars and galaxies. Nuclear fusion is also a key process in the production of energy for nuclear power plants."
# inputs = tokenizer(question, answer, return_tensors='pt')
# score = model(**inputs).logits[0].cpu().detach()
# print(score)


# dataset 
class OasstDataset(Dataset):
    def __init__(self, filename, tokenizer):
        self.data = pd.read_csv(filename)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text1, text2, label = row['prompt_text'], row['answer_text'], row['quality']
        inputs = self.tokenizer(text1, text2, padding='max_length', max_length=512, truncation=True)
        inputs = {key: torch.tensor(val) for key, val in inputs.items()}  # Convert lists to tensors
        inputs['labels'] = torch.tensor(label).unsqueeze(0).float()
        return inputs

# Create Dataloader
train_dataset = OasstDataset('sample_dataset.csv', tokenizer)
eval_dataset = OasstDataset('sample_dataset.csv', tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=True)

# Config
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
progress_bar = tqdm(range(num_training_steps))
# Create a summary writer
writer = SummaryWriter()


# Training
model.train()
for epoch in range(num_epochs):
    for i, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

        # Write loss to TensorBoard
        if i % 50 == 0:  # every 50 batch
            writer.add_scalar('Loss/train', loss, epoch*len(train_dataloader)+i)

# Evaluating
metric = evaluate.load("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

    # Write accuracy to TensorBoard
    if i % 50 == 0:  # every 50 batch
        writer.add_scalar('Accuracy/val', metric.compute(), epoch*len(eval_dataloader)+i)

writer.close()