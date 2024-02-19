import torch
import numpy as np
from torch.utils.data import DataLoader
from genslm import GenSLM, SequenceDataset
from Bio import SeqIO  # ??Biopython??SeqIO??

# Load model
model = GenSLM("genslm_25M_patric", model_cache_dir="/scratch/sp96859/GenSLM")
model.eval()

# Select GPU device if it is available, else use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Read gene sequences from a fasta file using Biopython
sequences = []
fasta_file = "/scratch/sp96859/GenSLM/B-samples-train.fasta"  # ??fasta?????
for record in SeqIO.parse(fasta_file, "fasta"):
    sequences.append(str(record.seq))

# Rest of your code remains the same
dataset = SequenceDataset(sequences, model.seq_length, model.tokenizer)
dataloader = DataLoader(dataset)

embeddings = []
with torch.no_grad():
    for batch in dataloader:
        outputs = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            output_hidden_states=True,
        )
        emb = outputs.hidden_states[-1].detach().cpu().numpy()
        emb = np.mean(emb, axis=1)
        embeddings.append(emb)

embeddings = np.concatenate(embeddings)
print (embeddings)
print(embeddings.shape)
