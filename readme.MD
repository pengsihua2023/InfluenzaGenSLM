# Installation  
pip install git+https://github.com/ramanathanlab/genslm   
The pre-trained models and datasets can be downloaded from this [Globus Endpoint](https://app.globus.org/file-manager?origin_id=25918ad0-2a4e-4f37-bcfc-8183b19c3150&origin_path=%2F&two_pane=false).  
''' python  
import torch
import numpy as np
from torch.utils.data import DataLoader
from genslm import GenSLM, SequenceDataset
from Bio import SeqIO  # 导入Biopython中的SeqIO模块

# Load model
model = GenSLM("genslm_25M_patric", model_cache_dir="/scratch/sp96859/GenSLM")
model.eval()

# Select GPU device if it is available, else use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Read gene sequences from a fasta file using Biopython
sequences = []
fasta_file = "/scratch/sp96859/GenSLM/B-samples-train.fasta"  # 指定fasta文件的路径
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
'''
