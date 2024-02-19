# Installation  
ml Miniconda3/4.10.3  
conda create -n GenSLM python=3.10  
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia  
pip install git+https://github.com/ramanathanlab/genslm   
# Download the pre-trained models
The pre-trained models and datasets can be downloaded from this [Globus Endpoint](https://app.globus.org/file-manager?origin_id=25918ad0-2a4e-4f37-bcfc-8183b19c3150&origin_path=%2F&two_pane=false).  
# embedding code with fasta file as DNA sequence Data  
```python  
import torch
import numpy as np
from torch.utils.data import DataLoader
from genslm import GenSLM, SequenceDataset
from Bio import SeqIO 

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
```  
# Generate synthetic sequences with fasta file as DNA sequence Data  
```python  

import torch
from genslm import GenSLM

# Load model
model = GenSLM("genslm_25M_patric", model_cache_dir="/scratch/sp96859/GenSLM")
model.eval()

# Select GPU device if it is available, else use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Prompt the language model with a start codon
prompt = model.tokenizer.encode("ATG", return_tensors="pt").to(device)

tokens = model.model.generate(
    prompt,
    max_length=2048,  # Increase this to generate longer sequencespython 
    min_length=2048,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    num_return_sequences=1,  # Change the number of sequences to generate
    remove_invalid_values=True,
    use_cache=True,
    pad_token_id=model.tokenizer.encode("[PAD]")[0],
    temperature=1.0,
)

sequences = model.tokenizer.batch_decode(tokens, skip_special_tokens=True)

for sequence in sequences:
    print(sequence)

```  
