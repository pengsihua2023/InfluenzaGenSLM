# Installation  
ml Miniconda3/4.10.3  
conda create -n GenSLM python=3.10  
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia  
pip install git+https://github.com/ramanathanlab/genslm   
# Download the pre-trained models
The pre-trained models and datasets can be downloaded from this [Globus Endpoint](https://app.globus.org/file-manager?origin_id=25918ad0-2a4e-4f37-bcfc-8183b19c3150&origin_path=%2F&two_pane=false).  
# Embedding code with fasta file as DNA sequence Data  
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
fasta_file = "/scratch/sp96859/GenSLM/B-samples-train.fasta"  # Specify the path to the fasta file
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

# Convert NumPy array to PyTorch tensor  
embeddings_tensor = torch.from_numpy(embeddings)

# Save the tensor to a file, specify the file name and path here  
torch.save(embeddings_tensor, '/scratch/sp96859/GenSLM/embeddings.pt')

# When loading this .pt file, you can use the torch.load method directly. For example:
# loaded_embeddings = torch.load('/scratch/sp96859/GenSLM/embeddings.pt')

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
# I submitted the job using sub_GPU_GenSLM-embedding.sh in Sapelo2 cluster  
```  

#!/bin/bash
#SBATCH --job-name=VirusBERT              # Job name
#SBATCH --partition=gpu_p             # Partition (queue) name, i.e., gpu_p 
#SBATCH --gres=gpu:A100:1             # Requests one GPU device 
#SBATCH --ntasks=1                    # Run a single task       
#SBATCH --cpus-per-task=2             # Number of CPU cores per task
#SBATCH --mem=200gb                    # Job memory request
#SBATCH --time=10:00:00               # Time limit hrs:min:sec
#SBATCH --output=Embedding-GenSLM.%j.out         # Standard output log
#SBATCH --error=Embedding-GenSLM.%j.err          # Standard error log
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sp96859@uga.edu  # Where to send mail       

cd $SLURM_SUBMIT_DIR

#python sihua-DNA-BERT-unsupervize-learning-more-GPU-Segmentembedding.py
#ml Biopython/1.78-foss-2020b
#ml scikit-learn/1.0.1-foss-2021b
#python Anomaly_Detection_OneHot-train-test-V9-Final-with-GPU.py
#python Anomaly_Detection_OneHot-train-test-V9-Final.py
python Virus-embeddings-with-GenSLM-fasta.py
#python two-sequenc-embedding.py
#python sachin.py
```  
