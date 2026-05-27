import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

MODEL_NAME = "zhihan1996/DNA_bert_6"

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# --- MOVE THESE OUTSIDE THE FUNCTION ---
DEVICE = get_device()
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
MODEL = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
MODEL.eval()
# ---------------------------------------

def sequence_to_kmers(sequence: str, k: int = 6) -> str:
    sequence = sequence.upper().replace(" ", "").strip()
    kmers = [sequence[i:i + k] for i in range(len(sequence) - k + 1)]
    return " ".join(kmers)

def get_embeddings_batched(sequences: list, batch_size: int = 32) -> np.ndarray:
    """Processes a list of sequences in batches using Mean Pooling for maximum GPU efficiency."""
    all_embeddings = []

    # Process in chunks (batches)
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i: i + batch_size]
        kmers = [sequence_to_kmers(s) for s in batch_seqs]

        # Use the global TOKENIZER
        inputs = TOKENIZER(kmers, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: val.to(DEVICE) for key, val in inputs.items()}

        with torch.no_grad():
            # Use the global MODEL
            outputs = MODEL(**inputs)

            mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask

        all_embeddings.append(mean_pooled.cpu().numpy())

    return np.vstack(all_embeddings)

if __name__ == "__main__":
    sample_dna = "ATGCGTACGTAGCTAGCTAGCATCGATCGATCGATCGA"
    print(f"Processing sample sequence: {sample_dna[:15]}...")
    embedding_vector = get_embeddings_batched([sample_dna])
    print(f"Successfully extracted embeddings!")
    print(f"Embedding shape: {embedding_vector.shape}")