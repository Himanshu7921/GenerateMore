import torch
import torch.nn as nn
from utils import TransformerConfig, causal_mask, DataLoader, generate, print_model_summary, save_checkpoint, load_checkpoint
from model import DecoderOnlyTransformerModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_data(address="tiny_shakespeare.txt"):
    config = TransformerConfig()
    loader = DataLoader(dataset_address=address, seq_length=config.seq_length)
    return loader, config

dataloader, config = load_data()
loaded_model, _ = load_checkpoint(
    path="checkpoints/decoder_only_transformer.pt",
    model_class=DecoderOnlyTransformerModel,
    device=device,
)

loaded_model.eval()

start_text = "ROMEO:"
start_ids = torch.tensor(
    [[dataloader.stoi[c] for c in start_text]],
    dtype=torch.long,
    device=device
)

generated_ids_gpt_style_loaded_model = generate(
        model = loaded_model,
        idx = start_ids,
        max_new_tokens = 300,
        device = device,
        temperature = 0.8,
        top_k = None,
        top_p = 0.9,
)

# decode back to text
gpt_style_generated_text_loaded_model = "".join([dataloader.itos[i.item()] for i in generated_ids_gpt_style_loaded_model[0]])
print("GPT Style Sampling from Loaded Model: ")
print(gpt_style_generated_text_loaded_model)