from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import torch
from PIL import Image

# Optional Transformers imports (only load what you use)
from transformers import CLIPProcessor, CLIPModel
from transformers import ViltProcessor, ViltModel
from transformers import AutoTokenizer, AutoModel


# --------------------------
# Global model for tokenization and dimension reference
model_name = "bert-base-uncased"   # any masked-LM or encoder works; dim stays fixed for a given model
tok = AutoTokenizer.from_pretrained(model_name)
mdl = AutoModel.from_pretrained(model_name)
# --------------------------
# Helpers
# --------------------------

def word_from_filename(p: Union[str, Path]) -> str:
    name = Path(p).stem
    name = name.replace("_", " ").replace("-", " ").strip()
    return name

def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

def _avg_last_k_layers(hidden_states: List[torch.Tensor], k: int = 4) -> torch.Tensor:
    # hidden_states: list length L, each [B, T, D]; return [B, T, D]
    k = min(k, len(hidden_states))
    return torch.stack(hidden_states[-k:]).mean(0)

def _token_indices_for_span(text: str, word: str, tokenizer) -> List[int]:
    """
    Find token indices that cover the first occurrence of `word` (case-insensitive) in `text`.
    Works with *fast* tokenizers that return offsets.
    """
    text_low = text.lower()
    word_low = word.lower()
    start_char = text_low.find(word_low)
    if start_char < 0:
        return []  # not found; fall back to heuristic if needed

    end_char = start_char + len(word_low)
    tok = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = tok["offset_mapping"]  # List[(s, e)]
    idxs = []
    for i, (s, e) in enumerate(offsets):
        # Special tokens often have (0,0); skip those
        if e == 0 and s == 0:
            continue
        if s >= start_char and e <= end_char:
            idxs.append(i)
    return idxs

# --------------------------
# Main embedder
# --------------------------

class MultimodalEmbedder:
    """
    backend='clip' -> returns {'image': [D], 'text': [D]}
    backend='vilt' -> returns {'word_fused': [D], 'cls_fused': [D]}
    """
    def __init__(
        self,
        backend: str = "clip",
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        fp16: bool = True,
    ):
        self.backend = backend.lower()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.fp16 = fp16 and self.device.type == "cuda"

        if self.backend == "clip":
            self.model_name = model_name or "openai/clip-vit-base-patch32"
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()

        elif self.backend == "vilt":
            self.model_name = model_name or "dandelin/vilt-b32-mlm"
            self.processor = ViltProcessor.from_pretrained(self.model_name)
            self.model = ViltModel.from_pretrained(self.model_name).to(self.device).eval()

        elif self.backend == "visualbert":
            raise NotImplementedError(
                "VisualBERT requires object-region features. "
                "If you can provide region features (visual_embeds/boxes), I can wire this up."
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

    @torch.no_grad()
    def embed_one(self, image_path: Union[str, Path], word: Optional[str] = None,
                  text_template: str = "a photo of {word}"):
        """
        Returns a dict of numpy vectors depending on backend.
        """
        image_path = Path(image_path)
        word = word or word_from_filename(image_path)
        # Remove numbers and convert letters to lowercase in the word
        word = ''.join([c.lower() for c in word if not c.isdigit()])
        image = Image.open(image_path).convert("RGB")

        if self.backend == "clip":
            text = word
            batch = self.processor(text=[text], images=[image], return_tensors="pt", padding=True)
            batch = _to_device(batch, self.device)

            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.fp16):
                out = self.model(**batch)

            img_vec = torch.nn.functional.normalize(out.image_embeds, dim=-1)[0].detach().cpu().numpy()
            txt_vec = torch.nn.functional.normalize(out.text_embeds, dim=-1)[0].detach().cpu().numpy()
            return {"image": img_vec, "text": txt_vec}

        elif self.backend == "vilt":
            # Build a short prompt to ensure the target word appears explicitly
            text = text_template.format(word=word)
            batch = self.processor(text=[text], images=[image], return_tensors="pt", padding=True)
            batch = _to_device(batch, self.device)

            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.fp16):
                out = self.model(**batch, output_hidden_states=True, return_dict=True)

            # CLS-pooled fused representation
            fused_cls = out.pooler_output[0].detach().float().cpu().numpy()

            # Word-level fused embedding: average last 4 layers over the subword span
            last4 = _avg_last_k_layers(out.hidden_states, k=4)  # [1, T, D]
            # Re-tokenize to get offsets (same tokenizer as processor)
            idxs = _token_indices_for_span(text, word, self.processor.tokenizer)
            if not idxs:
                # fallback: use the position of the first non-special token(s) of the word in plain tokenization
                # Here we just take the mean over all non-special text tokens (crude but safe)
                # You can improve by a smarter fallback if needed.
                ids = self.processor.tokenizer(text, add_special_tokens=True)["input_ids"]
                # exclude [CLS]=101 and [SEP]=102 if using BERT-base vocab
                idxs = [i for i, tid in enumerate(ids) if tid not in (101, 102)][:1]

            word_vec = last4[0, idxs].mean(0).detach().float().cpu().numpy()
            return {"word_fused": word_vec, "cls_fused": fused_cls}

    @torch.no_grad()
    def embed_folder(
        self,
        folder: Union[str, Path],
        patterns: tuple = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"),
        text_template: str = "a photo of {word}",
    ) -> Dict[str, np.ndarray]:
        """
        Iterates a folder, returns:
          - paths: list[str]
          - words: list[str]
          - vectors: np.ndarray [N, D] (for a single key) or dict of name->np.ndarray [N, D] if backend yields multiple vectors
        """
        folder = Path(folder)
        files = []
        for pat in patterns:
            files.extend(folder.rglob(pat))
        files = sorted(files)

        paths: List[str] = []
        words: List[str] = []
        # Collect per-key vectors in lists; we'll stack at the end
        per_key: Dict[str, List[np.ndarray]] = {}

        for p in files:
            w = word_from_filename(p)
            vecs = self.embed_one(p, word=w, text_template=text_template)

            paths.append(str(p))
            words.append(w.lower())
            for k, v in vecs.items():
                per_key.setdefault(k, []).append(v)

        # Stack
        for k in per_key:
            per_key[k] = np.stack(per_key[k], axis=0)  # [N, D]

        return {"paths": np.array(paths), "words": np.array(words), **per_key}
# --------------------------
# Example: get word embedding in context using BERT
# --------------------------
def word_embedding_in_context(sentence, target_word):
    # tokenize and find the token indices for the word (simple case; for robustness, match spans)
    enc = tok(sentence, return_tensors="pt")
    with torch.no_grad():
        out = mdl(**enc, output_hidden_states=True)
    # Use last hidden layer (or a mix); shape: [batch, seq_len, hidden]
    H = out.last_hidden_state[0]         

    # Find all token positions that decode to the target (handles subwords poorly if they split)
    tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
    positions = [i for i,t in enumerate(tokens) if t.replace("##","") == target_word]
    if not positions:
        # fallback: take mean of all subwords that contain the string
        positions = [i for i,t in enumerate(tokens) if target_word in t.replace("##","")]
    v = torch.mean(H[positions], dim=0) if positions else torch.mean(H, dim=0)  # fallback
    return v.numpy()


if __name__ == "__main__":  
    # Example usage
    import pickle as pk

    #embedder = MultimodalEmbedder(backend="clip", model_name="openai/clip-vit-base-patch32")
    embedder = MultimodalEmbedder(backend="vilt", model_name="dandelin/vilt-b32-mlm")
    res = embedder.embed_folder("data/pictureNaming VB")

    with open("data/pictureNaming VB/vilt_embeddings.pk", "wb") as f:
        pk.dump(res, f)
