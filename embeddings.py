from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import torch
from PIL import Image

# Optional Transformers imports (only load what you use)
from transformers import CLIPProcessor, CLIPModel
from transformers import ViltProcessor, ViltModel
from transformers import ViTModel, ViTImageProcessor
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


def _pool_clip_vision_layer(hidden_state: torch.Tensor) -> torch.Tensor:
    """Pool a vision transformer hidden state [B, num_patches+1, D] using the CLS token (index 0)."""
    return hidden_state[:, 0, :]  # [B, D]


def _pool_clip_text_layer(hidden_state: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Pool a text transformer hidden state [B, seq_len, D] using the EOS token position.
    CLIP uses the EOS token (highest token id in each sequence) as the pooled representation.
    """
    # EOS token is at the position of the largest token id (eos_token_id) in each sequence
    eos_positions = input_ids.argmax(dim=-1)  # [B]
    pooled = hidden_state[torch.arange(hidden_state.size(0)), eos_positions]  # [B, D]
    return pooled


# --------------------------
# Main embedder
# --------------------------

class MultimodalEmbedder:
    """
    backend='clip'  -> returns layer-wise embeddings from both vision and text encoders
    backend='vit'   -> returns layer-wise embeddings from a pure ImageNet-trained ViT
    backend='vilt'  -> returns {'word_fused': [D], 'cls_fused': [D]}
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

        elif self.backend == "vit":
            self.model_name = model_name or "google/vit-base-patch32-224-in21k"
            self.processor = ViTImageProcessor.from_pretrained(self.model_name)
            self.model = ViTModel.from_pretrained(self.model_name).to(self.device).eval()

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

        For CLIP, returns:
          - 'vision_layer_00' ... 'vision_layer_12': CLS-pooled embedding at each layer [D_vision]
              Layer 00 = initial patch embedding, layers 01-12 = transformer block outputs
          - 'text_layer_00' ... 'text_layer_12': EOS-pooled embedding at each layer [D_text]
              Layer 00 = initial token embedding, layers 01-12 = transformer block outputs
          - 'vision_projected': final projected + normalized image embedding [D_proj]
          - 'text_projected': final projected + normalized text embedding [D_proj]

        For ViT (pure vision), returns:
          - 'vit_layer_00' ... 'vit_layer_12': CLS-pooled embedding at each layer [D]
              Layer 00 = initial patch embedding, layers 01-12 = transformer block outputs
          - 'vit_pooled': final pooler output [D]
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
                out = self.model(**batch, output_hidden_states=True)

            result = {}

            # --- Vision encoder: layer-wise embeddings ---
            # out.vision_model_output.hidden_states is a tuple of (num_layers + 1) tensors
            # each of shape [B, num_patches+1, D_vision]
            vision_hidden = out.vision_model_output.hidden_states
            for i, hs in enumerate(vision_hidden):
                pooled = _pool_clip_vision_layer(hs)  # [1, D_vision]
                result[f"vision_layer_{i:02d}"] = pooled[0].detach().float().cpu().numpy()

            # --- Text encoder: layer-wise embeddings ---
            # out.text_model_output.hidden_states is a tuple of (num_layers + 1) tensors
            # each of shape [B, seq_len, D_text]
            text_hidden = out.text_model_output.hidden_states
            input_ids = batch["input_ids"]
            for i, hs in enumerate(text_hidden):
                pooled = _pool_clip_text_layer(hs, input_ids)  # [1, D_text]
                result[f"text_layer_{i:02d}"] = pooled[0].detach().float().cpu().numpy()

            # --- Final projected embeddings (after projection head + normalization) ---
            img_vec = torch.nn.functional.normalize(out.image_embeds, dim=-1)[0].detach().cpu().numpy()
            txt_vec = torch.nn.functional.normalize(out.text_embeds, dim=-1)[0].detach().cpu().numpy()
            result["vision_projected"] = img_vec
            result["text_projected"] = txt_vec

            return result

        elif self.backend == "vit":
            # Pure vision ViT — no text input needed
            batch = self.processor(images=[image], return_tensors="pt")
            batch = _to_device(batch, self.device)

            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.fp16):
                out = self.model(**batch, output_hidden_states=True)

            result = {}

            # Layer-wise embeddings: CLS token (index 0) at each layer
            # out.hidden_states is a tuple of (num_layers + 1) tensors
            # each of shape [B, num_patches+1, D]
            for i, hs in enumerate(out.hidden_states):
                cls_vec = hs[:, 0, :]  # [1, D]
                result[f"vit_layer_{i:02d}"] = cls_vec[0].detach().float().cpu().numpy()

            # Final pooler output (CLS token after the pooling layer)
            if out.pooler_output is not None:
                result["vit_pooled"] = out.pooler_output[0].detach().float().cpu().numpy()

            return result

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
                ids = self.processor.tokenizer(text, add_special_tokens=True)["input_ids"]
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
        Iterates a folder, returns dict with:
          - 'paths': np.ndarray of file paths [N]
          - 'words': np.ndarray of word labels [N]
          - For CLIP: 'vision_layer_00' ... 'vision_layer_12' each [N, D_vision]
                      'text_layer_00' ... 'text_layer_12' each [N, D_text]
                      'vision_projected', 'text_projected' each [N, D_proj]
          - For ViT:  'vit_layer_00' ... 'vit_layer_12' each [N, D]
                      'vit_pooled' [N, D]
          - For ViLT: 'word_fused' [N, D], 'cls_fused' [N, D]
        """
        folder = Path(folder)
        files = []
        for pat in patterns:
            files.extend(folder.rglob(pat))
        files = sorted(files)

        paths: List[str] = []
        words: List[str] = []
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
    enc = tok(sentence, return_tensors="pt")
    with torch.no_grad():
        out = mdl(**enc, output_hidden_states=True)
    H = out.last_hidden_state[0]

    tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
    positions = [i for i,t in enumerate(tokens) if t.replace("##","") == target_word]
    if not positions:
        positions = [i for i,t in enumerate(tokens) if target_word in t.replace("##","")]
    v = torch.mean(H[positions], dim=0) if positions else torch.mean(H, dim=0)
    return v.numpy()


if __name__ == "__main__":
    import pickle as pk

    data_folder = "data/pictureNaming VB"

    # --- CLIP layer-wise embeddings ---
    print("=" * 50)
    print("Extracting CLIP layer-wise embeddings...")
    print("=" * 50)
    embedder = MultimodalEmbedder(backend="clip", model_name="openai/clip-vit-base-patch32")
    res = embedder.embed_folder(data_folder)

    print("Keys in CLIP result:")
    for k, v in res.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    with open(f"{data_folder}/clip_layerwise_embeddings.pk", "wb") as f:
        pk.dump(res, f)
    print("Saved to clip_layerwise_embeddings.pk\n")

    del embedder  # free memory
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- Pure ViT (ImageNet-trained) layer-wise embeddings ---
    print("=" * 50)
    print("Extracting ViT (ImageNet) layer-wise embeddings...")
    print("=" * 50)
    embedder = MultimodalEmbedder(backend="vit", model_name="google/vit-base-patch32-224-in21k")
    res = embedder.embed_folder(data_folder)

    print("Keys in ViT result:")
    for k, v in res.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    with open(f"{data_folder}/vit_imagenet_layerwise_embeddings.pk", "wb") as f:
        pk.dump(res, f)
    print("Saved to vit_imagenet_layerwise_embeddings.pk")