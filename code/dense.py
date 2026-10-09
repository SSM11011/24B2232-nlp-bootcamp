"""Dense bi-encoders shared by Tasks 2-5.

A model is described by a `DenseModelSpec`; `load_encoder` returns an object with
`encode_queries(texts)`, `encode_docs(docs)` and `num_parameters()`. Embeddings are
L2-normalized, so retrieval is a plain matrix product (cosine similarity). They are
cached under cache/embeddings/<model>/ so later tasks reuse them without re-encoding.

Inference runs in fp16 on the GPU: about 8x faster than fp32 on our GPU, with
embeddings at cosine >= 0.9997 to the fp32 ones.
"""
from dataclasses import dataclass

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from common import PROJECT_ROOT, doc_text

EMBEDDINGS_DIR = PROJECT_ROOT / "cache" / "embeddings"
MEDCPT_QUERY_TOKENS = 64      # MedCPT's recommended query length
MEDCPT_ARTICLE_TOKENS = 512


@dataclass(frozen=True)
class DenseModelSpec:
    name: str                         # short label used in logs and cache paths
    family: str                       # "general", "scientific" or "finetuned"
    model_id: str                     # Hugging Face id (or local path) of the query encoder
    query_prefix: str = ""            # instruction prepended to claims (bge, e5, arctic)
    doc_prefix: str = ""              # instruction prepended to documents (e5)
    article_model_id: str | None = None  # MedCPT-style separate article encoder, fed (title, abstract)


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def for_inference(model):
    """Eval mode, and fp16 on the GPU."""
    model.eval()
    return model.half() if device() == "cuda" else model


class SentenceTransformerEncoder:
    """Symmetric bi-encoder loaded with sentence-transformers; documents use the shared doc_text."""

    def __init__(self, spec):
        self.spec = spec
        self.model = for_inference(SentenceTransformer(spec.model_id, device=device()))

    def num_parameters(self):
        return sum(p.numel() for p in self.model.parameters())

    def encode_docs(self, docs, batch_size=32):
        texts = [self.spec.doc_prefix + doc_text(doc) for doc in docs]
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)

    def encode_queries(self, texts, batch_size=64, as_tensor=False):
        return self.model.encode([self.spec.query_prefix + t for t in texts], batch_size=batch_size,
                                 normalize_embeddings=True, convert_to_tensor=as_tensor,
                                 convert_to_numpy=not as_tensor, show_progress_bar=False)


class MedCPTEncoder:
    """MedCPT: separate query and article encoders, [CLS] pooling, articles fed as (title, abstract) pairs."""

    def __init__(self, spec):
        self.query_tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        self.query_model = for_inference(AutoModel.from_pretrained(spec.model_id).to(device()))
        self.article_tokenizer = AutoTokenizer.from_pretrained(spec.article_model_id)
        self.article_model = for_inference(AutoModel.from_pretrained(spec.article_model_id).to(device()))

    def num_parameters(self):
        return sum(p.numel() for model in (self.query_model, self.article_model) for p in model.parameters())

    @torch.no_grad()
    def _embed(self, model, tokenizer, first, second, max_length, batch_size):
        chunks = []
        for start in range(0, len(first), batch_size):
            batch = tokenizer(first[start:start + batch_size],
                              second[start:start + batch_size] if second else None,
                              truncation=True, padding=True, max_length=max_length,
                              return_tensors="pt").to(device())
            cls = model(**batch).last_hidden_state[:, 0]
            chunks.append(torch.nn.functional.normalize(cls.float(), dim=-1))
        return torch.cat(chunks)

    def encode_docs(self, docs, batch_size=32):
        titles = [doc["title"] for doc in docs]
        abstracts = [" ".join(doc["abstract"]) for doc in docs]
        return self._embed(self.article_model, self.article_tokenizer, titles, abstracts,
                           MEDCPT_ARTICLE_TOKENS, batch_size).cpu().numpy()

    def encode_queries(self, texts, batch_size=64, as_tensor=False):
        embeddings = self._embed(self.query_model, self.query_tokenizer, list(texts), None,
                                 MEDCPT_QUERY_TOKENS, batch_size)
        return embeddings if as_tensor else embeddings.cpu().numpy()


def load_encoder(spec):
    return MedCPTEncoder(spec) if spec.article_model_id else SentenceTransformerEncoder(spec)


# ---------------------------------------------------------------- embedding cache
def save_embeddings(model_name, split, embeddings):
    """split is 'corpus' or a claim split ('train', 'val', 'eval')."""
    folder = EMBEDDINGS_DIR / model_name
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / f"{split}.npy", embeddings.astype(np.float32))


def load_embeddings(model_name, split):
    return np.load(EMBEDDINGS_DIR / model_name / f"{split}.npy")


def dense_score_matrix(model_name, split):
    """(num_claims, num_docs) cosine similarities from cached embeddings."""
    return load_embeddings(model_name, split) @ load_embeddings(model_name, "corpus").T
