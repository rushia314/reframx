"""Text binary classification with a small Attention network on GPU + MMEngine.

Dataset: https://huggingface.co/datasets/sg247/binary-classification
"""

from __future__ import annotations

import tiktoken
import torch
from datasets import load_dataset  # type: ignore[import-not-found]
from torch import nn
from torch.nn.utils.rnn import pad_sequence

import reframx as rf
from reframx.integrations.mmengine import MMEngineBackend
from reframx.integrations.pytorch import (
    BCEWithLogitsObjective,
    TorchAccuracyEvaluator,
    TorchClassificationMethod,
    TorchDatasetAdapter,
    TorchModelAdapter,
    TorchStrategy,
)


class TinyAttentionClassifier(nn.Module):
    """Small Transformer-Encoder classifier for binary text classification."""

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        max_len: int = 128,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.head = nn.Linear(d_model, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        pad_mask = input_ids.eq(self.pad_id)

        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = rf.probe("encoder/hidden", x)

        valid = (~pad_mask).unsqueeze(-1).to(x.dtype)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)


def main() -> None:
    torch.manual_seed(42)
    max_len = 128

    # 1. Load dataset & tokenize with tiktoken
    raw_ds = load_dataset("sg247/binary-classification")
    enc = tiktoken.get_encoding("cl100k_base")
    pad_id = enc.n_vocab

    cols = raw_ds["train"].column_names
    text_col = "tweet" if "tweet" in cols else cols[0]
    label_col = "label" if "label" in cols else cols[1]

    raw_ds = raw_ds.filter(
        lambda ex: ex[text_col] is not None and ex[label_col] is not None
    )

    def tokenize(example: dict) -> dict:
        ids = enc.encode(str(example[text_col]), disallowed_special=())[:max_len]
        return {
            "inputs": ids or [enc.eot_token],
            "targets": float(example[label_col]),
        }

    tokenized = raw_ds.map(tokenize, remove_columns=cols)
    val_key = "test" if "test" in tokenized else (
        "validation" if "validation" in tokenized else None
    )
    if val_key is not None and len(tokenized[val_key]) > 0:
        train_split, val_split = tokenized["train"], tokenized[val_key]
    else:
        split = tokenized["train"].train_test_split(test_size=0.2, seed=42)
        train_split, val_split = split["train"], split["test"]

    def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
        seqs = [torch.tensor(item["inputs"], dtype=torch.long) for item in batch]
        return {
            "inputs": pad_sequence(seqs, batch_first=True, padding_value=pad_id),
            "targets": torch.tensor(
                [[item["targets"]] for item in batch], dtype=torch.float32
            ),
        }

    dataset = TorchDatasetAdapter(
        datasets={"train": train_split, "val": val_split},
        dataloader_kwargs={
            "batch_size": 64,
            "collate_fn": collate_fn,
            "pin_memory": torch.cuda.is_available(),
        },
        capabilities=["inputs", "targets"],
    )

    # 2. Model, Optimizer, Instrumentation
    raw_model = TinyAttentionClassifier(vocab_size=enc.n_vocab + 1, pad_id=pad_id, max_len=max_len)
    model = TorchModelAdapter(raw_model)
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=1e-3, weight_decay=1e-2)

    inst = rf.InstrumentationManager(max_records=200)
    inst.watch_probe("encoder/hidden", rf.tensor_stats, every_n_steps=20)

    # 3. Compose & run on GPU with MMEngine (train + eval + save_best/save_last)
    experiment = rf.Experiment(
        name="TextAttention_Binary",
        task=rf.ClassificationTask(num_classes=2),
        method=TorchClassificationMethod(),
        model=model,
        dataset=dataset,
        objectives=BCEWithLogitsObjective(),
        optimizers=optimizer,
        evaluator=TorchAccuracyEvaluator(),
        strategy=TorchStrategy(
            device="auto",
            precision="fp16" if torch.cuda.is_available() else "fp32",
            max_grad_norm=1.0,
        ),
        backend=MMEngineBackend(
            work_dir="./work_dirs/text_attention_binary",
            default_hooks={
                "checkpoint": {
                    "type": "CheckpointHook",
                    "interval": 1,
                    "save_best": "accuracy",
                    "rule": "greater",
                    "save_last": True,
                    "max_keep_ckpts": 2,
                }
            },
        ),
        hooks=[rf.InstrumentationHook(inst)],
    )

    run = experiment.run(max_epochs=5)
    print(f"Run {run.run_id} finished: {run.status.value}")
    if run.eval_history:
        print(f"Final eval: {run.eval_history[-1]}")

    rfx_path = experiment.export("./work_dirs/text_attention_binary.rfx", profile="standard")
    print(f"Exported: {rfx_path}")


if __name__ == "__main__":
    main()
