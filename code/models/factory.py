# factory.py
# -*- coding: utf-8 -*-
"""Factory functions to build the ProtoMechanism model."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Any, Sequence

from config import CONFIG
from .proto_mech import ProtoMechanismClassifier


def _maybe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    if x is None:
        return default
    try:
        return float(x)
    except Exception:
        return default


def _build_proto_mech_from_flags(input_dim: int, flags: Dict[str, Any]) -> ProtoMechanismClassifier:
    feature_names = flags.get("proto_feature_names")
    pathway_indices = flags.get("proto_pathway_indices")
    drug_indices = flags.get("proto_drug_indices")
    prior_indices_raw = flags.get("proto_prior_indices") or {}
    prior_indices: Dict[int, Sequence[int]] = {}
    for key, value in prior_indices_raw.items():
        try:
            proto_idx = int(key)
        except (TypeError, ValueError):
            continue
        indices = list(value) if isinstance(value, (list, tuple)) else []
        prior_indices[proto_idx] = indices

    weight_decay = _maybe_float(flags.get("prototype_weight_decay", flags.get("weight_decay", 0.0)), 0.0)
    seed = flags.get("seed")
    if isinstance(seed, str):
        try:
            seed = int(seed)
        except Exception:
            seed = None

    device = flags.get("device")
    if isinstance(device, str):
        device = device.strip().lower()
        if device not in {"cuda", "cpu"}:
            device = None
    else:
        device = None

    prior_usage_weight = _maybe_float(
        flags.get(
            "proto_prior_usage_weight",
            flags.get("prototype_prior_usage_weight", 0.0),
        ),
        0.0,
    ) or 0.0
    prior_usage_target = _maybe_float(
        flags.get(
            "proto_prior_usage_target",
            flags.get("prototype_prior_usage_target"),
        ),
        None,
    )
    prior_usage_warmup = int(
        flags.get(
            "proto_prior_usage_warmup",
            flags.get("prototype_prior_usage_warmup", 0),
        )
        or 0
    )
    prior_usage_bias = _maybe_float(
        flags.get(
            "proto_prior_usage_bias",
            flags.get("prototype_prior_usage_bias", 0.0),
        ),
        0.0,
    ) or 0.0

    return ProtoMechanismClassifier(
        input_dim=input_dim,
        num_prototypes=int(flags.get("proto_num", 14)),
        top_k=int(flags.get("proto_topk", 3)),
        temperature=float(flags.get("prototype_temperature", 0.8)),
        dropout=float(flags.get("prototype_dropout", 0.1)),
        lr=float(flags.get("prototype_lr", 1e-3)),
        weight_decay=weight_decay if weight_decay is not None else 0.0,
        epochs=int(flags.get("prototype_epochs", 50)),
        batch_size=int(flags.get("prototype_batch_size", 256)),
        patience=int(flags.get("prototype_patience", 10)),
        mech_lambda=float(flags.get("proto_mech_lambda", 1.0)),
        diversity_lambda=float(flags.get("proto_diversity_lambda", 0.1)),
        balance_lambda=float(flags.get("proto_balance_lambda", 0.0)),
        entropy_lambda=float(flags.get("proto_entropy_lambda", 0.0)),
        prior_indices=prior_indices or None,
        feature_names=feature_names,
        enable_mechanism_film=bool(flags.get("enable_mechanism_film", False)),
        mechanism_film_hidden_dim=int(flags.get("mechanism_film_hidden_dim", 64)),
        pathway_feature_indices=pathway_indices,
        drug_feature_indices=drug_indices,
        prior_usage_target=prior_usage_target,
        prior_usage_weight=float(prior_usage_weight),
        prior_usage_warmup=prior_usage_warmup,
        prior_usage_bias=float(prior_usage_bias),
        seed=seed,
        device=device,
    )


def build_model(
    model_type: str,
    *,
    input_dim: int,
    random_state: int = 42,
    config: Dict = CONFIG,
    feature_names: Optional[Iterable[str]] = None,
    pathway_indices: Optional[Iterable[int]] = None,
    drug_indices: Optional[Iterable[int]] = None,
) -> ProtoMechanismClassifier:
    if model_type != "proto_mech":
        raise ValueError("Only 'proto_mech' is supported in this minimal repo.")
    flags = dict(config.get("pipeline_flags", {}))
    if feature_names is not None:
        flags["proto_feature_names"] = list(feature_names)
    if pathway_indices is not None:
        flags["proto_pathway_indices"] = list(pathway_indices)
    if drug_indices is not None:
        flags["proto_drug_indices"] = list(drug_indices)
    return _build_proto_mech_from_flags(input_dim, flags)
