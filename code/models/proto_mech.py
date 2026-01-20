"""Mechanism-aligned prototype classifier that operates in input feature space."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def _as_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32)


def _make_device(device: Optional[str]) -> torch.device:
    if device:
        dev = device.lower()
        if dev == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if dev == "cpu":
            return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ProtoMechanismClassifier:
    """Prototype classifier with mechanism alignment and diversity regularisation."""

    input_dim: int
    num_prototypes: int = 16
    top_k: int = 4
    dropout: float = 0.1
    temperature: float = 0.8
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 50
    batch_size: int = 256
    patience: int = 10
    mech_lambda: float = 1.5
    diversity_lambda: float = 0.1
    balance_lambda: float = 0.0
    entropy_lambda: float = 0.0
    prior_indices: Optional[Dict[int, Sequence[int]]] = None
    feature_names: Optional[Sequence[str]] = None
    prior_usage_target: Optional[float] = None
    prior_usage_weight: float = 0.0
    prior_usage_warmup: int = 0
    prior_usage_bias: float = 0.0
    track_usage: bool = True
    enable_mechanism_film: bool = False
    mechanism_film_hidden_dim: int = 64
    pathway_feature_indices: Optional[Sequence[int]] = None
    drug_feature_indices: Optional[Sequence[int]] = None
    seed: Optional[int] = None
    device: Optional[str] = None

    _prototypes: nn.Parameter = field(init=False, repr=False)
    _head: nn.Module = field(init=False, repr=False)
    _dropout: nn.Dropout = field(init=False, repr=False)
    _optimizer: optim.Optimizer = field(init=False, repr=False)
    _device: torch.device = field(init=False, repr=False)
    _prior_masks: List[Tuple[int, torch.Tensor]] = field(default_factory=list, init=False, repr=False)
    _prior_proto_ids: List[int] = field(default_factory=list, init=False, repr=False)
    _best_state: Optional[Dict[str, torch.Tensor]] = field(default=None, init=False, repr=False)
    _diagnostics: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _prior_usage_history: List[float] = field(default_factory=list, init=False, repr=False)
    _prototype_usage_history: List[List[float]] = field(default_factory=list, init=False, repr=False)
    _film: Optional[nn.Module] = field(default=None, init=False, repr=False)
    _film_enabled: bool = field(default=False, init=False, repr=False)
    _film_pathway_idx: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _film_drug_idx: Optional[torch.Tensor] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._device = _make_device(self.device)
        torch.manual_seed(self.seed or 42)
        if self._device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed or 42)
        if self.feature_names is not None:
            self.feature_names = list(self.feature_names)

        self._prototypes = nn.Parameter(
            torch.randn(self.num_prototypes, self.input_dim, device=self._device) * 0.01
        )
        self._dropout = nn.Dropout(self.dropout)
        self._head = nn.Linear(self.input_dim, 1).to(self._device)

        self._film_enabled = bool(self.enable_mechanism_film)
        self._film = None
        self._film_pathway_idx = None
        self._film_drug_idx = None
        if self._film_enabled:
            pathway_idx = [int(idx) for idx in (self.pathway_feature_indices or []) if 0 <= int(idx) < self.input_dim]
            drug_idx = [int(idx) for idx in (self.drug_feature_indices or []) if 0 <= int(idx) < self.input_dim]
            if pathway_idx and drug_idx:
                self._film_pathway_idx = torch.as_tensor(pathway_idx, dtype=torch.long, device=self._device)
                self._film_drug_idx = torch.as_tensor(drug_idx, dtype=torch.long, device=self._device)
                hidden_dim = max(1, int(self.mechanism_film_hidden_dim))
                out_dim = 2 * self._film_pathway_idx.numel()
                self._film = nn.Sequential(
                    nn.Linear(self._film_drug_idx.numel(), hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, out_dim),
                ).to(self._device)
            else:
                self._film_enabled = False

        params = list(self._head.parameters()) + [self._prototypes]
        if self._film_enabled and self._film is not None:
            params += list(self._film.parameters())
        self._optimizer = optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

        self._prior_masks = self._build_prior_masks()
        self._prior_proto_ids = [idx for idx, _ in self._prior_masks]

    # ------------------------------------------------------------------ #
    #                      Core forward computations                     #
    # ------------------------------------------------------------------ #

    def _build_prior_masks(self) -> List[Tuple[int, torch.Tensor]]:
        masks: List[Tuple[int, torch.Tensor]] = []
        if not self.prior_indices:
            return masks
        for key, indices in self.prior_indices.items():
            if key < 0 or key >= self.num_prototypes:
                continue
            mask = torch.zeros(self.input_dim, dtype=torch.bool)
            valid = [idx for idx in indices if 0 <= idx < self.input_dim]
            if not valid:
                continue
            mask[valid] = True
            masks.append((key, mask))
        return masks

    def _get_prior_target_value(self) -> float:
        if self.prior_usage_target is not None:
            return float(np.clip(self.prior_usage_target, 1e-4, 1.0))
        return float(1.0 / max(1, self.num_prototypes))

    def _forward(
        self,
        inputs: torch.Tensor,
        *,
        return_aux: bool = False,
        include_full_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        x = inputs.to(self._device)
        if (
            self._film_enabled
            and self._film is not None
            and self._film_pathway_idx is not None
            and self._film_drug_idx is not None
        ):
            x_path = x.index_select(1, self._film_pathway_idx)
            x_drug = x.index_select(1, self._film_drug_idx)
            film_out = self._film(x_drug)
            gamma, beta = torch.chunk(film_out, 2, dim=1)
            x_path = gamma * x_path + beta
            x = x.clone()
            x[:, self._film_pathway_idx] = x_path
        x_norm = F.normalize(x, dim=-1)
        proto_norm = F.normalize(self._prototypes, dim=-1)
        similarity = torch.matmul(x_norm, proto_norm.T)
        if self._prior_proto_ids and float(self.prior_usage_bias) > 0.0:
            similarity = similarity.clone()
            bias_value = float(self.prior_usage_bias)
            similarity[:, self._prior_proto_ids] = similarity[:, self._prior_proto_ids] + bias_value

        if self.top_k < self.num_prototypes:
            top_scores, top_indices = torch.topk(similarity, self.top_k, dim=-1)
        else:
            top_scores = similarity
            top_indices = torch.arange(self.num_prototypes, device=self._device).unsqueeze(0).expand(x.shape[0], -1)

        safe_temp = max(1e-6, float(self.temperature))
        weights = F.softmax(top_scores / safe_temp, dim=-1)
        selected_proto = self._prototypes[top_indices]  # [B, top_k, D]
        z_agg = (weights.unsqueeze(-1) * selected_proto).sum(dim=1)
        z_agg = self._dropout(z_agg)
        logits = self._head(z_agg).squeeze(-1)

        aux = None
        if return_aux:
            aux = {
                "similarity": similarity.detach().cpu(),
                "topk_indices": top_indices.detach().cpu(),
                "topk_weights": weights.detach().cpu(),
                "topk_weights_raw": weights,
                "z_agg": z_agg.detach().cpu(),
            }
            if self._film_enabled and self._film is not None:
                aux["film_gamma"] = gamma.detach().cpu()
                aux["film_beta"] = beta.detach().cpu()
            if include_full_weights:
                full_weights = torch.zeros(weights.size(0), self.num_prototypes, device=self._device)
                full_weights.scatter_add_(1, top_indices, weights)
                aux["weight_matrix"] = full_weights
        return logits, aux

    def _mechanism_loss(self) -> torch.Tensor:
        if not self._prior_masks:
            return torch.zeros(1, device=self._device).squeeze()
        abs_proto = self._prototypes.abs()
        eps = 1e-6
        deltas = []
        for proto_idx, mask in self._prior_masks:
            values_on = abs_proto[proto_idx, mask]
            values_off = abs_proto[proto_idx, ~mask]
            if values_on.numel() == 0:
                continue
            mu_plus = values_on.mean()
            mu_minus = values_off.mean() if values_off.numel() > 0 else torch.zeros(1, device=self._device).squeeze()
            delta = mu_plus / (mu_plus + mu_minus + eps)
            deltas.append(delta)
        if not deltas:
            return torch.zeros(1, device=self._device).squeeze()
        return -torch.stack(deltas).mean()

    def _diversity_loss(self) -> torch.Tensor:
        if self.num_prototypes <= 1:
            return torch.zeros(1, device=self._device).squeeze()
        proto_norm = F.normalize(self._prototypes, dim=-1)
        gram = torch.matmul(proto_norm, proto_norm.T)
        off_diag = torch.triu(gram, diagonal=1)
        return (off_diag ** 2).sum() * 2.0 / (self.num_prototypes * (self.num_prototypes - 1))

    def _balance_loss(self, weight_matrix: Optional[torch.Tensor]) -> torch.Tensor:
        if weight_matrix is None or weight_matrix.numel() == 0:
            return torch.zeros(1, device=self._device).squeeze()
        usage = weight_matrix.mean(dim=0)
        target = torch.full_like(usage, 1.0 / max(1, self.num_prototypes))
        return torch.mean((usage - target) ** 2)

    def _entropy_bonus(self, weights: Optional[torch.Tensor]) -> torch.Tensor:
        if weights is None:
            return torch.zeros(1, device=self._device).squeeze()
        eps = 1e-8
        entropy = -(weights * torch.log(weights + eps)).sum(dim=1)
        return entropy.mean()

    # ------------------------------------------------------------------ #
    #                             Training                               #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        **_: object,
    ) -> "ProtoMechanismClassifier":
        self._head.train()
        self._dropout.train()
        inputs = _as_tensor(X)
        targets = torch.as_tensor(y.reshape(-1, 1), dtype=torch.float32)
        if sample_weight is not None:
            weights = torch.as_tensor(sample_weight.reshape(-1, 1), dtype=torch.float32)
        else:
            weights = torch.ones_like(targets)

        dataset = TensorDataset(inputs, targets, weights)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )
        criterion = nn.BCEWithLogitsLoss(reduction="none")

        best_loss = float("inf")
        epochs_no_improve = 0
        self._diagnostics = {}
        self._prior_usage_history = []
        self._prototype_usage_history = []
        need_weight_matrix = (
            self.balance_lambda > 0
            or self.prior_usage_weight > 0
            or bool(self._prior_proto_ids)
            or bool(self.track_usage)
        )

        for epoch in range(self.epochs):
            epoch_loss = []
            mech_vals = []
            div_vals = []
            balance_vals = []
            entropy_vals = []
            prior_usage_vals = []
            prior_penalty_vals = []
            prior_vectors: List[np.ndarray] = []
            usage_vectors: List[np.ndarray] = []
            for batch_inputs, batch_targets, batch_weights in loader:
                self._optimizer.zero_grad()
                logits, aux = self._forward(
                    batch_inputs,
                    return_aux=True,
                    include_full_weights=need_weight_matrix,
                )
                loss_vec = criterion(logits.view(-1, 1), batch_targets.to(self._device))
                bce = (loss_vec * batch_weights.to(self._device)).mean()
                mech_loss = self._mechanism_loss()
                div_loss = self._diversity_loss()
                balance_loss = (
                    self._balance_loss(aux.get("weight_matrix"))
                    if (self.balance_lambda > 0 and aux is not None)
                    else torch.zeros(1, device=self._device).squeeze()
                )
                entropy_bonus = (
                    self._entropy_bonus(aux.get("topk_weights_raw"))
                    if (self.entropy_lambda > 0 and aux is not None)
                    else torch.zeros(1, device=self._device).squeeze()
                )
                weight_matrix = aux.get("weight_matrix") if (aux and need_weight_matrix) else None
                if weight_matrix is not None:
                    usage_vec = weight_matrix.mean(dim=0)
                    usage_vectors.append(usage_vec.detach().cpu().numpy())

                prior_loss = None
                if weight_matrix is not None and self._prior_proto_ids:
                    proto_usage = weight_matrix.mean(dim=0)
                    prior_vector = proto_usage[self._prior_proto_ids]
                    prior_vectors.append(prior_vector.detach().cpu().numpy())
                    prior_usage_vals.append(float(prior_vector.mean().detach().cpu()))
                    if (
                        self.prior_usage_weight > 0
                        and epoch >= int(self.prior_usage_warmup)
                    ):
                        target = self._get_prior_target_value()
                        target_tensor = torch.full_like(prior_vector, target)
                        prior_loss = torch.relu(target_tensor - prior_vector).mean()
                        prior_penalty_vals.append(float(prior_loss.detach().cpu()))

                total = bce + self.mech_lambda * mech_loss + self.diversity_lambda * div_loss
                if self.balance_lambda > 0:
                    total += self.balance_lambda * balance_loss
                if self.entropy_lambda > 0:
                    total -= self.entropy_lambda * entropy_bonus
                if prior_loss is not None:
                    total += self.prior_usage_weight * prior_loss
                total.backward()
                self._optimizer.step()
                epoch_loss.append(float(bce.detach().cpu()))
                mech_vals.append(float(mech_loss.detach().cpu()))
                div_vals.append(float(div_loss.detach().cpu()))
                if self.balance_lambda > 0:
                    balance_vals.append(float(balance_loss.detach().cpu()))
                if self.entropy_lambda > 0:
                    entropy_vals.append(float(entropy_bonus.detach().cpu()))

            mean_loss = float(np.mean(epoch_loss)) if epoch_loss else float("inf")
            self._diagnostics = {
                "train_bce": mean_loss,
                "mechanism_loss": float(np.mean(mech_vals)) if mech_vals else 0.0,
                "diversity_loss": float(np.mean(div_vals)) if div_vals else 0.0,
                "balance_loss": float(np.mean(balance_vals)) if balance_vals else 0.0,
                "entropy_bonus": float(np.mean(entropy_vals)) if entropy_vals else 0.0,
                "prototype_usage": (
                    np.stack(usage_vectors).mean(axis=0).tolist() if usage_vectors else None
                ),
            }
            if usage_vectors:
                self._prototype_usage_history.append(
                    np.stack(usage_vectors).mean(axis=0).tolist()
                )
            else:
                self._prototype_usage_history.append([])

            if self._prior_proto_ids:
                self._diagnostics.update(
                    {
                        "prior_usage_share": float(np.mean(prior_usage_vals)) if prior_usage_vals else None,
                        "prior_usage_penalty": float(np.mean(prior_penalty_vals)) if prior_penalty_vals else None,
                        "prior_proto_indices": list(self._prior_proto_ids),
                        "prior_proto_usage": (
                            np.stack(prior_vectors).mean(axis=0).tolist() if prior_vectors else None
                        ),
                        "prior_usage_target": self.prior_usage_target if self.prior_usage_target is not None else self._get_prior_target_value(),
                        "prior_usage_weight": float(self.prior_usage_weight),
                        "prior_usage_warmup": int(self.prior_usage_warmup),
                    }
                )
                if prior_usage_vals:
                    self._prior_usage_history.append(float(np.mean(prior_usage_vals)))
                else:
                    self._prior_usage_history.append(None)

            if mean_loss < best_loss - 1e-6:
                best_loss = mean_loss
                epochs_no_improve = 0
                self._best_state = {
                    "prototypes": self._prototypes.detach().cpu().clone(),
                    "head": self._head.state_dict(),
                }
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break

        if self._best_state:
            self._prototypes.data = self._best_state["prototypes"].to(self._device)
            self._head.load_state_dict(self._best_state["head"])
        return self

    # ------------------------------------------------------------------ #
    #                            Inference                               #
    # ------------------------------------------------------------------ #

    def _predict_logits(
        self,
        X: np.ndarray,
        *,
        return_aux: bool = False,
    ) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        self._head.eval()
        self._dropout.eval()
        inputs = _as_tensor(X)
        with torch.no_grad():
            logits, aux = self._forward(inputs, return_aux=return_aux, include_full_weights=False)
            logits_np = logits.detach().cpu().numpy()
            aux_np = None
            if aux:
                aux_np = {}
                for key, value in aux.items():
                    if key in {"topk_weights_raw", "weight_matrix"}:
                        continue
                    aux_np[key] = value.numpy()
            return logits_np, aux_np

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits, _ = self._predict_logits(X, return_aux=False)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.vstack([1.0 - probs, probs]).T

    def predict_with_activations(
        self,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        logits, aux = self._predict_logits(X, return_aux=True)
        probs = 1.0 / (1.0 + np.exp(-logits))
        if aux is None:
            aux = {}
        aux["probabilities"] = probs.copy()
        return probs, aux

    # ------------------------------------------------------------------ #
    #                              Utilities                             #
    # ------------------------------------------------------------------ #

    def get_training_diagnostics(self) -> Dict[str, float]:
        diag = dict(self._diagnostics)
        diag["prior_usage_history"] = list(self._prior_usage_history)
        diag["prototype_usage_history"] = list(self._prototype_usage_history)
        diag["film_enabled"] = bool(self._film_enabled)
        return diag

    def describe_functional_features(
        self,
        feature_names: Optional[Sequence[str]] = None,
        *,
        functional_prefixes: Optional[Sequence[str]] = None,
        non_functional_prefixes: Optional[Sequence[str]] = None,
        keywords: Optional[Sequence[str]] = None,
        max_examples: int = 10,
        keyword_match_limit: Optional[int] = 20,
    ) -> Dict[str, object]:
        """Summarise pathway/functional features to help crafting priors."""

        names_source = feature_names or self.feature_names
        if not names_source:
            raise ValueError("feature_names is required to describe functional features.")
        names = [str(name) for name in names_source]

        grouped: Dict[str, List[Tuple[int, str]]] = {}
        for idx, raw_name in enumerate(names):
            prefix = raw_name.split("_", 1)[0].lower()
            grouped.setdefault(prefix, []).append((idx, raw_name))

        prefix_counts = {prefix: len(entries) for prefix, entries in grouped.items()}
        default_non_functional = {"pca", "pc", "gene", "expr"}
        excluded = (
            {p.lower() for p in non_functional_prefixes}
            if non_functional_prefixes
            else default_non_functional
        )

        if functional_prefixes:
            functional = {p.lower() for p in functional_prefixes}
        else:
            functional = {prefix for prefix in grouped if prefix not in excluded}

        max_examples = max(0, int(max_examples))
        functional_details: Dict[str, Dict[str, object]] = {}
        for prefix in sorted(functional):
            entries = grouped.get(prefix)
            if not entries:
                continue
            indices = [idx for idx, _ in entries]
            examples = [name for _, name in entries[:max_examples]] if max_examples else []
            functional_details[prefix] = {
                "count": len(entries),
                "indices": indices,
                "examples": examples,
            }

        keyword_matches: Dict[str, List[Dict[str, object]]] = {}
        if keywords:
            limit = None if keyword_match_limit is None else max(0, int(keyword_match_limit))
            for keyword in keywords:
                term = str(keyword).strip()
                if not term:
                    continue
                needle = term.lower()
                hits: List[Dict[str, object]] = []
                for idx, name in enumerate(names):
                    if needle in name.lower():
                        hits.append({"index": idx, "feature": name})
                        if limit is not None and len(hits) >= limit:
                            break
                keyword_matches[term] = hits

        return {
            "total_features": len(names),
            "prefix_counts": prefix_counts,
            "functional_prefixes": sorted(functional_details),
            "functional_features": functional_details,
            "keyword_matches": keyword_matches if keyword_matches else None,
        }

    def get_feature_importance(
        self,
        feature_names: Sequence[str],
        top_k: int = 10,
    ) -> List[Dict[str, float]]:
        proto_tensor = self._prototypes.detach().cpu().numpy()
        importance = np.abs(proto_tensor).mean(axis=0)
        signed = proto_tensor.mean(axis=0).reshape(-1)
        indices = np.argsort(importance)[::-1][: min(top_k, len(feature_names))]
        return [
            {
                "feature": str(feature_names[idx]),
                "coefficient": float(signed[idx]),
                "importance": float(importance[idx]),
            }
            for idx in indices
        ]

    def export_prototypes(self) -> Dict[str, object]:
        return {
            "prototypes": self._prototypes.detach().cpu().numpy(),
            "feature_names": list(self.feature_names) if self.feature_names else None,
            "num_prototypes": self.num_prototypes,
            "top_k": self.top_k,
        }
