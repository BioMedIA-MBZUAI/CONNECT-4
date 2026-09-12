"""Leakage-safe five-fold GAT evaluation for the paper's downstream tasks."""
from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping
from os import PathLike
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)
from scipy.stats import wilcoxon
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

from data.protocol import (
    SynthesisExclusionEvidence,
    load_completed_synthesis_checkpoint,
    validate_unseen_cohort_manifest,
)
from data.provenance import canonical_sha256, sha256_file


DOWNSTREAM_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "f1",
    "precision",
    "recall",
)

# Public provenance identifiers.  The fold allocator is implemented locally in
# ``_build_folds``; it is deliberately not scikit-learn's
# ``StratifiedGroupKFold``.  Version the settings schema when the recorded
# algorithm identity changes so previously mislabeled result records fail
# closed instead of being compared as if their provenance were equivalent.
DOWNSTREAM_FOLD_ALLOCATOR_ID = (
    "connect4_seeded_greedy_single_label_patient_group_scan_balance_5fold_v1"
)
DOWNSTREAM_GAT_SETTINGS_FORMAT = "connect4_downstream_gat_settings_v2"

_GRAPH_STREAM_FORMAT = "connect4_downstream_graph_stream_v1"
_GRAPH_STREAMS_FORMAT = "connect4_downstream_graph_streams_v1"
_PROTOCOL_IDENTITY_FORMAT = "connect4_downstream_protocol_identity_v1"


def _as_nonempty_strings(values: Iterable, name: str) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence, not a single string")
    result = np.asarray([str(value).strip() for value in values], dtype=str)
    if result.ndim != 1 or len(result) == 0 or np.any(result == ""):
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    return result


def _validated_protocol_arrays(
    labels: Sequence,
    patient_ids: Sequence[str],
    cohort_ids: Sequence[str],
    exclusion: SynthesisExclusionEvidence,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate mandatory evidence that downstream samples were never seen."""
    y = _as_nonempty_strings(labels, "labels")
    groups = _as_nonempty_strings(patient_ids, "patient_ids")
    cohorts = _as_nonempty_strings(cohort_ids, "cohort_ids")
    if len(y) != len(groups) or len(y) != len(cohorts):
        raise ValueError("labels, patient_ids, and cohort_ids must be equally sized")

    patient_overlap = sorted(set(groups.tolist()) & exclusion.patient_ids)
    if patient_overlap:
        raise ValueError(
            "Downstream data contain patients used by synthesis training: "
            f"{patient_overlap[:10]}"
        )
    canonical_cohorts = np.char.upper(cohorts)
    cohort_overlap = sorted(set(canonical_cohorts.tolist()) & exclusion.cohorts)
    if cohort_overlap:
        raise ValueError(
            "Downstream data contain cohorts used by synthesis training: "
            f"{cohort_overlap}"
        )

    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError("downstream evaluation requires at least two classes")
    for label in classes:
        patient_count = len(set(groups[y == label].tolist()))
        if patient_count < 5:
            raise ValueError(
                f"class {label!r} has only {patient_count} patients; five-fold "
                "evaluation requires at least five patient groups per class"
            )
    return y, groups, canonical_cohorts


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_completed_synthesis_checkpoint(
    checkpoint_path: str | PathLike[str],
) -> Tuple[Mapping, SynthesisExclusionEvidence, str]:
    """Load unseen-data exclusions only from a completed training checkpoint."""
    checkpoint, exclusion, checkpoint_sha256 = load_completed_synthesis_checkpoint(
        checkpoint_path
    )
    return checkpoint, exclusion, checkpoint_sha256


def _validate_unseen_manifest_arrays(
    scan_ids: Sequence[str],
    patient_ids: Sequence[str],
    cohort_ids: Sequence[str],
    *,
    manifest_path: str | PathLike[str],
    checkpoint: Mapping,
) -> str:
    scans = _as_nonempty_strings(scan_ids, "scan_ids")
    if len(set(scans.tolist())) != len(scans):
        raise ValueError("scan_ids must be unique for exact manifest coverage")
    if len(scans) != len(patient_ids) or len(scans) != len(cohort_ids):
        raise ValueError(
            "scan_ids, patient_ids, and cohort_ids must be equally sized"
        )
    evidence = validate_unseen_cohort_manifest(
        str(manifest_path), scans.tolist(), checkpoint["split_identity"]
    )
    expected_patients = np.asarray(
        [evidence.patient_by_scan[scan_id] for scan_id in scans], dtype=str
    )
    expected_cohorts = np.asarray(
        [evidence.cohort_by_scan[scan_id] for scan_id in scans], dtype=str
    )
    actual_patients = _as_nonempty_strings(patient_ids, "patient_ids")
    actual_cohorts = np.char.upper(_as_nonempty_strings(cohort_ids, "cohort_ids"))
    if not np.array_equal(actual_patients, expected_patients):
        raise ValueError("patient_ids differ from the exact unseen-cohort manifest")
    if not np.array_equal(actual_cohorts, expected_cohorts):
        raise ValueError("cohort_ids differ from the exact unseen-cohort manifest")
    return sha256_file(manifest_path)


def five_fold_downstream_splits(
    labels: Sequence,
    patient_ids: Sequence[str],
    *,
    scan_ids: Sequence[str],
    cohort_ids: Sequence[str],
    unseen_cohort_manifest_path: str | PathLike[str],
    synthesis_checkpoint_path: str | PathLike[str],
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build five stratified folds only after proving the data are unseen.

    All overlap evidence is mandatory: silently omitting cohort or synthesis-set
    identity cannot establish the manuscript's "completely unseen" assertion.
    Repeat scans from one patient are always kept in the same fold.
    """
    checkpoint, exclusion, _ = _load_completed_synthesis_checkpoint(
        synthesis_checkpoint_path
    )
    _validate_unseen_manifest_arrays(
        scan_ids,
        patient_ids,
        cohort_ids,
        manifest_path=unseen_cohort_manifest_path,
        checkpoint=checkpoint,
    )
    y, groups, _ = _validated_protocol_arrays(labels, patient_ids, cohort_ids, exclusion)
    return _build_folds(y, groups, seed=int(seed))


def _build_folds(
    y: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create portable, exactly stratified patient-group folds.

    ``StratifiedGroupKFold`` changed its shuffled assignment across scikit-learn
    releases and can place no samples of a class in a fold even when every
    class has five or more patient groups.  The paper contract needs the same
    five valid folds on the workstation and CIAI cluster, so perform the small
    group allocation explicitly.  Groups are single-label, sorted by decreasing
    scan count after a seeded shuffle, and greedily assigned to the fold with
    the fewest samples of that class and then the fewest samples overall.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    if y.ndim != 1 or groups.ndim != 1 or len(y) != len(groups) or len(y) == 0:
        raise ValueError("labels and patient groups must be equally sized 1D arrays")

    group_indices: dict[str, np.ndarray] = {}
    group_labels: dict[str, str] = {}
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        labels = np.unique(y[indices])
        if len(labels) != 1:
            raise ValueError(
                f"patient group {group!r} has inconsistent downstream labels"
            )
        group_key = str(group)
        group_indices[group_key] = indices
        group_labels[group_key] = str(labels[0])

    rng = np.random.default_rng(int(seed))
    fold_groups: list[set[str]] = [set() for _ in range(5)]
    fold_sizes = np.zeros(5, dtype=np.int64)
    class_fold_sizes: dict[str, np.ndarray] = {}
    for label in sorted(set(group_labels.values())):
        label_groups = [
            group for group, group_label in group_labels.items()
            if group_label == label
        ]
        if len(label_groups) < 5:
            raise ValueError(
                f"class {label!r} has only {len(label_groups)} patient groups; "
                "five-fold evaluation requires at least five"
            )
        rng.shuffle(label_groups)
        # Python's sort is stable, so the seeded order resolves equal-size ties.
        label_groups.sort(key=lambda group: -len(group_indices[group]))
        per_class = class_fold_sizes.setdefault(label, np.zeros(5, dtype=np.int64))
        for group in label_groups:
            tie_order = rng.permutation(5)
            tie_rank = np.empty(5, dtype=np.int64)
            tie_rank[tie_order] = np.arange(5, dtype=np.int64)
            fold = min(
                range(5),
                key=lambda index: (
                    int(per_class[index]),
                    int(fold_sizes[index]),
                    int(tie_rank[index]),
                ),
            )
            fold_groups[fold].add(group)
            group_size = len(group_indices[group])
            per_class[fold] += group_size
            fold_sizes[fold] += group_size

    all_indices = np.arange(len(y), dtype=np.int64)
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for held_out_groups in fold_groups:
        test_mask = np.isin(groups.astype(str), sorted(held_out_groups))
        test_indices = all_indices[test_mask]
        train_indices = all_indices[~test_mask]
        folds.append((train_indices, test_indices))

    expected_classes = set(y.tolist())
    for train_indices, test_indices in folds:
        train_patients = set(groups[train_indices].tolist())
        test_patients = set(groups[test_indices].tolist())
        if not train_patients.isdisjoint(test_patients):
            raise RuntimeError("patient leakage detected between downstream folds")
        if set(y[train_indices].tolist()) != expected_classes:
            raise ValueError("a downstream training fold is missing one or more classes")
        if set(y[test_indices].tolist()) != expected_classes:
            raise ValueError("a downstream test fold is missing one or more classes")
    return folds


class DownstreamGATClassifier(nn.Module):
    """Two-layer graph-attention classifier with graph-level mean pooling."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        hidden_channels: int = 64,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if min(in_channels, num_classes, hidden_channels, heads) < 1:
            raise ValueError("GAT dimensions and head count must be positive")
        self.dropout = float(dropout)
        self.gat1 = GATConv(
            in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout
        )
        self.gat2 = GATConv(
            hidden_channels * heads,
            hidden_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat2(x, edge_index))
        return self.classifier(global_mean_pool(x, batch))


def _validate_graphs(
    node_features: Sequence,
    edge_indices: Sequence,
    num_samples: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
    if len(node_features) != num_samples or len(edge_indices) != num_samples:
        raise ValueError("node_features and edge_indices must contain one graph per label")
    nodes: List[torch.Tensor] = []
    edges: List[torch.Tensor] = []
    feature_dim = None
    for index, (raw_nodes, raw_edges) in enumerate(zip(node_features, edge_indices)):
        x = torch.as_tensor(raw_nodes, dtype=torch.float32).detach().cpu()
        edge_index = torch.as_tensor(raw_edges, dtype=torch.long).detach().cpu()
        if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
            raise ValueError(f"graph {index} node features must have shape [nodes, features]")
        if not torch.isfinite(x).all():
            raise ValueError(f"graph {index} node features contain NaN or infinity")
        if feature_dim is None:
            feature_dim = int(x.shape[1])
        elif int(x.shape[1]) != feature_dim:
            raise ValueError("all downstream graphs must share one feature dimension")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"graph {index} edge_index must have shape [2, edges]")
        if edge_index.numel() and (
            int(edge_index.min()) < 0 or int(edge_index.max()) >= int(x.shape[0])
        ):
            raise ValueError(f"graph {index} edge_index references a missing node")
        nodes.append(x)
        edges.append(edge_index)
    assert feature_dim is not None
    return nodes, edges, feature_dim


def _array_sha256(tensor: torch.Tensor, *, dtype: str) -> str:
    """Hash a validated tensor using an explicit portable dtype and shape."""
    array = np.ascontiguousarray(tensor.numpy(), dtype=np.dtype(dtype))
    digest = hashlib.sha256()
    digest.update(
        canonical_sha256(
            {"dtype": np.dtype(dtype).str, "shape": list(array.shape)}
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _graph_sha256(nodes: torch.Tensor, edges: torch.Tensor) -> str:
    return canonical_sha256(
        {
            "node_features_sha256": _array_sha256(nodes, dtype="<f4"),
            "edge_index_sha256": _array_sha256(edges, dtype="<i8"),
        }
    )


def _build_graph_stream_identity(
    graph_scan_ids: Sequence[str],
    expected_scan_ids: np.ndarray,
    nodes: Sequence[torch.Tensor],
    edges: Sequence[torch.Tensor],
    *,
    name: str,
) -> dict:
    """Bind every in-memory graph to an independently supplied scan ID."""
    ids = _as_nonempty_strings(graph_scan_ids, f"{name}_scan_ids")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError(f"{name}_scan_ids must be unique")
    if not np.array_equal(ids, expected_scan_ids):
        raise ValueError(
            f"{name}_scan_ids must exactly match scan_ids in graph order"
        )
    records = [
        {"scan_id": str(scan_id), "graph_sha256": _graph_sha256(node, edge)}
        for scan_id, node, edge in zip(ids, nodes, edges)
    ]
    payload = {
        "format": _GRAPH_STREAM_FORMAT,
        "num_graphs": len(records),
        "records": records,
    }
    return {**payload, "sha256": canonical_sha256(payload)}


def _build_hashed_identity(format_name: str, records: list[dict]) -> dict:
    payload = {"format": format_name, "num_records": len(records), "records": records}
    return {**payload, "sha256": canonical_sha256(payload)}


def _validate_positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _build_gat_settings_identity(
    *,
    seed: int,
    hidden_channels: int,
    heads: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    resolved_device: str,
) -> dict:
    """Build the signed settings record without misnaming the fold allocator."""

    payload = {
        "format": DOWNSTREAM_GAT_SETTINGS_FORMAT,
        "seed": int(seed),
        "hidden_channels": int(hidden_channels),
        "heads": int(heads),
        "dropout": float(dropout),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "optimizer": "AdamW",
        "architecture": "two_layer_gat_global_mean_pool",
        "num_folds": 5,
        "fold_allocator": DOWNSTREAM_FOLD_ALLOCATOR_ID,
        "feature_scaling": "training_graph_nodes_per_fold",
        "fold_seed_rule": "seed_plus_zero_based_fold_index",
        "resolved_device": str(resolved_device),
    }
    return {**payload, "sha256": canonical_sha256(payload)}


def _classification_metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        target,
        prediction,
        average="macro",
        labels=np.unique(target),
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


def evaluate_downstream_gat(
    node_features: Sequence,
    edge_indices: Sequence,
    labels: Sequence,
    patient_ids: Sequence[str],
    *,
    scan_ids: Sequence[str],
    graph_scan_ids: Sequence[str],
    cohort_ids: Sequence[str],
    unseen_cohort_manifest_path: str | PathLike[str],
    synthesis_checkpoint_path: str | PathLike[str],
    training_node_features: Sequence | None = None,
    training_edge_indices: Sequence | None = None,
    training_graph_scan_ids: Sequence[str] | None = None,
    hidden_channels: int = 64,
    heads: int = 4,
    dropout: float = 0.2,
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    device: str | torch.device | None = None,
) -> dict:
    """Train and test a fresh GAT in every patient-disjoint fold.

    Feature standardisation is fitted exclusively on the training graphs in each
    fold. Test predictions are produced once after a fixed number of epochs;
    test data are never used for model selection or preprocessing.
    """
    epochs = _validate_positive_integer(epochs, "epochs")
    batch_size = _validate_positive_integer(batch_size, "batch_size")
    hidden_channels = _validate_positive_integer(hidden_channels, "hidden_channels")
    heads = _validate_positive_integer(heads, "heads")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 5:
        raise ValueError("seed must be an integer in [0, 2**32 - 5]")
    numeric_hyperparameters = (learning_rate, weight_decay, dropout)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in numeric_hyperparameters
    ) or not np.isfinite(numeric_hyperparameters).all():
        raise ValueError("GAT hyperparameters must be finite")
    if learning_rate <= 0.0 or weight_decay < 0.0 or not 0.0 <= dropout < 1.0:
        raise ValueError("invalid GAT learning rate, weight decay, or dropout")

    checkpoint, exclusion, checkpoint_sha256 = _load_completed_synthesis_checkpoint(
        synthesis_checkpoint_path
    )
    unseen_manifest_sha256 = _validate_unseen_manifest_arrays(
        scan_ids,
        patient_ids,
        cohort_ids,
        manifest_path=unseen_cohort_manifest_path,
        checkpoint=checkpoint,
    )
    y, groups, cohorts = _validated_protocol_arrays(
        labels, patient_ids, cohort_ids, exclusion
    )
    scans = _as_nonempty_strings(scan_ids, "scan_ids")
    nodes, edges, feature_dim = _validate_graphs(node_features, edge_indices, len(y))
    evaluation_stream = _build_graph_stream_identity(
        graph_scan_ids, scans, nodes, edges, name="graph"
    )
    if (training_node_features is None) != (training_edge_indices is None):
        raise ValueError(
            "training_node_features and training_edge_indices must be supplied together"
        )
    if training_node_features is None:
        if training_graph_scan_ids is not None:
            raise ValueError(
                "training_graph_scan_ids may only be supplied with training graphs"
            )
        fit_nodes, fit_edges = nodes, edges
        training_stream = evaluation_stream
        domain_protocol = "same_stream"
    else:
        if training_graph_scan_ids is None:
            raise ValueError(
                "training_graph_scan_ids are required with cross-domain training graphs"
            )
        fit_nodes, fit_edges, training_feature_dim = _validate_graphs(
            training_node_features, training_edge_indices, len(y)
        )
        if training_feature_dim != feature_dim:
            raise ValueError(
                "training-domain and test-domain graphs must share a feature dimension"
            )
        training_stream = _build_graph_stream_identity(
            training_graph_scan_ids,
            scans,
            fit_nodes,
            fit_edges,
            name="training_graph",
        )
        domain_protocol = "cross_domain"
    classes = sorted(set(y.tolist()))
    label_to_index = {label: index for index, label in enumerate(classes)}
    encoded = np.asarray([label_to_index[label] for label in y], dtype=np.int64)
    folds = _build_folds(y, groups, seed=int(seed))
    run_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    protocol_records = [
        {
            "scan_id": str(scan),
            "patient_id": str(patient),
            "cohort": str(cohort),
            "label": str(label),
        }
        for scan, patient, cohort, label in zip(scans, groups, cohorts, y)
    ]
    protocol_identity = _build_hashed_identity(
        _PROTOCOL_IDENTITY_FORMAT, protocol_records
    )
    gat_settings = _build_gat_settings_identity(
        seed=seed,
        hidden_channels=hidden_channels,
        heads=heads,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        resolved_device=str(run_device),
    )
    graph_streams_payload = {
        "format": _GRAPH_STREAMS_FORMAT,
        "domain_protocol": domain_protocol,
        "evaluation": evaluation_stream,
        "training": training_stream,
    }
    graph_stream_identity = {
        **graph_streams_payload,
        "sha256": canonical_sha256(graph_streams_payload),
    }
    fold_metrics = []
    for fold_index, (train_indices, test_indices) in enumerate(folds):
        fold_seed = int(seed) + fold_index
        random.seed(fold_seed)
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(fold_seed)

        training_nodes = torch.cat([fit_nodes[index] for index in train_indices], dim=0)
        mean = training_nodes.mean(dim=0)
        scale = training_nodes.std(dim=0, unbiased=False)
        scale = torch.where(scale > 1e-8, scale, torch.ones_like(scale))

        def make_graph(index: int, graph_nodes, graph_edges) -> Data:
            return Data(
                x=(graph_nodes[index] - mean) / scale,
                edge_index=graph_edges[index],
                y=torch.tensor([encoded[index]], dtype=torch.long),
            )

        training_graphs = [
            make_graph(int(index), fit_nodes, fit_edges) for index in train_indices
        ]
        test_graphs = [make_graph(int(index), nodes, edges) for index in test_indices]
        generator = torch.Generator().manual_seed(fold_seed)
        training_loader = DataLoader(
            training_graphs,
            batch_size=min(batch_size, len(training_graphs)),
            shuffle=True,
            generator=generator,
        )
        test_loader = DataLoader(
            test_graphs, batch_size=min(batch_size, len(test_graphs)), shuffle=False
        )

        model = DownstreamGATClassifier(
            feature_dim,
            len(classes),
            hidden_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
        ).to(run_device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
        )
        model.train()
        for _ in range(epochs):
            for graph_batch in training_loader:
                graph_batch = graph_batch.to(run_device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(graph_batch.x, graph_batch.edge_index, graph_batch.batch)
                loss = F.cross_entropy(logits, graph_batch.y.view(-1))
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"non-finite downstream loss in fold {fold_index + 1}"
                    )
                loss.backward()
                optimizer.step()

        predictions: List[int] = []
        targets: List[int] = []
        model.eval()
        with torch.no_grad():
            for graph_batch in test_loader:
                graph_batch = graph_batch.to(run_device)
                logits = model(graph_batch.x, graph_batch.edge_index, graph_batch.batch)
                if not torch.isfinite(logits).all():
                    raise RuntimeError(
                        f"non-finite downstream logits in fold {fold_index + 1}"
                    )
                predictions.extend(logits.argmax(dim=-1).cpu().tolist())
                targets.extend(graph_batch.y.view(-1).cpu().tolist())
        result = _classification_metrics(np.asarray(targets), np.asarray(predictions))
        held_out_records = sorted(
            [protocol_records[int(index)] for index in test_indices],
            key=lambda record: record["scan_id"],
        )
        fold_identity_payload = {"held_out_records": held_out_records}
        result.update(
            {
                "fold": fold_index + 1,
                "train_patients": len(set(groups[train_indices].tolist())),
                "test_patients": len(set(groups[test_indices].tolist())),
                "held_out_records": held_out_records,
                "fold_identity_sha256": canonical_sha256(fold_identity_payload),
            }
        )
        fold_metrics.append(result)

    summary = {
        metric: {
            "mean": float(np.mean([fold[metric] for fold in fold_metrics])),
            "std": float(np.std([fold[metric] for fold in fold_metrics], ddof=1)),
        }
        for metric in DOWNSTREAM_METRICS
    }
    return {
        "num_folds": 5,
        "num_classes": len(classes),
        "class_labels": classes,
        "averaging": "macro",
        "domain_protocol": domain_protocol,
        "synthesis_checkpoint_sha256": checkpoint_sha256,
        "unseen_cohort_manifest_sha256": unseen_manifest_sha256,
        "protocol_identity": protocol_identity,
        "graph_stream_identity": graph_stream_identity,
        "gat_settings": gat_settings,
        "folds": fold_metrics,
        "summary": summary,
    }


def compare_downstream_gat_results(
    reference: Mapping,
    comparison: Mapping,
    *,
    alpha: float = 0.05,
) -> dict:
    """Run paired fold-wise Wilcoxon tests with Bonferroni correction.

    The two evaluations must contain the same five held-out patient folds, exact
    ordered graph streams, and GAT settings. This prevents results with changed
    samples, graphs, or randomisation from being mislabeled as paired. Five fold
    pairs cannot reproduce the paper's significance daggers: the smallest exact
    two-sided p-value is 0.0625, or 0.3125 after five-test Bonferroni.
    """
    if not np.isfinite(alpha) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be finite and strictly between zero and one")

    def validated_digest_mapping(
        value: object,
        *,
        name: str,
        required_keys: set[str],
        expected_format: str,
    ) -> Mapping:
        if not isinstance(value, Mapping) or set(value) != required_keys | {"sha256"}:
            raise ValueError(f"{name} has an invalid schema")
        if value.get("format") != expected_format or not _is_sha256(value.get("sha256")):
            raise ValueError(f"{name} has an invalid format or digest")
        payload = {key: value[key] for key in required_keys}
        try:
            payload_sha256 = canonical_sha256(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} payload is not canonical JSON") from exc
        if payload_sha256 != value["sha256"]:
            raise ValueError(f"{name} digest does not match its payload")
        return value

    def validated_protocol_identity(value: object, name: str):
        identity = validated_digest_mapping(
            value,
            name=f"{name} protocol identity",
            required_keys={"format", "num_records", "records"},
            expected_format=_PROTOCOL_IDENTITY_FORMAT,
        )
        records = identity["records"]
        if (
            isinstance(identity["num_records"], bool)
            or not isinstance(identity["num_records"], int)
            or not isinstance(records, list)
            or identity["num_records"] != len(records)
            or len(records) == 0
        ):
            raise ValueError(f"{name} protocol identity has invalid records")
        expected_keys = {"scan_id", "patient_id", "cohort", "label"}
        if any(
            not isinstance(record, Mapping)
            or set(record) != expected_keys
            or any(
                not isinstance(record[key], str) or not record[key]
                for key in expected_keys
            )
            for record in records
        ):
            raise ValueError(f"{name} protocol identity has invalid records")
        scan_ids = [record["scan_id"] for record in records]
        if len(set(scan_ids)) != len(scan_ids):
            raise ValueError(f"{name} protocol identity repeats scan IDs")
        return identity, scan_ids

    def validated_graph_stream(value: object, expected_scan_ids: list[str], name: str):
        stream = validated_digest_mapping(
            value,
            name=name,
            required_keys={"format", "num_graphs", "records"},
            expected_format=_GRAPH_STREAM_FORMAT,
        )
        records = stream["records"]
        if (
            isinstance(stream["num_graphs"], bool)
            or not isinstance(stream["num_graphs"], int)
            or not isinstance(records, list)
            or stream["num_graphs"] != len(records)
        ):
            raise ValueError(f"{name} has invalid graph records")
        if any(
            not isinstance(record, Mapping)
            or set(record) != {"scan_id", "graph_sha256"}
            or not isinstance(record["scan_id"], str)
            or not record["scan_id"]
            or not _is_sha256(record["graph_sha256"])
            for record in records
        ):
            raise ValueError(f"{name} has invalid graph records")
        if [record["scan_id"] for record in records] != expected_scan_ids:
            raise ValueError(f"{name} is not aligned to the protocol scan order")
        return stream

    def validated_graph_streams(
        value: object, expected_scan_ids: list[str], domain_protocol: str, name: str
    ):
        streams = validated_digest_mapping(
            value,
            name=f"{name} graph-stream identity",
            required_keys={"format", "domain_protocol", "evaluation", "training"},
            expected_format=_GRAPH_STREAMS_FORMAT,
        )
        if streams["domain_protocol"] != domain_protocol:
            raise ValueError(f"{name} graph-stream domain protocol is inconsistent")
        evaluation = validated_graph_stream(
            streams["evaluation"], expected_scan_ids, f"{name} evaluation graph stream"
        )
        training = validated_graph_stream(
            streams["training"], expected_scan_ids, f"{name} training graph stream"
        )
        if domain_protocol == "same_stream" and training != evaluation:
            raise ValueError(f"{name} same-stream graph identities differ")
        return streams

    def validated_gat_settings(value: object, name: str):
        required_keys = {
            "format",
            "seed",
            "hidden_channels",
            "heads",
            "dropout",
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "optimizer",
            "architecture",
            "num_folds",
            "fold_allocator",
            "feature_scaling",
            "fold_seed_rule",
            "resolved_device",
        }
        settings = validated_digest_mapping(
            value,
            name=f"{name} GAT settings",
            required_keys=required_keys,
            expected_format=DOWNSTREAM_GAT_SETTINGS_FORMAT,
        )
        for key in ("hidden_channels", "heads", "epochs", "batch_size"):
            _validate_positive_integer(settings[key], f"{name} GAT {key}")
        if (
            isinstance(settings["seed"], bool)
            or not isinstance(settings["seed"], int)
            or not 0 <= settings["seed"] <= 2**32 - 5
        ):
            raise ValueError(f"{name} GAT seed is invalid")
        numeric = [
            settings["dropout"],
            settings["learning_rate"],
            settings["weight_decay"],
        ]
        if (
            any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in numeric)
            or not np.isfinite(numeric).all()
            or not 0.0 <= settings["dropout"] < 1.0
            or settings["learning_rate"] <= 0.0
            or settings["weight_decay"] < 0.0
            or settings["optimizer"] != "AdamW"
            or settings["architecture"] != "two_layer_gat_global_mean_pool"
            or settings["num_folds"] != 5
            or settings["fold_allocator"] != DOWNSTREAM_FOLD_ALLOCATOR_ID
            or settings["feature_scaling"] != "training_graph_nodes_per_fold"
            or settings["fold_seed_rule"] != "seed_plus_zero_based_fold_index"
            or not isinstance(settings["resolved_device"], str)
            or not settings["resolved_device"]
        ):
            raise ValueError(f"{name} GAT settings are invalid")
        return settings

    def validated_result(result: Mapping, name: str):
        if not isinstance(result, Mapping) or result.get("num_folds") != 5:
            raise ValueError(f"{name} downstream result must declare five folds")
        domain_protocol = result.get("domain_protocol")
        if domain_protocol not in {"same_stream", "cross_domain"}:
            raise ValueError(f"{name} downstream domain protocol is invalid")
        protocol, protocol_scan_ids = validated_protocol_identity(
            result.get("protocol_identity"), name
        )
        streams = validated_graph_streams(
            result.get("graph_stream_identity"),
            protocol_scan_ids,
            domain_protocol,
            name,
        )
        settings = validated_gat_settings(result.get("gat_settings"), name)
        folds = result.get("folds")
        if not isinstance(folds, list) or len(folds) != 5:
            raise ValueError(f"{name} downstream result must contain exactly five folds")
        fold_numbers = [fold.get("fold") if isinstance(fold, Mapping) else None for fold in folds]
        if fold_numbers != [1, 2, 3, 4, 5]:
            raise ValueError(f"{name} downstream folds must be ordered uniquely 1..5")
        identities = [fold.get("fold_identity_sha256") for fold in folds]
        if (
            any(not _is_sha256(value) for value in identities)
            or len(set(identities)) != 5
        ):
            raise ValueError(f"{name} downstream fold identities are invalid or repeated")
        held_out_scan_ids = []
        protocol_by_scan = {
            record["scan_id"]: dict(record) for record in protocol["records"]
        }
        for fold in folds:
            records = fold.get("held_out_records")
            if not isinstance(records, list) or not records:
                raise ValueError(f"{name} downstream fold records are invalid")
            if any(not isinstance(record, Mapping) for record in records):
                raise ValueError(f"{name} downstream fold records are invalid")
            if records != sorted(records, key=lambda record: record.get("scan_id", "")):
                raise ValueError(f"{name} downstream fold records are not scan-ordered")
            try:
                fold_sha256 = canonical_sha256({"held_out_records": records})
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{name} downstream fold records are not canonical JSON"
                ) from exc
            if fold_sha256 != fold["fold_identity_sha256"]:
                raise ValueError(f"{name} downstream fold identity does not match its records")
            for record in records:
                if (
                    not isinstance(record, Mapping)
                    or record.get("scan_id") not in protocol_by_scan
                    or dict(record) != protocol_by_scan[record["scan_id"]]
                ):
                    raise ValueError(f"{name} downstream fold records mismatch the protocol")
                held_out_scan_ids.append(record["scan_id"])
        if sorted(held_out_scan_ids) != sorted(protocol_scan_ids) or len(
            set(held_out_scan_ids)
        ) != len(protocol_scan_ids):
            raise ValueError(
                f"{name} downstream folds do not partition the protocol scans exactly"
            )
        checkpoint_sha256 = result.get("synthesis_checkpoint_sha256")
        if not _is_sha256(checkpoint_sha256):
            raise ValueError(f"{name} synthesis checkpoint identity is invalid")
        manifest_sha256 = result.get("unseen_cohort_manifest_sha256")
        if not _is_sha256(manifest_sha256):
            raise ValueError(f"{name} unseen-cohort manifest identity is invalid")
        return (
            folds,
            identities,
            checkpoint_sha256,
            manifest_sha256,
            protocol,
            streams,
            settings,
            domain_protocol,
        )

    (
        reference_folds,
        reference_ids,
        reference_checkpoint,
        reference_manifest,
        reference_protocol,
        reference_streams,
        reference_settings,
        reference_domain,
    ) = validated_result(reference, "reference")
    (
        comparison_folds,
        comparison_ids,
        comparison_checkpoint,
        comparison_manifest,
        comparison_protocol,
        comparison_streams,
        comparison_settings,
        comparison_domain,
    ) = validated_result(comparison, "comparison")
    if reference_checkpoint != comparison_checkpoint:
        raise ValueError("downstream results derive from different synthesis checkpoints")
    if reference_manifest != comparison_manifest:
        raise ValueError("downstream results derive from different unseen-cohort manifests")
    if reference_ids != comparison_ids:
        raise ValueError("downstream results do not use the same held-out patient folds")
    if reference_domain != comparison_domain:
        raise ValueError("downstream results use different domain protocols")
    if reference_protocol != comparison_protocol:
        raise ValueError("downstream results use different scan/label protocols")
    if reference_streams != comparison_streams:
        raise ValueError("downstream results use different or reordered graph streams")
    if reference_settings != comparison_settings:
        raise ValueError("downstream results use different GAT settings or seeds")

    correction_factor = len(DOWNSTREAM_METRICS)
    tests = {}
    for metric in DOWNSTREAM_METRICS:
        try:
            first = np.asarray([float(fold[metric]) for fold in reference_folds])
            second = np.asarray([float(fold[metric]) for fold in comparison_folds])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"downstream results have invalid {metric} values") from exc
        if (
            not np.isfinite(first).all()
            or not np.isfinite(second).all()
            or np.any(first < 0.0)
            or np.any(first > 1.0)
            or np.any(second < 0.0)
            or np.any(second > 1.0)
        ):
            raise ValueError(f"downstream results have invalid {metric} values")
        if np.array_equal(first, second):
            statistic, raw_p = 0.0, 1.0
        else:
            statistic, raw_p = wilcoxon(first, second, alternative="two-sided")
            statistic, raw_p = float(statistic), float(raw_p)
        adjusted_p = min(1.0, raw_p * correction_factor)
        tests[metric] = {
            "statistic": float(statistic),
            "raw_p": raw_p,
            "bonferroni_p": adjusted_p,
            "fold_level_below_alpha": adjusted_p < float(alpha),
            "paper_significance": None,
        }
    return {
        "test": "paired two-sided Wilcoxon signed-rank",
        "pairing_unit": "held-out patient fold",
        "num_pairs": 5,
        "correction": "Bonferroni",
        "correction_factor": correction_factor,
        "alpha": float(alpha),
        "minimum_attainable_raw_p": 0.0625,
        "minimum_attainable_bonferroni_p": 0.3125,
        "paper_significance_reproducible": False,
        "paper_significance_limitation": (
            "The manuscript does not provide subject-level paired outputs; five "
            "fold-level pairs cannot reproduce its significance daggers."
        ),
        "synthesis_checkpoint_sha256": reference_checkpoint,
        "unseen_cohort_manifest_sha256": reference_manifest,
        "protocol_identity_sha256": reference_protocol["sha256"],
        "graph_stream_identity_sha256": reference_streams["sha256"],
        "gat_settings_sha256": reference_settings["sha256"],
        "metrics": tests,
    }


__all__ = [
    "DOWNSTREAM_METRICS",
    "DOWNSTREAM_FOLD_ALLOCATOR_ID",
    "DOWNSTREAM_GAT_SETTINGS_FORMAT",
    "DownstreamGATClassifier",
    "five_fold_downstream_splits",
    "evaluate_downstream_gat",
    "compare_downstream_gat_results",
]
