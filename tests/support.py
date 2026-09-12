import json
import hashlib
from pathlib import Path
import re

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from data.dataset import Connect4Dataset
from data.patch_descriptions import (
    KNOWN_FUNCTIONAL_CONNECTIVITY,
    PATCH_DESCRIPTION_SCHEMA_VERSION,
    build_patch_description,
)
from data.provenance import canonical_sha256, sha256_file
from data.spatial_contract import PATCH_COORDINATE_CONTRACT
from graphs.hypergraph import HypergraphBuilder
from normative.atrophy import ASEG_TO_NORM_LABEL
from preprocessing.source_acquisition_identity import (
    FMRIPREP_RUNTIME_IDENTITY_PURPOSE,
    FMRIPREP_RUNTIME_IDENTITY_SCHEMA,
    SOURCE_ACQUISITION_PURPOSE,
    SOURCE_ACQUISITION_SCHEMA,
    STRUCTURAL_SOURCE_AUTHORITY_PURPOSE,
    STRUCTURAL_SOURCE_AUTHORITY_SCHEMA,
    SYNTHSEG_PROVENANCE_SCHEMA,
    canonical_sha256 as source_canonical_sha256,
    load_source_acquisition_identity,
    load_structural_source_authority,
    snapshot_bids_input_inventory,
    structural_source_binding,
)


def _signed_source_record(value):
    record = dict(value)
    record["record_sha256"] = source_canonical_sha256(record)
    return record


def _source_artifact(path: Path):
    source = path.resolve(strict=True)
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "size_bytes": source.stat().st_size,
    }


def write_source_acquisition_fixture(
    root: Path,
    *,
    scan_id: str,
    t1_source: Path,
    synthseg_source: Path,
):
    """Create a fully signed raw T1/BOLD/SynthSeg identity for unit tests."""
    match = re.search(r"(?:^|_)sub-([A-Za-z0-9]+)(?:_|$)", scan_id)
    if match is None:
        raise ValueError("test scan_id requires a BIDS subject entity")
    participant = match.group(1)
    bids_root = (root / "raw_bids").resolve()
    anat_dir = bids_root / f"sub-{participant}" / "anat"
    func_dir = bids_root / f"sub-{participant}" / "func"
    anat_dir.mkdir(parents=True, exist_ok=True)
    func_dir.mkdir(parents=True, exist_ok=True)
    raw_t1 = anat_dir / f"sub-{participant}_run-1_T1w.nii.gz"
    raw_bold = func_dir / f"sub-{participant}_task-rest_run-1_bold.nii.gz"
    source_t1_image = nib.load(str(t1_source))
    source_mask_image = nib.load(str(synthseg_source))
    nib.save(source_t1_image, raw_t1)
    bold_values = np.ones((*source_t1_image.shape, 2), dtype=np.float32)
    nib.save(nib.Nifti1Image(bold_values, source_t1_image.affine), raw_bold)
    (bids_root / "dataset_description.json").write_text(
        json.dumps(
            {
                "Name": "CONNECT-4 certified unit-test BIDS input",
                "BIDSVersion": "1.10.0",
            }
        ),
        encoding="utf-8",
    )
    raw_bold_metadata = raw_bold.with_name(
        raw_bold.name.removesuffix(".nii.gz") + ".json"
    )
    slice_count = int(source_t1_image.shape[2])
    raw_bold_metadata.write_text(
        json.dumps(
            {
                "TaskName": "rest",
                "RepetitionTime": 3.0,
                "SliceTiming": [
                    3.0 * index / slice_count for index in range(slice_count)
                ],
            }
        ),
        encoding="utf-8",
    )
    raw_synthseg = (root / "source_masks" / f"{scan_id}_synthseg.nii.gz").resolve()
    raw_synthseg.parent.mkdir(parents=True, exist_ok=True)
    nib.save(source_mask_image, raw_synthseg)

    producer = {
        "name": "SynthSeg",
        "version": "2.0",
        "source_revision": "a" * 40,
        "model_sha256": "b" * 64,
        "container_image": "docker://freesurfer/synthseg:test",
        "container_manifest_sha256": "c" * 64,
    }
    provenance_path = (
        root / "source_masks" / f"{scan_id}_synthseg_provenance.json"
    ).resolve()
    provenance = _signed_source_record(
        {
            "schema": SYNTHSEG_PROVENANCE_SCHEMA,
            "scan_id": scan_id,
            "producer": producer,
            "input_t1": _source_artifact(raw_t1),
            "source_t1_sha256": sha256_file(raw_t1),
            "output_mask": _source_artifact(raw_synthseg),
        }
    )
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    provenance_artifact = _source_artifact(provenance_path)
    provenance_artifact["record_sha256"] = provenance["record_sha256"]
    structural_authority_path = (
        root / "structural_source_authority" / f"{scan_id}.json"
    ).resolve()
    structural_authority_path.parent.mkdir(parents=True, exist_ok=True)
    structural_authority = _signed_source_record(
        {
            "schema": STRUCTURAL_SOURCE_AUTHORITY_SCHEMA,
            "purpose": STRUCTURAL_SOURCE_AUTHORITY_PURPOSE,
            "scan_id": scan_id,
            "raw_t1": _source_artifact(raw_t1),
            "synthseg": {
                "mask": _source_artifact(raw_synthseg),
                "provenance": provenance_artifact,
                "producer": producer,
            },
        }
    )
    structural_authority_path.write_text(
        json.dumps(structural_authority), encoding="utf-8"
    )
    structural_authority_sha256 = sha256_file(structural_authority_path)
    admitted_structural, structural_evidence = load_structural_source_authority(
        structural_authority_path,
        expected_sha256=structural_authority_sha256,
    )
    identity_path = (root / "source_identity" / f"{scan_id}.json").resolve()
    identity_path.parent.mkdir(parents=True, exist_ok=True)
    identity = _signed_source_record(
        {
            "schema": SOURCE_ACQUISITION_SCHEMA,
            "purpose": SOURCE_ACQUISITION_PURPOSE,
            "scan_id": scan_id,
            "participant_label": participant,
            "bids_root": str(bids_root),
            "bids_entities": {
                "raw_t1": {
                    "run": "1",
                    "sub": participant,
                    "suffix": "T1w",
                },
                "raw_bold": {
                    "run": "1",
                    "sub": participant,
                    "suffix": "bold",
                    "task": "rest",
                },
            },
            "bids_input_inventory": snapshot_bids_input_inventory(
                bids_root,
                participant=participant,
                raw_t1=raw_t1,
                raw_bold=raw_bold,
            ),
            "raw_t1": _source_artifact(raw_t1),
            "raw_bold": _source_artifact(raw_bold),
            "synthseg": {
                "mask": _source_artifact(raw_synthseg),
                "provenance": provenance_artifact,
                "producer": producer,
            },
        }
    )
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    identity_sha256 = sha256_file(identity_path)
    admitted, _evidence = load_source_acquisition_identity(
        identity_path,
        expected_sha256=identity_sha256,
    )
    return {
        "identity_path": identity_path,
        "identity_sha256": identity_sha256,
        "identity": admitted,
        "binding": structural_source_binding(admitted_structural, structural_evidence),
        "structural_authority_path": structural_authority_path,
        "structural_authority_sha256": structural_authority_sha256,
        "structural_authority": admitted_structural,
        "raw_t1": raw_t1,
        "raw_bold": raw_bold,
        "raw_bold_metadata": raw_bold_metadata,
        "raw_synthseg": raw_synthseg,
        "synthseg_provenance": provenance_path,
    }


def write_fmriprep_runtime_fixture(
    root: Path,
    *,
    executable: Path,
    version: str = "25.2.5",
):
    """Create an externally SHA-pinned runtime identity for wrapper tests."""

    resolved_executable = executable.resolve(strict=True)
    runtime_payload = (root / "fmriprep-runtime-payload.bin").resolve()
    runtime_payload.write_bytes(b"pinned unit-test runtime payload\n")
    runtime_artifacts = sorted(
        (_source_artifact(resolved_executable), _source_artifact(runtime_payload)),
        key=lambda value: str(value["path"]),
    )
    identity = _signed_source_record(
        {
            "schema": FMRIPREP_RUNTIME_IDENTITY_SCHEMA,
            "purpose": FMRIPREP_RUNTIME_IDENTITY_PURPOSE,
            "fmriprep_version": version,
            "runtime_kind": "unit-test-standalone",
            "complete_runtime_inventory": True,
            "executable": _source_artifact(resolved_executable),
            "runtime_artifacts": runtime_artifacts,
        }
    )
    path = (root / "fmriprep-runtime-identity.json").resolve()
    path.write_text(json.dumps(identity), encoding="utf-8")
    return {
        "path": path,
        "sha256": sha256_file(path),
        "identity": identity,
        "runtime_payload": runtime_payload,
    }


def write_structural_sources(
    root: Path,
    scan_ids=("sub-01_run-1",),
    shape=(4, 4, 4),
    radiomics_dim=3,
):
    affine = np.diag([3.0, 3.0, 3.0, 1.0])
    (root / "T1").mkdir(parents=True, exist_ok=True)
    (root / "Masks").mkdir(parents=True, exist_ok=True)
    radiomics_rows = []
    normative_rows = []
    for subject_index, scan_id in enumerate(scan_ids):
        t1 = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        t1 = t1 + subject_index
        labels = np.zeros(shape, dtype=np.int16)
        labels[: shape[0] // 2] = 2
        labels[shape[0] // 2 :] = 17
        t1_path = root / "T1" / f"{scan_id}_T1.nii.gz"
        mask_path = root / "Masks" / f"{scan_id}_mask.nii.gz"
        nib.save(nib.Nifti1Image(t1, affine), t1_path)
        nib.save(nib.Nifti1Image(labels, affine), mask_path)

        for roi_index, (slug, label_id) in enumerate(Connect4Dataset.ROI_SPECS):
            row = {
                "scan_id": scan_id,
                "structure_id": label_id,
                "structure": slug,
            }
            for feature_index in range(radiomics_dim):
                row[f"radiomics_feature_{feature_index}"] = float(
                    1000 * subject_index + 10 * roi_index + feature_index
                )
            radiomics_rows.append(row)

        for roi_index, label_id in enumerate(sorted(ASEG_TO_NORM_LABEL)):
            normative_rows.append(
                {
                    "PatientID": scan_id,
                    "StructureID": label_id,
                    "MeasuredVolume": 1000.0 + roi_index,
                    "Age": 60.0,
                    "Sex": 1,
                    "FieldStrength": 0,
                    "Manufacturer": 3,
                    "ICV": 1_500_000.0,
                }
            )

        anatcl_dir = root / "AnatCL" / scan_id
        anatcl_dir.mkdir(parents=True, exist_ok=True)
        embedding_hashes = {}
        for roi_index, (slug, _label_id) in enumerate(Connect4Dataset.ROI_SPECS):
            embedding_path = anatcl_dir / f"{scan_id}_{slug}.pth"
            torch.save(
                torch.full((512,), float(subject_index + roi_index / 100.0)),
                embedding_path,
            )
            embedding_hashes[slug] = sha256_file(embedding_path)
        provenance = {
            "schema": Connect4Dataset.ANATCL_PROVENANCE_SCHEMA,
            "scan_id": scan_id,
            "source_sha256": {
                "t1w": sha256_file(t1_path),
                "segmentation": sha256_file(mask_path),
            },
            "model": {
                **Connect4Dataset.ANATCL_MODEL_CONTRACT,
            },
            "extraction": dict(Connect4Dataset.ANATCL_EXTRACTION_CONTRACT),
            "cat12_input": {
                "schema": Connect4Dataset.CAT12_INPUT_PROVENANCE_SCHEMA,
                "pipeline": dict(Connect4Dataset.CAT12_PREPROCESSING_CONTRACT),
                "authority_manifest_sha256": hashlib.sha256(
                    f"authority:{scan_id}".encode()
                ).hexdigest(),
                "scan_record_sha256": hashlib.sha256(
                    f"record:{scan_id}".encode()
                ).hexdigest(),
                "cat12_mwp1_sha256": hashlib.sha256(
                    f"mwp1:{scan_id}".encode()
                ).hexdigest(),
                "segmentation_in_cat12_vbm_sha256": hashlib.sha256(
                    f"labels:{scan_id}".encode()
                ).hexdigest(),
            },
            "embedding_sha256": embedding_hashes,
        }
        (anatcl_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

    radiomics_path = root / "all_radiomics.csv"
    pd.DataFrame(radiomics_rows).to_csv(radiomics_path, index=False)
    radiomics_provenance = {
        "schema": Connect4Dataset.RADIOMICS_PROVENANCE_SCHEMA,
        "extractor": {
            **Connect4Dataset.PYRADIOMICS_EXTRACTOR_CONTRACT,
            "runtime_lock_sha256": "6" * 64,
            "runtime_tree_sha256": "7" * 64,
        },
        "source_sha256": {
            scan_id: {
                "t1w": sha256_file(root / "T1" / f"{scan_id}_T1.nii.gz"),
                "segmentation": sha256_file(root / "Masks" / f"{scan_id}_mask.nii.gz"),
            }
            for scan_id in scan_ids
        },
        "feature_columns": sorted(
            column
            for column in pd.DataFrame(radiomics_rows).columns
            if column.startswith("radiomics_")
        ),
        "row_count": len(radiomics_rows),
        "output_sha256": sha256_file(radiomics_path),
    }
    (root / "radiomics_provenance.json").write_text(
        json.dumps(radiomics_provenance, indent=2)
    )
    normative_path = root / "normative_inputs.csv"
    pd.DataFrame(normative_rows).to_csv(normative_path, index=False)
    dwi_path = root / "dwi.csv"
    np.savetxt(dwi_path, np.eye(32, dtype=np.float32), delimiter=",")
    brainiac_path = root / "BrainIAC.ckpt"
    brainiac_path.write_bytes(b"paper-test-brainiac-checkpoint")
    brainiac_source_files = {
        "connect4/models/brainiac_wrapper.py": sha256_file(
            Path(__file__).resolve().parents[1] / "models" / "brainiac_wrapper.py"
        ),
        "brainiac/load_brainiac.py": "a" * 64,
        "brainiac/models/vit.py": "b" * 64,
    }
    brainiac_fingerprint = {
        "implementation": "BrainIAC",
        "checkpoint_sha256": sha256_file(brainiac_path),
        "source_files_sha256": brainiac_source_files,
        "source_files_fingerprint_sha256": canonical_sha256(brainiac_source_files),
        "adapter": {
            "input_channels": 1,
            "resize_shape": [96, 96, 96],
            "resize_mode": "trilinear",
            "align_corners": False,
            "output": "CLS token embedding",
            "embedding_dim": 768,
        },
    }
    modernbert_path = root / "modernbert"
    modernbert_path.mkdir()
    (modernbert_path / "config.json").write_text('{"hidden_size": 768}\n')
    (modernbert_path / "tokenizer_config.json").write_text("{}\n")
    modernbert_fingerprint = {
        "implementation": "Clinical-ModernBERT",
        "upstream_model_id": "Simonlee711/Clinical_ModernBERT",
        "model_name": str(modernbert_path),
        "revision": "a" * 40,
        "local_files_sha256": {
            "config.json": sha256_file(modernbert_path / "config.json"),
            "tokenizer_config.json": sha256_file(
                modernbert_path / "tokenizer_config.json"
            ),
        },
    }
    modernbert_fingerprint["fingerprint_sha256"] = canonical_sha256(
        modernbert_fingerprint
    )
    return {
        "normative": normative_path,
        "dwi": dwi_path,
        "brainiac": brainiac_path,
        "brainiac_fingerprint": brainiac_fingerprint,
        "modernbert": modernbert_path,
        "modernbert_fingerprint": modernbert_fingerprint,
    }


def write_certified_graph_cache(
    root: Path,
    sources: dict,
    scan_id: str = "sub-01_run-1",
    *,
    target_shape=(4, 4, 4),
    patch_size=(2, 4, 4),
) -> Path:
    """Create a small, internally reproducible production-cache fixture."""
    source_dataset = Connect4Dataset(
        root_dir=str(root),
        patch_size=patch_size,
        target_shape=target_shape,
        dwi_matrix_path=str(sources["dwi"]),
        normative_csv_path=str(sources["normative"]),
    )
    segmentation, affine = source_dataset._load_nifti(
        root / "Masks" / f"{scan_id}_mask.nii.gz", is_mask=True
    )
    expected_patches = int(
        np.prod([size // patch for size, patch in zip(target_shape, patch_size)])
    )
    distributions = [
        source_dataset._compute_patch_distribution(segmentation, index)
        for index in range(expected_patches)
    ]
    normative_subject = source_dataset.scan_to_normative_subject[scan_id]
    descriptions = [
        build_patch_description(
            patch_idx=index,
            center_mm=source_dataset._compute_patch_center_mm(
                index, segmentation=segmentation, affine=affine
            ),
            distribution=distribution,
            id_to_slug=source_dataset.ID_TO_SLUG,
            normative_index=source_dataset.normative_index,
            patient_id=normative_subject,
        )
        for index, distribution in enumerate(distributions)
    ]

    precomputed = root / "precomputed"
    graphs = precomputed / "graphs"
    image_emb = precomputed / "image_emb"
    text_emb = precomputed / "modernbert_emb"
    hypergraphs = precomputed / "hypergraphs"
    for folder in (graphs, image_emb, text_emb, hypergraphs):
        folder.mkdir(parents=True, exist_ok=True)
    image_nodes = np.arange(expected_patches * 768, dtype=np.float32).reshape(
        expected_patches, 768
    )
    mask_nodes = image_nodes + 0.25
    roi_nodes = source_dataset[source_dataset.scan_ids.index(scan_id)][
        "roi_embeddings"
    ].numpy()
    np.save(graphs / f"{scan_id}_image_nodes.npy", image_nodes)
    np.save(graphs / f"{scan_id}_mask_nodes.npy", mask_nodes)
    np.save(graphs / f"{scan_id}_roi_nodes.npy", roi_nodes)
    np.save(image_emb / f"{scan_id}_image_patch_embeddings.npy", image_nodes)
    np.save(text_emb / f"{scan_id}_mask_patch_embeddings.npy", mask_nodes)
    distribution_path = hypergraphs / f"{scan_id}_patch_distributions.json"
    description_path = hypergraphs / f"{scan_id}_patch_descriptions.json"
    distribution_path.write_text(json.dumps(distributions, indent=2))
    description_path.write_text(json.dumps(descriptions, indent=2))
    builder = HypergraphBuilder(expected_patches, Connect4Dataset.NUM_ROIS)
    hyperedge_index, hyperedge_weights = builder.build_hyperedges(
        distributions,
        source_dataset.structure_to_roi_idx,
        torch.device("cpu"),
    )
    np.save(
        hypergraphs / f"{scan_id}_hyperedge_index.npy",
        hyperedge_index.numpy(),
    )
    np.save(
        hypergraphs / f"{scan_id}_hyperedge_weights.npy",
        hyperedge_weights.numpy(),
    )

    recorded_sources = {
        "dataset": source_dataset.source_fingerprint(scan_id),
        "brainiac": sources["brainiac_fingerprint"],
        "modernbert": sources["modernbert_fingerprint"],
        "scalers": {"enabled": False},
        "target_shape": list(target_shape),
        "patch_size": list(patch_size),
        "patch_description_schema_version": PATCH_DESCRIPTION_SCHEMA_VERSION,
        "patch_coordinate_contract": dict(PATCH_COORDINATE_CONTRACT),
        "functional_connectivity_sha256": canonical_sha256(
            KNOWN_FUNCTIONAL_CONNECTIVITY
        ),
    }
    artifact_paths = [
        graphs / f"{scan_id}_image_nodes.npy",
        image_emb / f"{scan_id}_image_patch_embeddings.npy",
        text_emb / f"{scan_id}_mask_patch_embeddings.npy",
        graphs / f"{scan_id}_mask_nodes.npy",
        graphs / f"{scan_id}_roi_nodes.npy",
        hypergraphs / f"{scan_id}_hyperedge_index.npy",
        hypergraphs / f"{scan_id}_hyperedge_weights.npy",
        distribution_path,
        description_path,
    ]
    normative_hash = canonical_sha256(
        {
            "workbook_sha256": source_dataset.normative_workbook_sha256,
            "subject_rows": source_dataset.normative_source_rows[normative_subject],
        }
    )
    metadata = {
        "scan_id": scan_id,
        "spatial_shape": list(target_shape),
        "patch_size": list(patch_size),
        "num_patches": expected_patches,
        "num_rois": Connect4Dataset.NUM_ROIS,
        "patch_description_schema_version": PATCH_DESCRIPTION_SCHEMA_VERSION,
        "patch_coordinate_contract": dict(PATCH_COORDINATE_CONTRACT),
        "structural_grid_geometry": recorded_sources["dataset"][
            "structural_grid_geometry"
        ],
        "patch_text_components": [
            "roi_distribution_and_patch_center",
            "known_functional_connectivity",
            "subject_specific_normative_volume",
        ],
        "normative_context_sha256": normative_hash,
        "potvin_supported_roi_ids": sorted(source_dataset.POTVIN_ROI_IDS),
        "source_fingerprint": recorded_sources,
        "source_fingerprint_sha256": canonical_sha256(recorded_sources),
        "artifact_sha256": {path.name: sha256_file(path) for path in artifact_paths},
    }
    (hypergraphs / f"{scan_id}_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    return precomputed
