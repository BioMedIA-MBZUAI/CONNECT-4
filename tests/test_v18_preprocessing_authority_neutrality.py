import copy
import hashlib
import json
import re

import pytest

from architecture_contract import (
    SYNTHESIS_ARCHITECTURE_CONTRACT_SHA256,
    SYNTHESIS_ARCHITECTURE_SCHEMA,
    SYNTHESIS_LAUNCH_ADMISSION_SCHEMA,
    TOKEN_CODEC_STATE_VERSION,
    TRAINING_CHECKPOINT_FORMAT,
    require_current_synthesis_architecture,
    synthesis_architecture_contract,
)


def _canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def test_v18_requires_separate_current_independent_go_preprocessing_authority():
    contract = synthesis_architecture_contract()

    assert contract["preprocessing_authority_contract"] == (
        "separate-current-authority-with-independent-terminal-GO-required"
    )
    assert contract["preprocessing_authority_required_for_model_admission_roles"] == [
        "exploratory",
        "production",
    ]
    assert (
        contract["preprocessing_authority_version_preauthorization"]
        == "forbidden-none"
    )
    assert "v9_preprocessing_authority_independently_admissible" not in contract
    assert (
        "v9_synthesis_model_gate_training_inference_authority_accepted"
        not in contract
    )


def test_v18_signed_architecture_blesses_no_concrete_preprocessing_version():
    contract = synthesis_architecture_contract()
    preprocessing_clauses = {
        key: value
        for key, value in contract.items()
        if "preprocess" in key.casefold()
    }

    assert set(preprocessing_clauses) == {
        "preprocessing_authority_contract",
        "preprocessing_authority_required_for_model_admission_roles",
        "preprocessing_authority_version_preauthorization",
    }
    serialized = json.dumps(preprocessing_clauses, sort_keys=True)
    assert re.search(r"(?i)(?<![a-z0-9])v\d+(?!\d)", serialized) is None
    assert SYNTHESIS_ARCHITECTURE_CONTRACT_SHA256 == _canonical_sha256(contract)


@pytest.mark.parametrize("terminal_no_go_version", [9, 10])
def test_v18_rejects_terminal_no_go_preprocessing_version_claims(
    terminal_no_go_version,
):
    claim = copy.deepcopy(synthesis_architecture_contract())
    claim[
        f"v{terminal_no_go_version}_preprocessing_authority_independently_admissible"
    ] = True

    with pytest.raises(RuntimeError, match="pre-v18"):
        require_current_synthesis_architecture(claim, _canonical_sha256(claim))


@pytest.mark.parametrize("version", [9, 10, 11])
def test_v18_rejects_any_embedded_preprocessing_version_preauthorization(version):
    claim = copy.deepcopy(synthesis_architecture_contract())
    claim["preprocessing_authority_version_preauthorization"] = f"v{version}"

    with pytest.raises(RuntimeError, match="pre-v18"):
        require_current_synthesis_architecture(claim, _canonical_sha256(claim))


def test_v18_issues_fresh_model_identities():
    assert TRAINING_CHECKPOINT_FORMAT == "connect4_iteration_exact_v18"
    assert SYNTHESIS_ARCHITECTURE_SCHEMA == (
        "connect4-synthesis-architecture-contract-v18"
    )
    assert SYNTHESIS_LAUNCH_ADMISSION_SCHEMA == (
        "connect4-synthesis-v18-launch-admission-v1"
    )
    assert TOKEN_CODEC_STATE_VERSION == 18
