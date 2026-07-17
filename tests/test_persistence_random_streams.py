import json

import pytest

from distributed_inference.persistence.random_streams import RandomStreamSpec


@pytest.fixture
def random_stream_spec() -> RandomStreamSpec:
    return RandomStreamSpec(
        algorithm="numpy.pcg64",
        seed=42,
        stream_id="replicate-0",
        schema_version="1",
    )


def test_random_stream_spec_rehydrates_reproducible_generator(
    random_stream_spec: RandomStreamSpec,
) -> None:
    first = random_stream_spec.to_generator().normal()
    second = random_stream_spec.to_generator().normal()

    assert first == pytest.approx(second)


def test_random_stream_spec_manifest_is_json_serializable(
    random_stream_spec: RandomStreamSpec,
) -> None:
    payload = random_stream_spec.to_manifest()

    assert isinstance(json.dumps(payload), str)


def test_random_stream_spec_manifest_does_not_serialize_live_state(
    random_stream_spec: RandomStreamSpec,
) -> None:
    payload = random_stream_spec.to_manifest()

    assert "state" not in payload
