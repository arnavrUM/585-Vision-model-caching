import sys
import types

if "vllm" not in sys.modules:
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = object
    fake_vllm.SamplingParams = object
    sys.modules["vllm"] = fake_vllm

if "faiss" not in sys.modules:
    fake_faiss = types.ModuleType("faiss")

    class _IndexFlatIP:
        def __init__(self, dim):
            self.dim = dim

        def add(self, vectors):
            return None

        def search(self, vector, k):
            import numpy as np

            scores = np.zeros((1, k), dtype="float32")
            indices = -np.ones((1, k), dtype="int64")
            return scores, indices

    def _normalize_L2(vectors):
        return None

    fake_faiss.IndexFlatIP = _IndexFlatIP
    fake_faiss.normalize_L2 = _normalize_L2
    sys.modules["faiss"] = fake_faiss

if "sentence_transformers" not in sys.modules:
    fake_st = types.ModuleType("sentence_transformers")

    class _SentenceTransformer:
        def __init__(self, *args, **kwargs):
            self.dim = 384

        def encode(self, texts, normalize_embeddings=True):
            import numpy as np

            return np.zeros((len(texts), self.dim), dtype="float32")

    fake_st.SentenceTransformer = _SentenceTransformer
    sys.modules["sentence_transformers"] = fake_st

from experiment2.specs import ExperimentSpec


def test_experiment_spec_consumes_preset_values() -> None:
    spec = ExperimentSpec.from_dict({"name": "intern-demo", "preset": "internvl3.5-2b"})
    assert spec.model.startswith("OpenGVLab/InternVL3_5-2B")
    assert spec.embedding_hook == "prompt_vision"
    assert spec.embedding_layers == ["prompt:384:0.8", "vision:512:0.82"]
    assert spec.preset == "internvl3.5-2b"
    assert spec.enable_semantic_text_cache is True
    assert spec.enable_exact_text_cache is True


def test_experiment_spec_preset_can_be_overridden() -> None:
    spec = ExperimentSpec.from_dict(
        {
            "name": "override",
            "preset": "internvl3.5-2b",
            "model": "custom/model",
            "embedding_layers": "prompt:768:0.9",
        }
    )
    assert spec.model == "custom/model"
    assert spec.embedding_layers == ["prompt:768:0.9"]


def test_experiment_spec_inherits_default_preset() -> None:
    defaults = {"preset": "internvl3.5-2b", "max_samples": 32}
    spec = ExperimentSpec.from_dict({"name": "defaults-only"}, defaults=defaults)
    assert spec.model.startswith("OpenGVLab/InternVL3_5-2B")
    assert spec.max_samples == 32
    assert spec.preset == "internvl3.5-2b"
    assert spec.enable_semantic_text_cache is True
    assert spec.enable_exact_text_cache is True
