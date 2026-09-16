"""
Central configuration for MemORAI reproduction.
All hyperparameters referenced in the paper (Sections 3, 4) live here so
ablations (Section 4.3) can be toggled from one place / CLI overrides.
"""
import os
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    api_base: str = os.environ.get("MEMORAI_LLM_API_BASE", "http://localhost:8000/v1")
    api_key: str = os.environ.get("MEMORAI_LLM_API_KEY", "EMPTY")
    temperature: float = 0.0
    max_tokens: int = 512       
    timeout: int = 120
    retries: int = 3
    enable_thinking: bool = os.environ.get("MEMORAI_LLM_ENABLE_THINKING", "0") == "1"


@dataclass
class JudgeConfig:
    # GPT-4o as judge (Appendix C.7). Uses the real OpenAI API.
    model_name: str = os.environ.get("MEMORAI_JUDGE_MODEL", "gpt-4o")
    api_key: str = os.environ.get("OPENAI_API_KEY", "")
    temperature: float = 0.0
    max_tokens: int = 10


@dataclass
class EmbeddingConfig:
    # Contriever used for fair comparison with baselines (paper §4.1)
    model_name: str = os.environ.get("MEMORAI_EMBED_MODEL", "facebook/contriever")
    device: str = os.environ.get("MEMORAI_EMBED_DEVICE", "cpu")
    max_length: int = 512
    batch_size: int = 16


@dataclass
class RetrievalConfig:
    top_k_seed: int = 3          # top-k seed nodes per aspect (segments/entities/relations)
    top_m_turns: int = 3         # top-m turns returned after ranking
    damping: float = 0.85        # d in Eq. 2
    pagerank_iters: int = 20
    convergence_eps: float = 1e-6


@dataclass
class AblationConfig:
    """Flags matching Tables 4-8 ablations."""
    use_topic_segmentation: bool = True      # w/o Topic Seg ablation
    use_selective_filtering: bool = True     # w/o Selective ablation
    use_dynamic_weighting: bool = True       # Uniform (w=1) ablation
    use_subgraph_retrieval: bool = True      # Full Graph ablation
    use_triplet_enrichment: bool = True      # Turn only ablation


@dataclass
class PathConfig:
    data_dir: str = "data"
    prompts_dir: str = "prompts"
    cache_dir: str = ".cache"
    output_dir: str = "outputs"


@dataclass
class MemoraiConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    paths: PathConfig = field(default_factory=PathConfig)


DEFAULT_CONFIG = MemoraiConfig()
