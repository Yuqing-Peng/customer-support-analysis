"""All the knobs in one place: column names, how many clusters to try, batch size,
and which Claude model to use."""
from dataclasses import dataclass
from typing import Tuple

from .insights import DEFAULT_MODEL


@dataclass
class PipelineConfig:
    random_state: int = 42

    # Ticket schema (column names after normalisation: lower_snake_case)
    id_col: str = "ticket_id"
    text_cols: Tuple[str, ...] = ("subject", "body")
    created_col: str = "created_at"
    resolved_col: str = "resolved_at"
    min_text_chars: int = 15

    # Text features
    max_features: int = 5000
    svd_components: int = 50

    # Batch K-means
    k_min: int = 3
    k_max: int = 10
    batch_size: int = 256
    n_epochs: int = 3
    silhouette_sample: int = 2000

    # Cloud API insights
    use_api: bool = True
    model: str = DEFAULT_MODEL
    samples_per_cluster: int = 5
