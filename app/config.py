import json
import os
from pathlib import Path

from dotenv import load_dotenv


# Load variables from .env
load_dotenv()


# =====================================================================
# PROJECT ROOT
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =====================================================================
# BUSINESS LOGIC CONFIGURATION
# =====================================================================

METRIC_CONFIG_PATH = (
    Path(__file__).resolve().parent / "metrics_config.json"
)

try:
    with METRIC_CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        METRIC_EXTRACTION_CONFIG = json.load(f)

except FileNotFoundError as e:
    raise RuntimeError(
        f"Metric configuration file not found: "
        f"{METRIC_CONFIG_PATH}"
    ) from e

except json.JSONDecodeError as e:
    raise RuntimeError(
        f"Invalid metrics_config.json: {e}"
    ) from e


# Basic validation of metric configuration
if not isinstance(METRIC_EXTRACTION_CONFIG, dict):
    raise RuntimeError(
        "metrics_config.json must contain a JSON object."
    )

for key, config in METRIC_EXTRACTION_CONFIG.items():

    if not isinstance(config, dict):
        raise RuntimeError(
            f"Metric '{key}' must be a JSON object."
        )

    required_fields = {
        "name",
        "query",
        "type",
    }

    missing_fields = required_fields - config.keys()

    if missing_fields:
        raise RuntimeError(
            f"Metric '{key}' is missing required fields: "
            f"{sorted(missing_fields)}"
        )

    if config["type"] not in {
        "financial_metric",
        "insight",
    }:
        raise RuntimeError(
            f"Metric '{key}' has invalid type: "
            f"{config['type']}"
        )


# =====================================================================
# SYSTEM & RUNTIME SETTINGS
# =====================================================================

class Settings:

    # --------------------------------------------------------------
    # Project
    # --------------------------------------------------------------

    PROJECT_ROOT = str(PROJECT_ROOT)

    # --------------------------------------------------------------
    # Storage
    # --------------------------------------------------------------

    RAW_PDF_DIR = os.getenv(
        "RAW_PDF_DIR",
        str(PROJECT_ROOT / "data" / "raw"),
    )

    MARKDOWN_DIR = os.getenv(
        "MARKDOWN_DIR",
        str(PROJECT_ROOT / "data" / "markdown"),
    )

    CHROMA_DIR = os.getenv(
        "CHROMA_DIR",
        str(PROJECT_ROOT / "data" / "chroma_db"),
    )

    # --------------------------------------------------------------
    # Chunking
    # --------------------------------------------------------------

    CHUNK_SIZE = int(
        os.getenv(
            "CHUNK_SIZE",
            500,
        )
    )

    CHUNK_OVERLAP = int(
        os.getenv(
            "CHUNK_OVERLAP",
            50,
        )
    )

    # --------------------------------------------------------------
    # LLM
    # --------------------------------------------------------------

    GROQ_API_KEY = os.getenv(
        "GROQ_API_KEY"
    )

    LLM_MODEL_NAME = os.getenv(
        "LLM_MODEL_NAME",
        "qwen/qwen3.8-27b",
    )

    # Number of retries performed internally by ChatGroq
    # for transient API failures.
    LLM_MAX_RETRIES = int(
        os.getenv(
            "LLM_MAX_RETRIES",
            2,
        )
    )

    # Number of times the complete extraction operation
    # should be attempted by the document-processing worker.
    EXTRACTION_MAX_RETRIES = int(
        os.getenv(
            "EXTRACTION_MAX_RETRIES",
            3,
        )
    )

    # --------------------------------------------------------------
    # Vector Database
    # --------------------------------------------------------------

    COLLECTION_NAME = os.getenv(
        "COLLECTION_NAME",
        "investor_reports",
    )

    DEFAULT_TOP_K = int(
        os.getenv(
            "DEFAULT_TOP_K",
            3,
        )
    )

    # --------------------------------------------------------------
    # Upload Behaviour
    # --------------------------------------------------------------

    ALLOW_DUPLICATE_UPLOADS = (
        os.getenv(
            "ALLOW_DUPLICATE_UPLOADS",
            "False",
        ).lower()
        == "true"
    )


# Single application-wide settings instance
settings = Settings()
