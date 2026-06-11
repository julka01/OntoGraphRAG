"""Shared retrieval configuration constants."""

# Configurable parameters for different question types
RAG_CONFIG = {
    "statistical": {
        "default_max_chunks": 100,  # More chunks for statistical analysis
        "threshold_floor": 0.05,
        "threshold_factor": 0.03
    },
    "semantic": {
        "default_max_chunks": 15,  # Fewer chunks for focused semantic questions
        "threshold_floor": 0.08,
        "threshold_ceiling": 0.15,
        "threshold_boost": 0.02
    },
    "generic": {
        "default_max_chunks": 20,  # Default chunk count
        "default_threshold": 0.08
    }
}

