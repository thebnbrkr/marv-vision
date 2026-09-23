"""Verified config shapes, vendored so the shape-reading path works offline and
on a transformers build that does not know `qwen3_vl` yet (< 4.57).

Fetched from https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/raw/main/config.json
on 2026-09-23. Only the fields the shape math needs are kept. If the live config
disagrees, the live one wins -- this is a fallback, not a source of truth.
"""

KNOWN = {
    "Qwen/Qwen3-VL-2B-Instruct": {
        "tie_word_embeddings": True,
        "text_config": {
            "hidden_act": "silu",
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "vocab_size": 151936,
            "tie_word_embeddings": True,
        },
        "vision_config": {
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "depth": 24,
            "num_heads": 16,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 2048,
            "deepstack_visual_indexes": [5, 11, 17],
        },
    },
}
