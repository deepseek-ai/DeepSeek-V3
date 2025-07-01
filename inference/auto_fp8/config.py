# Adapted from AutoFP8 (deprecated project)
import re
from typing import List, Optional, Tuple


class BaseQuantizeConfig:
    """Configuration for model quantization.

    Args:
        quant_method: Type/precision of quantization method to use.
            At the moment, this is just "fp8" which specifically means
            the fp8_e4m3 format in pytorch.
        activation_scheme: Choice of either "dynamic" or "static" quantization
            of activations. If "static", then calibration samples are required
            during quantization to produce accurate per-tensor scales for
            activations of Linear modules.
        ignore_patterns: List of patterns used to ignore layers. If a string
            starts with "re:", then everything afterwards is used as python
            regex style matching i.e. re.search(), for each Linear layer.
            By default, "re:.*lm_head" is included to ignore the embedding
            Linear layer usually at the end of decoder LLMs
        kv_cache_quant_targets: Tuple of Linear module names to target for
            calibration of the output scales for KV cache quantization.
            Usually, these should be `("k_proj", "v_proj")`.
    """

    def __init__(
        self,
        quant_method: str = "fp8",
        activation_scheme: str = "static",
        ignore_patterns: List[str] = ["re:.*lm_head"],
        kv_cache_quant_targets: Optional[Tuple[str]] = None,
    ):
        if quant_method != "fp8":
            raise ValueError("Only FP8 quantization is supported.")
        if activation_scheme not in ["static", "dynamic"]:
            raise ValueError(
                "Invalid activation_scheme. Choose either 'static' or 'dynamic'."
            )
        self.quant_method = quant_method
        self.activation_scheme = activation_scheme
        self.re_ignore_patterns = ignore_patterns
        self.ignore_patterns = [
            re.compile(regex_pat, re.VERBOSE) for regex_pat in ignore_patterns
        ]
        self.kv_cache_quant_targets = kv_cache_quant_targets
        self.ignored_layers = []
