# DeepSeek-V3 Weight File Documentation

## New Fields in `config.json`

- **model_type**: Specifies the model type, which is now set to `deepseek_v3` in this release.
- **num_nextn_predict_layers**: Defines the number of Multi-Token Prediction (MTP) Modules. The open-sourced V3 weights contain **1 MTP Module**.
- **quantization_config**: Details the configuration for FP8 quantization.

---

## Weight File Structure Overview

The DeepSeek-V3 weight file is divided into two primary components: **Main Model Weights** and **MTP Modules**.

### 1. Main Model Weights

- **Composition**:
  - Includes input/output embedding layers and a full set of 61 Transformer hidden layers.
- **Parameter Count**:
  - Total parameters: **671B**
  - Activation parameters: **36.7B** (which includes 0.9B for Embedding and 0.9B for the Output Head).

#### Structural Details

- **Embedding Layer**:
  - `model.embed_tokens.weight`
- **Transformer Hidden Layers**:
  - From `model.layers.0` to `model.layers.60`, which correspond to `num_hidden_layers` layers.
- **Output Layer**:
  - `model.norm.weight`
  - `lm_head.weight`

### 2. Multi-Token Prediction (MTP) Modules

- **Composition**:
  - These modules are determined by the `num_nextn_predict_layers` parameter. In this model, the value is set to 1.
- **Parameter Count**:
  - Parameters: **11.5B unique parameters** (excluding the shared 0.9B Embedding and 0.9B Output Head).
  - Activation parameters: **2.4B** (including the shared 0.9B Embedding and 0.9B Output Head).

#### Structural Details

- **embed_tokens**: **Shares parameters** with the Main Model’s Embedding layer.
- **enorm & hnorm**: RMSNorm parameters used for speculative decoding.
- **eh_proj**: Parameters used for dimensionality reduction of the normalized outputs.
- **Additional Transformer Hidden Layer**:
  - `model.layers.61.self_attn & mlp` (these are structured the same as the Main Model hidden layers).
- **shared_head**: **Shares parameters** with the Output Head of the Main Model.

---

### Layer Loading Rules

- **Main Model Weights**: These are loaded according to the `num_hidden_layers` field in `config.json`.
- **MTP Modules**: These are loaded using the `num_nextn_predict_layers` field, with MTP layer IDs appended directly after the Main Model’s hidden layers. For example:
  - With `num_hidden_layers = 61` and `num_nextn_predict_layers = 1`, the MTP Module layer ID will be `61`.

---

## FP8 Weight Documentation

DeepSeek-V3 natively supports the FP8 weight format with 128x128 block scaling.

### FP8 Configuration

The FP8 weight file introduces a `quantization_config` field, which defines the quantization method. Below is an example of the configuration:

```json
"quantization_config": {
  "activation_scheme": "dynamic",
  "fmt": "e4m3",
  "quant_method": "fp8",
  "weight_block_size": [128, 128]
}
```

- **Quantization Format**:
  - Format type: `fp8` and `e4m3` (aligned with `torch.float8_e4m3fn`).
  - Weight block size: `128x128`.
- **Activation Quantization Scheme**:
  - Uses dynamic activation quantization (`dynamic`).

### Dequantization Method

The FP8 weight file includes a `weight_scale_inv` field, which stores the dequantization scale for each weight block.

- **Storage Format**: Stored as a `float32 Tensor`, alongside the weight data.
- **Dequantization Formula**:
  - If a weight block is not aligned to 128, it is zero-padded to 128 before calculating the scale. The padded portion is discarded after quantization.
  - Dequantization is performed using the formula: `(128x128 weight block) * weight_scale_inv`.

This dequantization process enables runtime operations to apply online quantization on a per-token, per-128-channel basis.
