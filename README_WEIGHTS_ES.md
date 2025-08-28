# Documentación del Archivo de Pesos DeepSeek-V3

## Nuevos Campos en config.json

- **model_type**: Especifica el tipo de modelo, que se actualiza a `deepseek_v3` en esta versión.
- **num_nextn_predict_layers**: Indica la cantidad de Módulos de Predicción de Múltiples Tokens (MTP, por sus siglas en inglés). Los pesos de la versión V3 de código abierto incluyen 1 módulo MTP.
- **quantization_config**: Describe la configuración para la cuantización en formato FP8.

## Estructura General de los Pesos

El archivo de pesos de DeepSeek-V3 se compone de dos componentes principales: Pesos del Modelo Principal y Módulos MTP.

### 1. Pesos del Modelo Principal

**Composición:**

- Capas de embedding de entrada/salida y un conjunto completo de 61 capas ocultas tipo Transformer.

**Cantidad de Parámetros:**

- Total de parámetros: 671 mil millones
- Parámetros de activación: 36.7 mil millones (incluyendo 0.9B del Embedding y 0.9B de la capa de salida Head).

**Detalles Estructurales:**

- **Capa de Embedding:**
    - `model.embed_tokens.weight`
- **Capas Ocultas del Transformer:**
    - `model.layers.0` a `model.layers.60`, totalizando `num_hidden_layers` capas.
- **Capa de Salida:**
    - `model.norm.weight`
    - `lm_head.weight`

### 2. Módulos de Predicción de Múltiples Tokens (MTP)

**Composición:**

- Módulos MTP adicionales definidos por el campo `num_nextn_predict_layers`. En este modelo, el valor es 1.

**Cantidad de Parámetros:**

- Parámetros: 11.5 mil millones de parámetros únicos (sin incluir los 0.9B compartidos del Embedding ni los 0.9B del Head).
- Parámetros de activación: 2.4 mil millones (incluyendo los 0.9B compartidos del Embedding y los 0.9B del Head).

**Detalles Estructurales:**

- **embed_tokens:** Comparte parámetros con la capa de Embedding de los pesos del Modelo Principal.
- **enorm & hnorm:** Parámetros RMSNorm requeridos para la decodificación especulativa.
- **eh_proj:** Parámetros para la proyección de reducción dimensional sobre los resultados normalizados.
- **Capa Oculta Adicional tipo Transformer:**
    - `model.layers.61.self_attn` & `mlp` (estructura idéntica a las capas ocultas del Modelo Principal).
- **shared_head:** Comparte parámetros con la capa de salida Head del Modelo Principal.

## Reglas de Carga

- **Pesos del Modelo Principal:** Se cargan mediante el parámetro `num_hidden_layers` en `config.json`.
- **Módulos MTP:** Se cargan mediante el parámetro `num_nextn_predict_layers`, con los IDs de capa que se agregan inmediatamente después de las capas ocultas del Modelo Principal. Por ejemplo:
    - Si `num_hidden_layers = 61` y `num_nextn_predict_layers = 1`, el ID de capa del módulo MTP será 61.

## Documentación de Pesos FP8

DeepSeek-V3 admite de forma nativa el formato de pesos en FP8 con escalado por bloques de 128x128.

### Configuración FP8

El archivo de pesos FP8 introduce un campo `quantization_config` que describe el método de cuantización. A continuación, un ejemplo de configuración:

```json
"quantization_config": {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128]
}
```

**Formato de Cuantización:**

  "activation_scheme": "dynamic",
  "fmt": "e4m3",
  "quant_method": "fp8",
  "weight_block_size": [128, 128]

**Formato de Cuantización:**

Tipo de formato: fp8 y e4m3 (correspondiente a torch.float8_e4m3fn).

Tamaño del bloque de pesos: 128x128.

**Esquema de Cuantización de Activaciones:**

Utiliza cuantización dinámica de activaciones (dynamic).

Método de Descuantización
El archivo de pesos FP8 incluye un campo weight_scale_inv, que almacena la escala de descuantización para cada bloque de pesos.

**Formato de Almacenamiento:**  `float32 Tensor`, almacenado junto con los datos de peso.

**sFórmula de Descuantización:**

Si el bloque de peso no está alineado a 128, se rellena con ceros (padding) hasta 128 antes de calcular la escala. Luego de cuantizar, la parte rellenada se elimina.

El proceso de descuantización se realiza así:  `(128x128 weight block) * weight_scale_inv`.

Mediante la descuantización de los pesos FP8, las operaciones en tiempo de ejecución permiten la cuantización en línea `per-token-per-128-channel`.

---