# Resumen del Proyecto DeepSeek-V3

## Estructura del Proyecto

1. **README.md**: Proporciona una descripción general del modelo, su arquitectura, pre-entrenamiento, post-entrenamiento, resultados de evaluación, y cómo ejecutar el modelo localmente.

2. **README_WEIGHTS.md**: Documenta la estructura de los archivos de pesos del modelo, incluyendo los nuevos campos en `config.json`, la estructura de los pesos del modelo principal y los módulos de predicción de múltiples tokens (MTP).

3. **README_WEIGHTS_ES.md**: Versión en español de `README_WEIGHTS.md`.

4. **README_ES.md**: Versión en español de `README.md`.

5. **LICENSE-MODEL**: Licencia del modelo DeepSeek, que incluye restricciones de uso y condiciones de distribución.

6. **LICENSE-CODE**: Licencia MIT para el código del proyecto.

7. **.gitignore**: Lista de archivos y directorios que deben ser ignorados por Git.

8. **requirements.txt**: Lista de dependencias necesarias para ejecutar el proyecto.

## Código Principal

### inference/model.py

Define la arquitectura del modelo Transformer, incluyendo capas de atención, capas de feed-forward, y mecanismos de mezcla de expertos (MoE). Algunas clases y funciones clave incluyen:

- **ModelArgs**: Clase de datos que define los argumentos y parámetros del modelo.
- **ParallelEmbedding**: Capa de embedding con soporte para paralelismo distribuido.
- **Linear, ColumnParallelLinear, RowParallelLinear**: Capas lineales personalizadas con soporte para pesos cuantizados y paralelismo.
- **RMSNorm**: Normalización de capa basada en la raíz cuadrada media.
- **MLA**: Capa de atención multi-cabeza.
- **MLP**: Perceptrón multicapa utilizado como capa de feed-forward.
- **Gate**: Mecanismo de enrutamiento para MoE.
- **Expert**: Capa de experto para MoE.
- **MoE**: Módulo de mezcla de expertos.
- **Block**: Bloque Transformer que combina atención y capas de feed-forward.
- **Transformer**: Modelo Transformer completo con embeddings posicionales, múltiples capas y proyección de salida.

### inference/kernel.py

Define funciones y kernels de Triton para operaciones de cuantización y des-cuantización, así como para multiplicación de matrices en precisión FP8.

- **act_quant_kernel**: Kernel para cuantización de activaciones.
- **weight_dequant_kernel**: Kernel para des-cuantización de pesos.
- **fp8_gemm_kernel**: Kernel para multiplicación de matrices en precisión FP8.

### inference/generate.py

Proporciona funciones para generar texto utilizando el modelo Transformer.

- **sample**: Función para muestrear un token a partir de logits utilizando escalado de temperatura.
- **generate**: Función para generar nuevos tokens basados en tokens de entrada proporcionados.
- **main**: Función principal para cargar el modelo y realizar generación de texto interactiva o por lotes.

### inference/fp8_cast_bf16.py

Convierte pesos en formato FP8 a BF16 y guarda los pesos convertidos.

### inference/convert.py

Convierte y guarda archivos de puntos de control del modelo en un formato específico.

## Configuraciones

- **configs/config_671B.json**: Configuración del modelo con 671 mil millones de parámetros.
- **configs/config_236B.json**: Configuración del modelo con 236 mil millones de parámetros.
- **configs/config_16B.json**: Configuración del modelo con 16 mil millones de parámetros.

Este resumen proporciona una visión general del funcionamiento del proyecto DeepSeek-V3, destacando las principales clases, funciones y archivos involucrados en la implementación y ejecución del modelo.
