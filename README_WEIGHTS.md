# DeepSeek-V3 Ağırlık Dosyası Dokümantasyonu

## `config.json` İçindeki Yeni Alanlar

- **model_type**: Model türünü belirtir, bu sürümde `deepseek_v3` olarak güncellenmiştir.
- **num_nextn_predict_layers**: Çoklu Token Tahmin (MTP) Modüllerinin sayısını belirtir. Açık kaynaklı V3 ağırlıkları **1 MTP Modülü** içerir.
- **quantization_config**: FP8 kuantizasyonu için yapılandırmayı tanımlar.

---

## Ağırlık Yapısı Genel Bakış

DeepSeek-V3 ağırlık dosyası iki ana bileşenden oluşur: **Ana Model Ağırlıkları** ve **MTP Modülleri**.

### 1. Ana Model Ağırlıkları

- **Bileşenler**:
  - Giriş/çıkış gömme katmanları ve toplam 61 Transformer gizli katmanı.
- **Parametre Sayısı**:
  - Toplam parametreler: **671B**
  - Aktivasyon parametreleri: **36.7B** (0.9B Gömme ve 0.9B Çıkış Kafası dahil).

#### Yapısal Detaylar

- **Gömme Katmanı**:
  - `model.embed_tokens.weight`
- **Transformer Gizli Katmanları**:
  - `model.layers.0` - `model.layers.60`, toplamda `num_hidden_layers` katman.
- **Çıkış Katmanı**:
  - `model.norm.weight`
  - `lm_head.weight`

### 2. Çoklu Token Tahmin (MTP) Modülleri

- **Bileşenler**:
  - `num_nextn_predict_layers` alanı tarafından tanımlanan ek MTP Modülleri. Bu modelde değer **1** olarak ayarlanmıştır.
- **Parametre Sayısı**:
  - **11.5B benzersiz parametre**, (paylaşılan 0.9B Gömme ve 0.9B Çıkış Kafası hariç).
  - Aktivasyon parametreleri: **2.4B** (paylaşılan 0.9B Gömme ve 0.9B Çıkış Kafası dahil).

#### Yapısal Detaylar

- **embed_tokens**: **Ana Model ağırlıklarının Gömme katmanı ile parametreleri paylaşır**.
- **enorm & hnorm**: Spekülatif kod çözme için gerekli olan RMSNorm parametreleri.
- **eh_proj**: Norm sonuçları üzerinde boyut indirgeme projeksiyon parametreleri.
- **Ek Transformer Gizli Katmanı**:
  - `model.layers.61.self_attn & mlp` (Ana Model gizli katmanlarıyla aynı yapıdadır).
- **shared_head**: **Ana Model ağırlıklarının Çıkış Kafası ile parametreleri paylaşır**.

---

### Yükleme Kuralları

- **Ana Model Ağırlıkları**: `config.json` içindeki `num_hidden_layers` parametresi kullanılarak yüklenir.
- **MTP Modülleri**: `num_nextn_predict_layers` parametresi ile yüklenir ve katman kimlikleri Ana Model gizli katmanlarından hemen sonra eklenir. Örneğin:
  - Eğer `num_hidden_layers = 61` ve `num_nextn_predict_layers = 1` ise, MTP Modülünün katman kimliği `61` olur.

---

## FP8 Ağırlık Dokümantasyonu

DeepSeek-V3, 128x128 blok ölçeklendirmesiyle FP8 ağırlık formatını yerel olarak destekler.

### FP8 Yapılandırması

FP8 ağırlık dosyası, kuantizasyon yöntemini tanımlayan bir `quantization_config` alanı içerir. Örnek yapılandırma aşağıda verilmiştir:

```json
"quantization_config": {
  "activation_scheme": "dynamic",
  "fmt": "e4m3",
  "quant_method": "fp8",
  "weight_block_size": [128, 128]
}
```

- **Kuantizasyon Formatı**:
  - Format türü: `fp8` ve `e4m3` (karşılığı `torch.float8_e4m3fn`).
  - Ağırlık blok boyutu: `128x128`.
- **Aktivasyon Kuantizasyon Şeması**:
  - Dinamik aktivasyon kuantizasyonu kullanır (`dynamic`).

### De-kuantizasyon Yöntemi

FP8 ağırlık dosyası, her ağırlık bloğu için de-kuantizasyon ölçeğini depolayan `weight_scale_inv` alanını içerir.

- **Depolama Formatı**: `float32 Tensor`, ağırlık verileriyle birlikte saklanır.
- **De-kuantizasyon Formülü**:
  - Ağırlık bloğu 128’e hizalanmamışsa, önce 128’e sıfır dolgu yapılır, ardından ölçek hesaplanır. Kuantizasyondan sonra dolgu kısmı kaldırılır.
  - De-kuantizasyon işlemi şu şekilde gerçekleştirilir: `(128x128 ağırlık bloğu) * weight_scale_inv`.

FP8 ağırlıklarının de-kuantizasyonu sayesinde, çalışma zamanı işlemleri **token başına 128 kanal granülerliği** ile çevrimiçi kuantizasyona olanak tanır.

---
```  
Bu çeviri, hem teknik doğruluğu hem de Markdown uyumluluğunu koruyarak çevrilmiştir.
```  
