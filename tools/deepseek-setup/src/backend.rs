use crate::check::CheckResult;
use anyhow::Result;
use clap::ValueEnum;
use dialoguer::{theme::SimpleTheme, Select};

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum Backend {
    /// The bare DeepSeek-Infer demo (torchrun + generate.py). Good for
    /// verifying an environment works before committing to a serving stack.
    Demo,
    Vllm,
    Sglang,
    Lmdeploy,
    /// NVIDIA-only, BF16 + INT4/8 today, FP8 in progress upstream.
    Trtllm,
}

impl Backend {
    pub fn name(&self) -> &'static str {
        match self {
            Backend::Demo => "DeepSeek-Infer Demo",
            Backend::Vllm => "vLLM",
            Backend::Sglang => "SGLang",
            Backend::Lmdeploy => "LMDeploy",
            Backend::Trtllm => "TensorRT-LLM",
        }
    }

    /// Pinned/floor package requirements per README + each project's docs.
    /// Kept centralized so setup.rs and check.rs stay in sync with one
    /// source of truth as these versions move.
    pub fn pip_requirements(&self) -> Vec<&'static str> {
        match self {
            Backend::Demo => vec![
                "torch==2.4.1",
                "triton==3.0.0",
                "transformers==4.46.3",
                "safetensors==0.4.5",
            ],
            Backend::Vllm => vec!["vllm>=0.6.6"],
            Backend::Sglang => vec!["sglang[all]"],
            Backend::Lmdeploy => vec!["lmdeploy"],
            Backend::Trtllm => vec![], // installed from NVIDIA's custom branch, not pip
        }
    }

    pub fn requires_nvidia_only(&self) -> bool {
        matches!(self, Backend::Trtllm)
    }

    pub fn needs_fp8_to_bf16_conversion(&self) -> bool {
        // The demo path and TensorRT-LLM's BF16-only mode both need the
        // conversion script; vLLM/SGLang/LMDeploy consume FP8 natively.
        matches!(self, Backend::Demo | Backend::Trtllm)
    }
}

/// Picks a backend: explicit CLI flag wins, otherwise filter by what the
/// hardware check ruled out, then prompt interactively among what's left.
pub fn choose_backend(forced: Option<Backend>, check: Option<&CheckResult>) -> Result<Backend> {
    if let Some(b) = forced {
        return Ok(b);
    }

    let mut candidates = vec![
        Backend::Demo,
        Backend::Vllm,
        Backend::Sglang,
        Backend::Lmdeploy,
        Backend::Trtllm,
    ];

    if let Some(c) = check {
        if let Some(vendor) = &c.gpu_vendor {
            if vendor != "nvidia" {
                candidates.retain(|b| !b.requires_nvidia_only());
            }
        }
    }

    let labels: Vec<&str> = candidates.iter().map(|b| b.name()).collect();
    let idx = Select::with_theme(&SimpleTheme)
        .with_prompt("Which serving backend would you like to set up?")
        .items(&labels)
        .default(1.min(labels.len().saturating_sub(1))) // default to vLLM if present
        .interact()?;

    Ok(candidates[idx])
}