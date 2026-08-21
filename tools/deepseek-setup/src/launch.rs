use crate::backend::Backend;
use anyhow::{bail, Context, Result};
use console::style;
use std::process::Command;

const VENV_DIR: &str = ".deepseek-venv";

/// Launches the chosen backend with sane single-node defaults. This is
/// meant as a working starting point, not a production launch config -
/// flags like tensor-parallel size, port, and max-model-len should be
/// exposed as CLI args in a follow-up once this lands.
pub fn launch(backend: &Backend, model_path: &str) -> Result<()> {
    println!("{} {}", style("Launching:").bold(), backend.name());

    match backend {
        Backend::Demo => {
            println!(
                "{}",
                style(
                    "Demo backend requires multi-node torchrun and interactive flags \
                     specific to your cluster - printing the command instead of running it:"
                )
                .yellow()
            );
            println!(
                "  torchrun --nnodes 2 --nproc-per-node 8 --node-rank $RANK --master-addr $ADDR \\\n    generate.py --ckpt-path {model_path} --config configs/config_671B.json --interactive"
            );
        }
        Backend::Vllm => {
            run_python_module(
                "vllm.entrypoints.openai.api_server",
                &["--model", model_path, "--trust-remote-code", "--port", "8000"],
            )?;
        }
        Backend::Sglang => {
            run_python_module(
                "sglang.launch_server",
                &["--model-path", model_path, "--trust-remote-code", "--port", "30000"],
            )?;
        }
        Backend::Lmdeploy => {
            run(
                &format!("{VENV_DIR}/bin/lmdeploy"),
                &["serve", "api_server", model_path],
            )?;
        }
        Backend::Trtllm => {
            println!(
                "{}",
                style(
                    "TensorRT-LLM launch is engine-build-specific - see NVIDIA's \
                     DeepSeek-V3 TensorRT-LLM docs for the trtllm-build + serve steps."
                )
                .yellow()
            );
        }
    }

    Ok(())
}

fn run_python_module(module: &str, args: &[&str]) -> Result<()> {
    let python = format!("{VENV_DIR}/bin/python");
    let mut full_args = vec!["-m", module];
    full_args.extend(args);
    run(&python, &full_args)
}

fn run(cmd: &str, args: &[&str]) -> Result<()> {
    println!("  $ {cmd} {}", args.join(" "));
    let status = Command::new(cmd)
        .args(args)
        .status()
        .with_context(|| format!("failed to spawn {cmd}"))?;
    if !status.success() {
        bail!("command failed: {cmd} {}", args.join(" "));
    }
    Ok(())
}
