use crate::backend::Backend;
use anyhow::{bail, Context, Result};
use console::style;
use std::process::Command;

/// Runs fp8_cast_bf16.py (and, for the demo path, convert.py's sharding
/// step) only when the chosen backend actually needs it. vLLM/SGLang/
/// LMDeploy consume the native FP8 checkpoint directly, so this is a
/// no-op for those - saves a very slow, disk-heavy step nobody asked for.
pub fn maybe_convert(
    backend: &Backend,
    repo_path: &str,
    hf_ckpt_path: &str,
    save_path: &str,
) -> Result<()> {
    if !backend.needs_fp8_to_bf16_conversion() {
        println!(
            "{}",
            style(format!(
                "{} consumes FP8 weights natively - skipping conversion.",
                backend.name()
            ))
            .dim()
        );
        return Ok(());
    }

    println!("{}", style("Converting FP8 weights to BF16...").bold());
    let script = format!("{}/inference/fp8_cast_bf16.py", repo_path.trim_end_matches('/'));
    run(
        "python3",
        &[
            &script,
            "--input-fp8-hf-path",
            hf_ckpt_path,
            "--output-bf16-hf-path",
            save_path,
        ],
    )?;

    if *backend == Backend::Demo {
        // The demo additionally needs the model-parallel sharding step.
        println!("{}", style("Sharding weights for the demo's torchrun launch...").bold());
        let convert_script = format!("{}/inference/convert.py", repo_path.trim_end_matches('/'));
        run(
            "python3",
            &[
                &convert_script,
                "--hf-ckpt-path",
                save_path,
                "--save-path",
                save_path,
                "--n-experts",
                "256",
                "--model-parallel",
                "16",
            ],
        )?;
    }

    println!("{}", style("Conversion complete.").green());
    Ok(())
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
