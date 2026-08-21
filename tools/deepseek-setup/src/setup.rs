use crate::backend::Backend;
use anyhow::{bail, Context, Result};
use console::style;
use std::path::Path;
use std::process::Command;

const VENV_DIR: &str = ".deepseek-venv";

pub fn prepare_environment(backend: &Backend, repo_path: &str) -> Result<()> {
    println!("{}", style("Setting up environment...").bold());

    if !Path::new(VENV_DIR).exists() {
        run("python3", &["-m", "venv", VENV_DIR])?;
    } else {
        println!("{}", style(format!("Reusing existing venv at {VENV_DIR}")).dim());
    }

    let pip = format!("{VENV_DIR}/bin/pip");
    run(&pip, &["install", "--upgrade", "pip"])?;

    if *backend == Backend::Demo {
        // The demo's own requirements.txt is the source of truth; still
        // pass the pinned floor versions in case requirements.txt drifts.
        let req_file = format!("{}/inference/requirements.txt", repo_path.trim_end_matches('/'));
        if Path::new(&req_file).exists() {
            run(&pip, &["install", "-r", &req_file])?;
        } else {
            let reqs = backend.pip_requirements();
            let mut args = vec!["install"];
            args.extend(reqs.iter().copied());
            run(&pip, &args)?;
        }
    } else {
        let reqs = backend.pip_requirements();
        if !reqs.is_empty() {
            let mut args = vec!["install"];
            args.extend(reqs.iter().copied());
            run(&pip, &args)?;
        } else if *backend == Backend::Trtllm {
            println!(
                "{}",
                style(
                    "TensorRT-LLM installs from NVIDIA's custom branch, not pip. \
                     See: https://github.com/NVIDIA/TensorRT-LLM - this step is manual for now."
                )
                .yellow()
            );
        }
    }

    println!("{}", style("Environment ready.").green());
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
