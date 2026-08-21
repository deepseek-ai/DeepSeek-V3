mod backend;
mod check;
mod convert;
mod launch;
mod setup;

use anyhow::Result;
use clap::Parser;
use console::style;

/// deepseek-setup: one command to go from a bare Linux box to a running
/// DeepSeek-V3 inference server.
///
/// Pipeline: check_requirements.py -> pick backend -> create venv & install
/// pinned deps -> convert FP8->BF16 if needed -> launch server.
#[derive(Parser, Debug)]
#[command(name = "deepseek-setup", version, about)]
struct Cli {
    /// Path to the deepseek-ai/DeepSeek-V3 repo checkout.
    #[arg(long, default_value = ".")]
    repo: String,

    /// Path to the check_requirements.py script (defaults to <repo>/check_requirements.py).
    #[arg(long)]
    checker: Option<String>,

    /// Force a specific backend instead of prompting: demo, vllm, sglang, lmdeploy, trtllm.
    #[arg(long)]
    backend: Option<backend::Backend>,

    /// Path to the downloaded HF checkpoint (FP8 weights).
    #[arg(long)]
    hf_ckpt_path: Option<String>,

    /// Where to write converted / prepared weights.
    #[arg(long, default_value = "./deepseek-v3-prepared")]
    save_path: String,

    /// Skip the requirements check step (not recommended).
    #[arg(long)]
    skip_check: bool,

    /// Only check + report; don't install anything or launch a server.
    #[arg(long)]
    dry_run: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    println!("{}", style("== DeepSeek-V3 setup ==").bold().cyan());

    // 1. Requirements check (delegates to the existing Python checker so
    //    there's a single source of truth for hardware/software validation).
    let check_result = if cli.skip_check {
        println!("{}", style("Skipping requirements check (--skip-check).").yellow());
        None
    } else {
        let checker_path = cli
            .checker
            .clone()
            .unwrap_or_else(|| format!("{}/check_requirements.py", cli.repo.trim_end_matches('/')));
        Some(check::run_checker(&checker_path)?)
    };

    // 2. Backend selection - either forced via --backend or chosen
    //    interactively based on what the check found (GPU vendor, VRAM,
    //    single vs multi-node).
    let chosen = backend::choose_backend(cli.backend, check_result.as_ref())?;
    println!(
        "{} {}",
        style("Selected backend:").bold(),
        style(chosen.name()).green()
    );

    if cli.dry_run {
        println!("{}", style("Dry run: stopping before install/convert/launch.").yellow());
        return Ok(());
    }

    // 3. Environment setup: venv + pinned deps for the chosen backend.
    setup::prepare_environment(&chosen, &cli.repo)?;

    // 4. Weight conversion (FP8 -> BF16) only if the backend needs it and
    //    the user gave us a checkpoint path.
    if let Some(hf_path) = &cli.hf_ckpt_path {
        convert::maybe_convert(&chosen, &cli.repo, hf_path, &cli.save_path)?;
    } else {
        println!(
            "{}",
            style("No --hf-ckpt-path given; skipping weight conversion/prep step.").yellow()
        );
    }

    // 5. Launch with sane defaults for the chosen backend.
    launch::launch(&chosen, &cli.save_path)?;

    Ok(())
}
