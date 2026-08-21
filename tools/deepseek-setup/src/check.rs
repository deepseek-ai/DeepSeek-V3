use anyhow::{bail, Context, Result};
use console::style;
use serde::Deserialize;
use std::process::Command;

/// Mirrors whatever check_requirements.py reports, in JSON form.
/// NOTE: this assumes check_requirements.py gains a `--json` output flag
/// (see PR follow-up note below) - it's a small addition to #1584 that
/// makes this tool possible to drive programmatically instead of scraping
/// human-readable stdout.
#[allow(dead_code)] // fields populate once check_requirements.py gains --json output
#[derive(Debug, Deserialize)]
pub struct CheckResult {
    pub python_ok: bool,
    pub os_ok: bool,
    pub cuda_version: Option<String>,
    pub gpu_count: u32,
    pub gpu_vendor: Option<String>, // "nvidia" | "amd" | "ascend" | None
    pub vram_per_gpu_gb: Option<f64>,
    pub total_vram_gb: Option<f64>,
    pub disk_free_gb: Option<f64>,
    pub multi_node_capable: bool,
    pub missing_packages: Vec<String>,
}

/// Runs check_requirements.py and parses its JSON report.
///
/// Falls back to a plain "did it exit 0" check plus a warning if the
/// script doesn't yet support --json (e.g. before that flag lands
/// upstream) so this tool degrades gracefully rather than hard failing.
pub fn run_checker(checker_path: &str) -> Result<CheckResult> {
    println!(
        "{} {}",
        style("Running requirements check:").bold(),
        checker_path
    );

    // NOTE: check_requirements.py does not currently support a --json flag
    // upstream (as of #1584). Run it plain and try to parse structured
    // output opportunistically; if that ever lands, this starts working
    // automatically without needing a flag here.
    let output = Command::new("python3")
        .arg(checker_path)
        .output()
        .with_context(|| format!("failed to execute {checker_path}"))?;

    if !output.status.success() {
        eprintln!(
            "{}",
            style("check_requirements.py reported problems - see output above.").red()
        );
        eprintln!("{}", String::from_utf8_lossy(&output.stdout));
        eprintln!("{}", String::from_utf8_lossy(&output.stderr));
        bail!("requirements check failed; fix the issues above or re-run with --skip-check to override");
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    match serde_json::from_str::<CheckResult>(&stdout) {
        Ok(result) => Ok(result),
        Err(_) => {
            // The checker ran fine but doesn't emit --json yet. Print its
            // human output and fall back to conservative defaults so the
            // rest of the pipeline can still prompt the user manually.
            println!("{stdout}");
            println!(
                "{}",
                style(
                    "Note: check_requirements.py didn't return structured JSON. \
                     Add a --json flag upstream to enable automatic backend selection; \
                     falling back to manual prompts."
                )
                .yellow()
            );
            Ok(CheckResult {
                python_ok: true,
                os_ok: true,
                cuda_version: None,
                gpu_count: 0,
                gpu_vendor: None,
                vram_per_gpu_gb: None,
                total_vram_gb: None,
                disk_free_gb: None,
                multi_node_capable: false,
                missing_packages: vec![],
            })
        }
    }
}