print("CrewAI pipeline running...")

import subprocess
import os
from datetime import datetime

def log(msg):
    print(f"[CrewAI] {msg}")

def run_chronograph():
    log("Running chronograph_engine.ps1...")
    subprocess.run(
        [
            "powershell.exe",
            "-ExecutionPolicy", "Bypass",
            "C:\\RobdoeDeepSeekN-V3\\chronograph_engine.ps1"
        ],
        shell=True
    )

def git_cmd(args):
    subprocess.run(["git"] + args, shell=True)

def auto_commit_cycle():
    log("Staging changes...")
    git_cmd(["add", "."])

    msg = f"Auto: CrewAI + Chronograph cycle @ {datetime.now().isoformat()}"
    log(f"Committing with message: {msg}")
    git_cmd(["commit", "-m", msg])

    log("Pushing to origin/main...")
    git_cmd(["push", "origin", "main"])

def main():
    log("CrewAI pipeline engaged.")
    run_chronograph()
    auto_commit_cycle()
    log("CrewAI cycle complete.")

if __name__ == "__main__":
    main()

