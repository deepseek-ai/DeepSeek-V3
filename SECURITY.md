# Security Policy

## Supported Versions

This repository distributes DeepSeek-V3 model weights (checkpoints) together with the official inference code. Security fixes are provided for:

| Artifact | Supported |
| --- | --- |
| Latest official checkpoint release | :white_check_mark: |
| Inference code on `main` | :white_check_mark: |
| Older checkpoints | :x: — please move to the latest release |
| Fine-tuned / derived weights, third-party forks and integrations | :x: — out of scope |

## Reporting a Vulnerability

**Please do not open a public issue for security problems.**

Report vulnerabilities privately by email to **service@deepseek.com**, the official contact address listed in section 9 of this repository's README. Put `[SECURITY]` at the start of the subject line so the report is routed correctly.

> Note: GitHub private vulnerability reporting is currently not enabled for this repository, so the peer-to-peer advisory form is not available. Email is the supported channel.

If the issue concerns the hosted service (platform.deepseek.com, chat.deepseek.com) or an API endpoint rather than the code or weights in this repository, say so in the first paragraph; the report will be forwarded to the responsible team.

Helpful information to include:

- Affected artifact: checkpoint name/tag, or the commit SHA of the inference code, or the dependency version;
- Reproduction steps or a proof of concept (minimal and self-contained if possible);
- Impact assessment (e.g. code execution, model extraction, denial of service);
- Whether the issue is already public anywhere;
- Preferred name/handle for credit, or "anonymous".

Please do not include secrets, personal data or third-party content in the report.

## What to Expect

- We will acknowledge your report as soon as practicable and aim to provide an initial assessment in a reasonable time.
- We may ask for additional information or a reproduction; please keep the report private while it is being investigated.
- We will agree on a coordinated disclosure date with you. As a rule of thumb we ask for up to 90 days before public disclosure, and we will tell you if a fix is expected to take longer.
- Credit will be given with your preferred name/handle when a fix is published, unless you ask to remain anonymous.

## Scope Notes

Security-sensitive areas of this repository include, but are not limited to:

- Checkpoint loading and deserialization paths (weights, tokenizer, conversion scripts);
- FP8 block-wise quantization and checkpoint conversion code;
- Inference dependencies pinned in `inference/requirements.txt` and related environment files.

Please do not test against the hosted service or its infrastructure; reports about the hosted platform are handled by the platform team as described above.
