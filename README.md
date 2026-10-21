# Adaptive Step Allocation for Diffusion Sampling

This repository holds the code and reports for my work on adaptive step allocation for diffusion sampling.

It is mainly about one question: at a fixed step budget, how much does tailoring the denoising schedule to each image improve on a shared schedule optimized for the same budget?

The research summary is in [research_summary.pdf](research_summary.pdf).

For more details, see:

- [`step-placement/`](step-placement/): code and the full technical report, [*How Much Is Per-Sample Step Placement Worth at a Fixed Budget?*](step-placement/How%20Much%20Is%20Per-Sample%20Step%20Placement%20Worth%20at%20a%20Fixed%20Budget%20Latest.pdf)
- [`taed/`](taed/): code and the [TAED report](taed/team_5_report.pdf)
