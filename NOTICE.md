# Attribution

This repository is built on **LoRA[AB]** and **LoRA+**. We acknowledge their authors' contributions and publicly released implementations.

The initialization comparisons and decoder experiment pipeline build on [Leopold1423/non_zero_lora-icml25](https://github.com/Leopold1423/non_zero_lora-icml25), the code for *Beyond Zero Initialization: Investigating the Impact of Non-Zero Initialization on LoRA Fine-Tuning Dynamics* (Shiwei Li et al., ICML 2025). Its nonzero A/B initialization and subtraction baseline is represented here by `AB_reset`.

The LoRA+ optimizer/trainer and several GLUE utilities derive from [nikhilgsh/loraplus](https://github.com/nikhilgsh/loraplus), the code for *LoRA+: Efficient Low Rank Adaptation of Large Models* (Soufiane Hayou, Nikhil Ghosh, and Bin Yu, ICML 2024); its MIT notice is preserved in [LICENSE](LICENSE).

The instruction-tuning format and generation benchmark evaluation derive from [AGI-Edgerunners/LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters), licensed under Apache 2.0. The GLUE training entry point also derives from the Hugging Face Transformers GLUE example under Apache 2.0. The full Apache license is included in [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt).

Model weights and datasets retain their upstream terms. They are not bundled with the public source snapshot. The benchmark downloader records the upstream revision and includes its code and data license files beside the downloaded data.

The README figures are author-provided assets from the NeurIPS manuscript and existing experiment outputs. Their sources and settings are recorded in [assets/README.md](assets/README.md).
