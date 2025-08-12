# A Real-Time Self-Tuning Moderator Framework for Adversarial Prompt Detection
A moderator framework for LLM security. This repository holds source code and benchmarking datasets used during the research investigation of RTST.

Please be aware that some datasets may contain sensitive or potentially harmful information. Harmful prompts in no way reflect our values.

## Installation
Clone the repository, then use `pip install -r requirements.txt` to install required dependencies.

Replication of the research methodology can be conducted by running the Python files with names corresponding to each benchmark; however, results may vary slightly due to the use of commercial models. Modification of the `pipeline_base.py` file is the best way to utilize the framework for a new task.

## Attribution
When utilizing this code or research in your work, we'd appreciate it if you would include the following citation:
```
@article{zhang2025realtimeselftuningmoderatorframework,
      author={Ivan Zhang},
      title={A Real-Time, Self-Tuning Moderator Framework for Adversarial Prompt Detection}, 
      journal = {arXiv e-prints},
      year={2025},
      eprint={2508.07139},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2508.07139}
}
```
