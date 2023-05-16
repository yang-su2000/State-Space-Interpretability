# State-Space-Interpretability

Investigation of state space model interpretability, credit to lydhr and LancasterCT.

### Interpreting Structured State Space Model

- [Paper Report](XML_project_final.pdf)

**Abstract** 

- The structured state space model(SSM) is a novel model that is good at modeling long-range continuous signals.
This was a hard or even intractable task and taking in long input makes many SOTA models not working.
Therefore, we are interested in interpreting how the SSM-based model is good at long-range tasks.
Since we are a group of three, we worked on two SSM-based models on multiple tasks. First, in the recent series
of work per SSM, the S4 model is an improved version for efficiency. Secondly, the SPACETIME model is a
variant for predicting the time series autoregressively. We train them on our selected datasets, including time
series, text, and images. And we interpret the trained model using SHapley Additive exPlanations (SHAP) and
surrogate model. Finally, we discuss the trustworthiness of the S4 model by conducting a qualitative user study.

**Parallel Repositories**

- [SPACETIME Interpretability GitHub Repo](https://github.com/lydhr/SPACETIME_Finance)
- [Human Study Data and Analysis Github Repo](https://github.com/LancasterCT/XML_human_study)

### SHAP Explainer for S4

- setup
    - go to `state-spaces/`, then follow https://github.com/HazyResearch/state-spaces/tree/main#setup
    - download checkpoints from https://huggingface.co/krandiash/sashimi-release/tree/main/checkpoints, and put into `checkpoints/`
- train
    - mnist (3hrs for a single A6000)
    ```
    python -m train pipeline=mnist dataset.permute=True \
        model=s4 model.n_layers=3 model.d_model=128 model.norm=batch model.prenorm=True wandb=null
    ```
    - cifar (6hrs for a single A6000)
    ```
    python -m train pipeline=cifar
    ```
- interpret
    - put trained checkpoints into `checkpoints/` too, rename to `{model_name}.ckpt`
    - text generation by permutation SHAP
    ```
    python -m generate experiment=lm/s4-wt103 \
        checkpoint_path=checkpoints/s4-wt103.pt \
        n_samples=1 l_sample=10 l_prefix=5 decode=text
    ```
    - mnist classification by partition SHAP
    ```
    python -m checkpoints.evaluate pipeline=mnist dataset.permute=True \
        model=s4 model.n_layers=3 model.d_model=128 model.norm=batch model.prenorm=True \
        checkpoint_path=checkpoints/s4-mnist.ckpt decode=image
    ```
- evaluate layer-wise importance
    - select k, k-1 == number of layers to drop
    - python -m generate experiment=lm/s4-wt103 checkpoint_path=checkpoints/s4-wt103.pt n_samples=1 l_sample=4000 l_prefix=2000 decode=text-auc multiplier=k
