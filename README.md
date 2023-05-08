# State-Space-Interpretability

Investigation of state space model interpretability.

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