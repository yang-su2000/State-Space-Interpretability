import argparse
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchaudio

import hydra
from omegaconf import OmegaConf
from torch.distributions import Categorical
from tqdm.auto import tqdm

from src import utils
from src.dataloaders.audio import mu_law_decode
from src.models.baselines.wavenet import WaveNetModel
from train import SequenceLightningModule

import shap
import scipy as sp
import matplotlib.pyplot as plt

def test_step(model):
    B, L = 2, 64
    x = torch.ones(B, L, dtype=torch.long).to('cuda')

    # Forward
    batch = (x, None)
    y, _, _ = model(batch) # Forward pass expects a batch which has both x and y (inputs and targets)

    # Step
    model._reset_state(batch, device='cuda')
    ys = []
    for x_ in torch.unbind(x, dim=-1):
        y_ = model.step(x_)
        ys.append(y_)
    ys = torch.stack(ys, dim=1)

    print(torch.norm(y-ys))

    breakpoint()

@torch.inference_mode()
def generate(
    model,
    batch,
    tau=1.0,
    l_prefix=0,
    T=None,
    debug=False,
    top_p=1.0,
    benchmark=False,
    return_logprobs=False,
    multiplier=1,
):

    x, _, *_ = batch # (B, L)
    try:
        x = x.to('cuda')
    except:
        x = torch.LongTensor(x).to('cuda')
    T = x.shape[1] if T is None else T

    # Special logic for WaveNet
    if isinstance(model.model, WaveNetModel) and not benchmark:
        l_prefix += model.model.receptive_field
        T += model.model.receptive_field
        x = F.pad(x, (model.model.receptive_field, 0), value=128)

    # Set up the initial state
    model._reset_state(batch, device='cuda')

    # First sample
    x_t = x[:, 0]
    y_all = []
    logprobs = np.zeros(x.shape[0])
    entropy = np.zeros(x.shape[0])

    if debug:
        y_raw = []

    # Generation loop
    for t in tqdm(range(T)):

        # Step through the model with the current sample
        y_t = model.step(x_t)

        # Handle special loss functions such as ProjectedAdaptiveSoftmax
        if hasattr(model.loss, "compute_logits"): y_t = model.loss.compute_logits(y_t)

        if debug:
            y_raw.append(y_t.detach().cpu())

        # Output distribution
        probs = F.softmax(y_t, dim=-1)

        # Optional: nucleus sampling
        if top_p < 1.0:
            sorted_probs = probs.sort(dim=-1, descending=True)
            csum_probs = sorted_probs.values.cumsum(dim=-1) > top_p
            csum_probs[..., 1:] = csum_probs[..., :-1].clone()
            csum_probs[..., 0] = 0
            indices_to_remove = torch.zeros_like(csum_probs)
            indices_to_remove[torch.arange(sorted_probs.indices.shape[0])[:, None].repeat(1, sorted_probs.indices.shape[1]).flatten(), sorted_probs.indices.flatten()] = csum_probs.flatten()
            y_t = y_t + indices_to_remove.int() * (-1e20)

        # Sample from the distribution
        y_t = Categorical(logits=y_t/tau).sample()

        # Feed back to the model
        if t < l_prefix-1:
            if t % multiplier == 0:
                x_t = x[:, t+1]
            else:
                x_t = y_t
        else:
            x_t = y_t

            # Calculate the log-likelihood
            if return_logprobs:
                probs = probs.squeeze(1)
                if len(y_t.shape) > 1:
                    logprobs += torch.log(probs[torch.arange(probs.shape[0]), y_t.squeeze(1)]).cpu().numpy()
                else:
                    logprobs += torch.log(probs[torch.arange(probs.shape[0]), y_t]).cpu().numpy()
                entropy += -(probs * (probs + 1e-6).log()).sum(dim=-1).cpu().numpy()

        y_all.append(x_t.cpu())
        # y_all.append(y_t.cpu())

    y_all = torch.stack(y_all, dim=1) # (batch, length)

    if isinstance(model.model, WaveNetModel) and not benchmark:
        y_all = y_all[:, model.model.receptive_field:]


    if not return_logprobs:
        if debug:
            y_raw = torch.stack(y_raw)
            return y_all, y_raw
        return y_all
    else:
        assert not debug
        return y_all, logprobs, entropy


@hydra.main(config_path="configs", config_name="generate.yaml")
def main(config: OmegaConf):
    ### See configs/generate.yaml for descriptions of generation flags ###

    # Load train config from existing Hydra experiment
    if config.experiment_path is not None:
        config.experiment_path = hydra.utils.to_absolute_path(config.experiment_path)
        experiment_config = OmegaConf.load(os.path.join(config.experiment_path, '.hydra', 'config.yaml'))
        # config = OmegaConf.merge(config, experiment_config)
        config.model = experiment_config.model
        config.task = experiment_config.task
        config.encoder = experiment_config.encoder
        config.decoder = experiment_config.decoder
        config.dataset = experiment_config.dataset
        config.loader = experiment_config.loader

    # Special override flags
    if not config.load_data:
        OmegaConf.update(config, "train.disable_dataset", True)

    if config.n_batch is None:
        config.n_batch = config.n_samples
    OmegaConf.update(config, "loader.batch_size", config.n_batch)

    # Create the Lightning Module - same as train.py

    config = utils.train.process_config(config)
    utils.train.print_config(config, resolve=True)

    print("Loading model...")
    assert torch.cuda.is_available(), 'Use a GPU for generation.'

    if config.train.seed is not None:
        pl.seed_everything(config.train.seed, workers=True)

    # Define checkpoint path smartly
    if not config.experiment_path:
        ckpt_path = hydra.utils.to_absolute_path(config.checkpoint_path)
    else:
        ckpt_path = os.path.join(config.experiment_path, config.checkpoint_path)
    print("Full checkpoint path:", ckpt_path)

    # Load model
    if ckpt_path.endswith('.ckpt'):
        model = SequenceLightningModule.load_from_checkpoint(ckpt_path, config=config)
        model.to('cuda')
    elif ckpt_path.endswith('.pt'):
        model = SequenceLightningModule(config)
        model.to('cuda')

        # Load checkpoint
        state_dict = torch.load(ckpt_path, map_location='cuda')
        model.load_state_dict(state_dict)

    # Setup: required for S4 modules in SaShiMi
    for module in model.modules():
        if hasattr(module, '_setup_step'): module._setup_step()
    model.eval()

    if config.load_data:
        # Get the eval dataloaders
        eval_dataloaders = model.val_dataloader()
        dl = eval_dataloaders[0] if config.split == 'val' else eval_dataloaders[1]
    else:
        assert config.l_prefix == 0, 'Only unconditional generation when data is not loaded.'

    # Handle save directory intelligently
    if config.save_dir:
        save_dir = hydra.utils.to_absolute_path(config.save_dir)
    else:
        save_dir = os.path.join(os.getcwd(), "samples/")
    os.makedirs(save_dir, exist_ok=True)

    # Test
    if config.test_model:
        test_step(model)

    # Generate
    assert config.n_samples % config.n_batch == 0, "For convenience, n_samples should be a multiple of n_batch"
    # input = []
    y = []
    logprobs =  []
    for _ in range(config.n_samples // config.n_batch):
        # Construct a batch
        if config.load_data:
            x, _, *_ = next(iter(dl))
            
            # [Added code starts]
            # customized input
            if config.decode == 'text':
                # x.shape [1, 8192]
                input_sentence = 'The Sinclair Scientific Programmable was introduced'
                tokenized_sentence = model.dataset.vocab.tokenize(input_sentence)
                x = model.dataset.vocab.convert_to_tensor(tokenized_sentence)
                x = x[None, :]
                # we want to predict on the full input this way
                config.l_prefix = x.shape[1]
                # and we want the output to be generated after the input
                config.l_sample += x.shape[1]
                batch = (x.repeat(config.n_reps, 1), None, None)
            elif config.decode == 'image':
                # x.shape [1, 784, 1]
                x = x[:, :config.l_prefix, :]
                batch = (x.repeat(config.n_reps, 1, 1), None, None)
                breakpoint()
            else:
                batch = (x.repeat(config.n_reps, 1), None, None)
            # [Added code ends]
        else:
            batch = (torch.zeros(config.n_batch * config.n_reps, 1).to(torch.long) + 128, None, None)

        _y, _logprobs, _ = generate(
            model, # lightning module (SequenceLightningModule from `train.py`)
            batch, # pass data to condition the generation
            l_prefix=config.l_prefix, # length of conditioning prefix
            T=config.l_sample, # length of generated sequence
            top_p=config.top_p, # nucleus sampling: always set to 1.0 for SaShiMi experiments
            tau=config.temp, # temperature: always set to 1.0 for SaShiMi experiments
            return_logprobs=True, # calc exact likelihoods
            multiplier=config.multiplier,
        )
        # input.append(x)
        y.append(_y)
        logprobs.append(_logprobs)

    # Sort based on likelihoods and save
    # input = torch.cat(input, dim=0)
    y = torch.cat(y, dim=0)
    logprobs = np.concatenate(logprobs, axis=0)
    y = y[np.argsort(logprobs.flatten())]

    # Decode quantization
    if config.decode == 'audio':
        print("Saving samples into:", save_dir)
        y = mu_law_decode(y)
        for i, d in enumerate(y):
            filename = f'{save_dir}/unconditional_{config.dataset._name_}_{config.model._name_}_len_{config.l_sample/16000.:.2f}s_gen_{i+1}.wav'
            torchaudio.save(filename, d.unsqueeze(0), 16000)
        np.save(f'{save_dir}/unconditional_{config.dataset._name_}_{config.model._name_}_len_{config.l_sample/16000.:.2f}s_logprobs.npy', logprobs)
    # [Added code starts]
    elif config.decode == 'text-auc':
        x_val = x[:, 1:config.l_sample+1]
        y_val = y[:, :]
        x_sym = [model.dataset.vocab.get_symbols(_x) for _x in x_val] # list of str
        y_sym = [model.dataset.vocab.get_symbols(_y) for _y in y_val] # list of str
        print('x_sym', x_sym)
        print('y_sym', y_sym)
        print(torch.sum(x_val == y_val))
        import pandas as pd
        ls = [1 if (x_val[0,i] == y_val[0,i]).item() else 0 for i in range(x_val.shape[1])]
        if config.multiplier == 1:
            df = pd.DataFrame(data={'m1': ls})
        else:
            df = pd.read_csv("../../../result.csv")
            df['m' + str(config.multiplier)] = ls
        df.to_csv("../../../result.csv", index=False)
        
    elif config.decode == 'text':
        # we need to run the model once first as we do not know y_sym - which is the output_names for the plot
        x_val = x[:, :config.l_prefix] # [1, l_prefix], list of str-ids
        y_val = y # [1, l_sample], list of str-ids
        # y starts from the 2nd input of x, so x[1:] == y[:l_prefix]
        # we can also get the generated part by
        # y_val = y[:, config.l_prefix-1:]
        x_sym = [model.dataset.vocab.get_symbols(_x) for _x in x_val] # list of str
        y_sym = [model.dataset.vocab.get_symbols(_y) for _y in y_val] # list of str
        print('x', x.shape, x)
        print('x_sym', x_sym)
        print('y', y.shape, y)
        print('y_sym', y_sym)
        
        # customized mask function: we need to tokenize the splitted input (or x_val above)
        # we do not take in the full sentence as what the documentation says, because in transformer, the explainer is default to be the
        # exact_explainer, which takes in a sequence to sequence generation model and mask out words on the fly
        # but we - as there is no built-in tokenizer, SHAP would assign permutation_explainer to us, which takes in a splitted input
        # and randomly mask out word to predict the output's difference
        def mask_func(mask, input):
            tokenized_sentence = model.dataset.vocab.tokenize(' '.join(input))
            input_tensor = model.dataset.vocab.convert_to_tensor(tokenized_sentence)
            output = (input_tensor * mask).reshape(1, len(input_tensor))
            return output
        
        # the full model to explain
        def f(input):
            # print("explainer input", input)
            # Generate
            assert config.n_samples % config.n_batch == 0, "For convenience, n_samples should be a multiple of n_batch"
            batch = (input.repeat(config.n_reps, 1), None, None)

            y, logprobs, _ = generate(
                    model, # lightning module (SequenceLightningModule from `train.py`)
                    batch, # pass data to condition the generation
                    l_prefix=config.l_prefix, # length of conditioning prefix
                    T=config.l_sample, # length of generated sequence
                    top_p=config.top_p, # nucleus sampling: always set to 1.0 for SaShiMi experiments
                    tau=config.temp, # temperature: always set to 1.0 for SaShiMi experiments
                    return_logprobs=True, # calc exact likelihoods
            )
            
            # Sort based on likelihoods and save
            y = y[np.argsort(logprobs.flatten())]

            # Decode quantization
            if config.decode == 'text':
                # print("explainer output", y.shape, y)
                return y
        
        # output_names is a dictionary to map word (token) to their ids, see
        # https://shap.readthedocs.io/en/latest/example_notebooks/text_examples/sentiment_analysis/Using%20custom%20functions%20and%20tokenizers.html?highlight=output_names#Create-an-explainer
        labels = sorted(model.dataset.vocab.sym2idx, key=model.dataset.vocab.sym2idx.get)
        
        # list of sentences (tensor) -> masker (every sentence -> every sentence masked) -> list of sentences masked -> f -> list of outputs
        # set max_evals to at least 2 * num_of_words_to_generate + 1 - required by permutation_explainer
        # set higher for better explainability, but takes longer
        explainer = shap.Explainer(f, masker=mask_func, output_names=labels, feature_names=None, max_evals=2*config.l_sample+1)
        
        # permutation_explainer needs to take in pd dataframe, opposed to list of sentences in exact_explainer used in the naive transformer
        import pandas as pd
        df = pd.DataFrame(data=x_sym)
        print('pd', df)
        
        # batch_size needs to match sentence size, or explainer cannot find the correct shape
        shap_values = explainer(df, batch_size=1)
        
        # needs to manually set output_names, explainer parameter does not work - as permutation explainer is not supposed to be called 
        # to text input, the code in `_explainer.py` assumes text input takes in transformer model
        # and there is no support for handling name matching in `_permutation.py`
        shap_values.output_names = y_sym[0]
        
        # token should be splitted for better visualization - auto handled in transformer
        for i in range(1, len(shap_values.data[0])):
            shap_values.data[0][i] = ' ' + shap_values.data[0][i]

        # .values [1, input_len, output_len]
        # .base_values [1, output_len]
        # .data = [output_len] - str
        print('shap_values', shap_values)

        # finally - plot in jupyter notebook
        # see `_text.py` to debug the plot
        shap.plots.text(shap_values)
        # [Added code ends]
    else: pass


if __name__ == "__main__":
    main()
