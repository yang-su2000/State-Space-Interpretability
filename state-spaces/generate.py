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

    def tokenize(s, return_offsets_mapping=True):
        import re
        pos = 0
        offset_ranges = []
        input_ids = []
        for m in re.finditer(r"\W", s):
            start, end = m.span(0)
            offset_ranges.append((pos, start))
            input_ids.append(s[pos:start])
            pos = end
        if pos != len(s):
            offset_ranges.append((pos, len(s)))
            input_ids.append(s[pos:])
        out = {}
        out["input_ids"] = input_ids
        if return_offsets_mapping:
            out["offset_mapping"] = offset_ranges
        print(out)
        return out
        # print('sentence', s)
        # tokenized_sentence = model.dataset.vocab.tokenize(s)
        # tensor = model.dataset.vocab.convert_to_tensor(tokenized_sentence)
        # print('tensor', tensor)
        # return tensor

    def f(input):
            print("explainer start with input", input)
            # input_tokens = [model.dataset.vocab.convert_to_tensor(tokens) for tokens in input]
            # input_tokens = torch.stack(input, dim=0)
            # breakpoint()
            # input = tokenize(input)
            # print("tokenized with input", input_tokens)
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
            if config.decode == 'audio':
                print("Saving samples into:", save_dir)
                y = mu_law_decode(y)
                for i, d in enumerate(y):
                    filename = f'{save_dir}/unconditional_{config.dataset._name_}_{config.model._name_}_len_{config.l_sample/16000.:.2f}s_gen_{i+1}.wav'
                    torchaudio.save(filename, d.unsqueeze(0), 16000)
                np.save(f'{save_dir}/unconditional_{config.dataset._name_}_{config.model._name_}_len_{config.l_sample/16000.:.2f}s_logprobs.npy', logprobs)
            elif config.decode == 'text':
                # breakpoint()
                y_val = y
                y = [model.dataset.vocab.get_symbols(_y) for _y in y]
                y_sentences = [' '.join(_y) for _y in y]
                print("shap forward pass done with output", y_val.shape)
                # scores = (np.exp(logprobs.flatten()).T / np.exp(logprobs.flatten()).sum(-1)).T
                # val = sp.special.logit(scores[:,1])
                return logprobs#[None, :]


    # input, truth, _ = next(iter(dl))
    # input = input[:][:config.l_sample]
    # truth = input[:][:config.l_sample]
    # print('before explainer')
    # breakpoint()
    # explainer = shap.Explainer(f, input)
    # # explainer = shap.explainers.Permutation(f, max_evals = len(model.dataset.vocab) * 2 + 1) # 267735 * 2 + 1
    # shap_values = explainer(input) #np.array(x_sentences[:][:config.l_sample])) #np.array(x_val[:][:config.l_sample]))
    # # breakpoint() # Inspect output manually for now
    # shap.plots.text(shap_values)
    # return

    # Generate
    assert config.n_samples % config.n_batch == 0, "For convenience, n_samples should be a multiple of n_batch"
    # input = []
    y = []
    logprobs =  []
    for _ in range(config.n_samples // config.n_batch):
        # Construct a batch
        if config.load_data:
            x, _, *_ = next(iter(dl))
            batch = (x.repeat(config.n_reps, 1), None, None)
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
    elif config.decode == 'text':
        x_val = [_x[:config.l_prefix] for _x in x]
        y_val = [_y[:config.l_prefix] for _y in y]
        # breakpoint()
        x = [model.dataset.vocab.get_symbols(_x) for _x in x_val]
        y = [model.dataset.vocab.get_symbols(_y) for _y in y_val]
        x_sentences = [' '.join(_x) for _x in x]
        y_sentences = [' '.join(_y) for _y in y]
        print('x', x_val, x, x_sentences)
        print('y', y_val, y, y_sentences)
        
        # shap.initjs()
        
        def mask_func(mask, input):
            print('x', input)
            tokenized_sentence = model.dataset.vocab.tokenize(input[0])
            tensor = model.dataset.vocab.convert_to_tensor(tokenized_sentence)
            # print('x_tensor', tensor)
            mask = True
            output = (tensor * mask).reshape(1, len(tensor))
            print('mask', mask)
            print('output', output)
            return output
        
        # explainer = shap.Explainer(f, np.array(x_sentences))
        # shap_values = explainer(np.array(x_sentences))
        # breakpoint()
        labels = sorted(model.dataset.vocab.sym2idx, key=model.dataset.vocab.sym2idx.get)
        # labels = list(model.dataset.vocab.sym2idx.values())
        # masker = shap.maskers.Text(tokenize)
        masker = mask_func
        # masker = shap.maskers.Text(r"\W")
        # masker = x_val
        # list of sentences (tensor) -> masker (every sentence -> every sentence masked) -> list of sentences masked -> f -> list of outputs
        explainer = shap.Explainer(f, masker, output_names=y[0], feature_names=y[0], max_evals=2*config.l_sample+1) #, max_evals=16385)
        # explainer = shap.explainers.Permutation(f, max_evals = len(model.dataset.vocab) * 2 + 1) # 267735 * 2 + 1
        import pandas as pd
        df = pd.DataFrame(data=x_sentences)
        print(df)
        with open("/home/ys724/S4/State-Space-Interpretability/state-spaces/fig.html", "w") as file:
            file.close()
        shap_values = explainer(df, batch_size=1) # x_val[:,:config.l_sample])
        print('success') # Inspect output manually for now
        # breakpoint()
        shap_values.output_names = y[0]
        # breakpoint()
        shap.plots.text(shap_values)
        # breakpoint()
        # with open("/home/ys724/S4/State-Space-Interpretability/state-spaces/fig.html", "w") as file:
        #     file.write(html.data)
        # plt.show()
        # df_val = pd.DataFrame(data=x, dtype=int)
        # shap.summary_plot(shap_values[0,:].base_values, x_val[0][None,:])
        # shap.force_plot(shap_values.base_values, shap_values.values[0,:], df_val.iloc[0,:], show=False, matplotlib=True) \
        #     .savefig('/home/ys724/S4/State-Space-Interpretability/state-spaces/fig.png')
        # plt.savefig('/home/ys724/S4/State-Space-Interpretability/state-spaces/fig.png')
        # plt.show()
        # plt.close()
    else: pass


if __name__ == "__main__":
    main()
