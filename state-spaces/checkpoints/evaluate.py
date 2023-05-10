from tqdm.auto import tqdm
import hydra
import torch
import numpy as np
from pathlib import Path
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.nn.modules import module
import torch.nn.functional as F
from torch.distributions import Categorical
from src import utils
from einops import rearrange, repeat, reduce

from train import SequenceLightningModule
from omegaconf import OmegaConf

@hydra.main(config_path="../configs", config_name="generate.yaml")
def main(config: OmegaConf):
    # Load train config from existing Hydra experiment
    if config.experiment_path is not None:
        config.experiment_path = hydra.utils.to_absolute_path(config.experiment_path)
        experiment_config = OmegaConf.load(os.path.join(config.experiment_path, '.hydra', 'config.yaml'))
        config.model = experiment_config.model
        config.task = experiment_config.task
        config.encoder = experiment_config.encoder
        config.decoder = experiment_config.decoder
        config.dataset = experiment_config.dataset
        config.loader = experiment_config.loader

    config = utils.train.process_config(config)
    utils.train.print_config(config, resolve=True)

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
        model.eval()

    if config.decode == 'text':
        ## Test single batch
        debug = False
        if debug:
            val_dataloaders = model.val_dataloader()
            loader = val_dataloaders[0] if isinstance(val_dataloaders, list) else val_dataloaders

            model = model.to('cuda')
            model.eval()
            batch = next(iter(loader))
            batch = (batch[0].cuda(), batch[1].cuda(), batch[2])
            model.model.layers[0].layer.kernel()
            with torch.no_grad():
                x, y, *w = model.forward(batch)
                loss = model.loss_val(x, y, *w)
                print("Single batch loss:", loss)

        ## Use PL test to calculate final metrics
        from train import create_trainer
        trainer = create_trainer(config)
        trainer.test(model)

    # [Added code starts]
    elif config.decode == 'image-auc':
        val_dataloaders = model.val_dataloader()
        loader = val_dataloaders[0] if isinstance(val_dataloaders, list) else val_dataloaders

        model = model.to('cuda')
        model.eval()
        batch = next(iter(loader))
        # batch shape ([50, 784, 1], [50], {})
        batch_x = (batch[0].cuda(), batch[1].cuda(), batch[2])
        # batch_y shape ([50, 10], [50], {})
        batch_y = model(batch_x)
        # target = batch_x[1].cpu().numpy()
        from src.tasks.metrics import accuracy, roc_auc_macro
        acc = accuracy(batch_y[0], batch_y[1])
        # roc_auc = roc_auc_macro(batch_y[0], batch_y[1])
        breakpoint()
    
    elif config.decode == 'image':
        val_dataloaders = model.val_dataloader()
        loader = val_dataloaders[0] if isinstance(val_dataloaders, list) else val_dataloaders

        model = model.to('cuda')
        model.eval()
        batch = next(iter(loader))
        # batch shape ([50, 784, 1], [50], {})
        batch_x = (batch[0].cuda(), batch[1].cuda(), batch[2])
        # batch_y shape ([50, 10], [50], {})
        batch_y = model(batch_x)
        
        import shap
        use_permutation = True
        if use_permutation:
            # [Permutation SHAP]
            batch_size = batch[0].shape[0]
            input_size = 1
            class_dict = {idx: c for c, idx in model.dataset.dataset_val.dataset.class_to_idx.items()}
            n_evals = 2 * 784 + 1
            input = batch[0][:input_size].view(1, 784).cpu().numpy()
            output_names = [class_dict[idx.item()] for idx in np.argsort(batch_y[0][0, :].flatten().cpu().detach().numpy())]
            class_names = list(class_dict.values())
            
            def mask_func(mask, input):
                output = (input * mask).reshape(1, len(input))
                return output
            
            def f(x):
                x = torch.from_numpy(x).view(1, 784, 1)
                y = torch.zeros(1)
                input = (x.cuda(), y.cuda(), {})
                with torch.no_grad():
                    output = model(input)
                pclass = torch.argmax(output[0], axis=1).cpu().numpy()
                output = output[0].cpu()
                outarg = output[0][np.argsort(output.flatten())].unsqueeze(0)
                return outarg

            # create an explainer with model and image masker
            labels = sorted(model.dataset.dataset_val.dataset.class_to_idx, key=model.dataset.dataset_val.dataset.class_to_idx.get)
            explainer = shap.Explainer(f, mask_func, output_names=labels, max_evals=n_evals)

            import pandas as pd
            df = pd.DataFrame(data=input)
            print('pd', df)
            
            # feed only one image
            # here we explain two images using 100 evaluations of the underlying model to estimate the SHAP values
            # .values shape [input_size, 784, 10] -> (10, [input_size, 28, 28, 1])
            # .data shape [input_size, 784]
            # .base_values shape [input_size, 10]
            shap_values = explainer(df, batch_size=input_size)
            # shap_values.data = torch.from_numpy(shap_values.data).permute(0, 3, 1, 2).numpy()
            # shap_values.values = [val for val in np.moveaxis(shap_values.values, -1, 0)]
            breakpoint()
            shap_values.values = shap_values.values.reshape(1, 28, 28, 1, 10)
            shap_values.data = shap_values.data.reshape(1, 28, 28, 1)
            shap_values.output_names = output_names
            shap_values.values = [val for val in np.moveaxis(shap_values.values, -1, 0)]
            shap.image_plot(shap_values=shap_values.values, pixel_values=shap_values.data, labels=shap_values.output_names)
        else:
            # [Partition SHAP]
            batch_size = batch[0].shape[0]
            input_size = 4
            class_dict = {idx: c for c, idx in model.dataset.dataset_val.dataset.class_to_idx.items()}
            n_evals = 2000
            topk = 10 # topk class to explain
            input = batch[0][:input_size].view(input_size, 28, 28, 1).cpu().numpy()
            class_names = list(class_dict.values())
            
            # define a masker that is used to mask out partitions of the input image.
            masker_blur = shap.maskers.Image("inpaint_ns", input[0].shape)
            
            def f(x):
                x = torch.from_numpy(x).view(x.shape[0], 784, 1)
                y = torch.zeros(x.shape[0])
                input = (x.cuda(), y.cuda(), {})
                with torch.no_grad():
                    output = model(input)
                pclass = torch.argmax(output[0], axis=1).cpu().numpy()
                output = output[0].cpu()
                return output

            # create an explainer with model and image masker
            labels = sorted(model.dataset.dataset_val.dataset.class_to_idx, key=model.dataset.dataset_val.dataset.class_to_idx.get)
            explainer = shap.Explainer(f, masker_blur, output_names=class_names)

            # feed only one image
            # here we explain two images using 100 evaluations of the underlying model to estimate the SHAP values
            # .values shape [input_size, 28, 28, 1, topk] -> (topk, [input_size, 28, 28, 1])
            # .data shape [input_size, 28, 28, 1]
            # .base_values shape [input_size, topk]
            # .output_names (input_size, topk)
            shap_values = explainer(input, batch_size=batch_size, max_evals=n_evals, outputs=shap.Explanation.argsort.flip[:topk])
            # breakpoint()
            # shap_values.data = torch.from_numpy(shap_values.data).permute(0, 3, 1, 2).numpy()
            shap_values.values = [val for val in np.moveaxis(shap_values.values, -1, 0)]
            shap.image_plot(shap_values=shap_values.values, pixel_values=shap_values.data, labels=shap_values.output_names)
        
        # [Deep SHAP - only for Tensorflow]
        # background = (batch[0][:47], batch[1][:47], batch[2])
        # test_images = (batch[0][47:50], batch[1][47:50], batch[2])

        # e = shap.DeepExplainer(f, background)
        # shap_values = e.shap_values(test_images)
        # breakpoint()
        # shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        # test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)
        # # plot the feature attributions
        # shap.image_plot(shap_numpy, -test_numpy)
        
        print('success')
    # [Added code ends]

if __name__ == '__main__':
    main()
