from typing import Optional

import torch

import os
from tqdm import tqdm

import wandb

from optimizers import OPTIMIZER_DICT
import utils.constants as constants
from utils.dot_dict import DotDict
from utils.logging_utils import LogSection


class BaseTrainer:

    def __init__(
        self,
        project: str,
        name: str,
        config: dict,
        debug: Optional[bool] = False,
        notes: Optional[str] = None,
        resume_id: Optional[str] = None,
        resume_step: Optional[int] = None,
    ):
        """ A trainer to train models using PyTorch.

        Args:
            project (str): name of the project to save to
            name (str): name of the run in the project
            config (dict): configuration for the trainer
            debug (bool, optional): Whether to disable saving. Defaults to False.
            notes (str, optional): Notes to add to the run. Defaults to None.
        """
        self.project = project
        self.name = name
        self.config = config
        self.debug = debug

        self.save_name = f"{project}_{name}"

        if not self.debug:
            os.makedirs(constants.LOCAL_DATA_PATH, exist_ok=True)

            wandb.init(
                project=project,
                name=name,
                config=config,
                notes=notes,
                fork_from=f"{resume_id}?_step={resume_step}" if resume_id is not None else None,
            )

        # apply hyperparams
        for k in config:
            setattr(self, k, config[k])

        # init log
        self.log = DotDict()


    def log_step(self):
        if not self.debug:
            wandb.log(self.log.to_dict())
        
        self.log = DotDict()
        

    @torch.no_grad()
    def save_checkpoint(
        self,
        model,
        optimizer,
        step
    ):
        if self.debug:
            return

        with LogSection("checkpoint saving"):

            run_path = os.path.join(
                constants.LOCAL_DATA_PATH,
                self.save_name,
            )
            os.makedirs(run_path, exist_ok=True)

            curr_path = os.path.join(
                run_path,
                f"{step:012d}"
            )
            os.makedirs(curr_path, exist_ok=True)

            ckpt_path = os.path.join(
                curr_path,
                f"checkpoint.ckpt"
            )
                
            ckpt = {
                "model": model.state_dict(),
            }
            if self.save_optimizer:
                ckpt["optimizer"] = optimizer.state_dict()

            torch.save(ckpt, ckpt_path)

            try:
                model.config.save_pretrained(curr_path, push_to_hub=False)
            except:
                print("Warning: model config not saved to checkpoint")


    def train(
        self,
        model: torch.nn.Module,
        loader
    ):

        # init model
        for p in model.parameters():
            p.requires_grad_(True)
        model.train()

        # init training objs
        optimizer = OPTIMIZER_DICT[self.optimizer_type](
            model.parameters(),
            **self.optimizer_kwargs
        )

        # init loop vars
        curr_step = 0
        pbar = tqdm(desc=f"Training {self.project}/{self.name}")
        pbar.update(0)

        # run loop
        for epoch in range(self.num_epochs):
            for batch in loader:

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    results = self.train_step(
                        curr_step,
                        model,
                        **batch
                    )

                results.loss.backward()

                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = torch.linalg.vector_norm(p.grad.detach())
                        grad_norm += param_norm.item() ** 2
                results.grad_norm = (grad_norm ** 0.5)

                if curr_step == 0:
                    with open(os.path.join(constants.LOCAL_DATA_PATH, "gradients.txt"), "w") as f:

                        f.write("\n === GRADIENTS === \n\n")
                        for n, p in model.named_parameters():
                            if p.grad is not None:
                                f.write(f"{n}\n")

                        f.write("\n === NO GRADIENT === \n\n")
                        for n, p in model.named_parameters():
                            if p.grad is None:
                                f.write(f"{n}\n")

                # clip the gradients
                torch.nn.utils.clip_grads_with_norm_(
                    model.parameters(),
                    1.0,
                    results.grad_norm,
                )

                # perform a single optimizer step
                if "reset_optimizer" in results.keys():
                    results.pop("reset_optimizer")

                    del optimizer
                    optimizer = OPTIMIZER_DICT[self.optimizer_type](
                        model.parameters(),
                        **self.optimizer_kwargs
                    )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # update tracking
                curr_step += 1
                self.log.steps_completed = curr_step
                self.log.examples_seen = curr_step * self.bs

                # update pbar
                pbar.update(1)
                pbar.set_postfix(
                    loss=results.loss.item(),
                )

                # log results
                for k, v in results.items():
                    if isinstance(v, torch.Tensor):
                        self.log[k] = v.item()
                    else:
                        self.log[k] = v

                # save
                self.log_step()
                if curr_step % self.checkpoint_interval == 0:
                    self.save_checkpoint(
                        model,
                        optimizer,
                        curr_step
                    )

        self.save_checkpoint(
            model,
            optimizer,
            curr_step
        )
    

    def train_step(
        self,
        step,
        model,
        **kwargs
    ):
        """ Get results of a single training step.
         - Must return DotDict of results
         - Results must include 'loss' key

        Args:
            model: model to train
            **kwargs: inputs from the minibatch

        Returns:
            DotDict: result tensors containing 'loss' key
        """
        raise NotImplementedError("train_step must be implemented in child class!")
    