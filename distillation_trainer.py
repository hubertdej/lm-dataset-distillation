import logging
import time
from typing import List, Tuple
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from basics import evaluate_steps, final_objective_loss, task_loss
from data import Datapoint
from networks.language_model import LanguageModelWrapper
from state import State
from utils.distributed import all_reduce_coalesced, broadcast_coalesced
from utils.io import save_results
from transformers import TrainingArguments, Trainer
from itertools import zip_longest
import datasets
import copy

def permute_list(list):
    indices = np.random.permutation(len(list))
    return [list[i] for i in indices]

StepsType = List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
class DistillationTrainer(object):
    def __init__(self, state: State, model: LanguageModelWrapper, tokenizer, train_loader, loss_fn = torch.nn.functional.cross_entropy, sampled_sentences = None, initial_data = None):
        self.state = state
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.loss_fn = loss_fn
        self.model_state = model.state_dict()

        with torch.no_grad():
            self.embedding_matrix = self.model.extract_embedding_matrix()
            self.embedding_matrix.requires_grad_(False)
            self.embedding_matrix = self.embedding_matrix.to(self.state.device)

        self.num_data_steps = state.distill_steps
        self.T = state.distill_steps * state.distill_epochs
        self.num_per_step = state.batch_size * state.seq_len
        assert state.distill_lr >= 0, 'distill_lr must be >= 0'

        if sampled_sentences is None:
            self.init_data_optim(initial_data)
            self.sampled_sentences = None
        else:
            self.generated_text = []
            self.generated_embeddings = []
            self.generated_training_data = []
            self.generated_training_labels = []
            self.current_dataloader = None
            self.sampled_sentences = sampled_sentences
            self.training_args = TrainingArguments(
                "output",
                evaluation_strategy = "epoch",
                learning_rate=2e-5,
                weight_decay=0.01,
                num_train_epochs=2,
                save_strategy = "no",
                report_to="none",
            )

    def iterative_generation(self):
        if self.sampled_sentences is None:
            return

        for iter, batch in enumerate(self.sampled_sentences):
            self.init_data_optim(batch)
            self.train()

            resulting_text, token_ids, labels, embeddings = self.get_train_data_and_text()
            self.generated_text += resulting_text
            self.generated_training_data += token_ids
            self.generated_training_labels += labels
            self.generated_embeddings += embeddings

            frequency = 2
            mid_freq = frequency // 2

            print(f"Batch {iter} generated out of {self.sampled_sentences}")

            if (1 + iter) % frequency == 0:
                with open('generated_embeddings', 'wb') as fp:
                    pickle.dump(self.generated_embeddings, fp)

                with open('generated_text', 'wb') as fp:
                    pickle.dump(self.generated_text, fp)

                with open('generated_data', 'wb') as fp:
                    pickle.dump(self.generated_training_data, fp)

                with open('generated_labels', 'wb') as fp:
                    pickle.dump(self.generated_training_labels, fp)

            if (1 + iter) % frequency == mid_freq:
                with open('generated_embeddings_backup', 'wb') as fp:
                    pickle.dump(self.generated_embeddings, fp)

                with open('generated_text_backup', 'wb') as fp:
                    pickle.dump(self.generated_text, fp)

                with open('generated_data_backup', 'wb') as fp:
                    pickle.dump(self.generated_training_data, fp)

                with open('generated_labels_backup', 'wb') as fp:
                    pickle.dump(self.generated_training_labels, fp)



    def init_data_optim(self, init_data = None):
        self.params = []
        state = self.state

        # Synthetic data (sequences)
        self.data: List[Datapoint] = []
        for i in range(self.num_data_steps):
            if init_data is None:
                token_ids = torch.randint(low=0, high=state.vocab_size, size=(state.batch_size, state.seq_len),
                                         device=state.device)
            else:
                token_ids = init_data[i * state.batch_size : (i + 1) * state.batch_size]
                token_ids = [x[:state.seq_len] for x in token_ids]
                token_ids = np.array(list(zip_longest(*token_ids, fillvalue=self.tokenizer.pad_token_id))).T
                token_ids = torch.tensor(token_ids).to(state.device)

            datapoint = Datapoint(token_ids=token_ids, pad_token_id=self.tokenizer.pad_token_id)
            datapoint.update_embeddings(self.embedding_matrix)
            datapoint.update_labels()
            datapoint.to(state.device)
            datapoint.embeddings.requires_grad_(True)
            self.data.append(datapoint)
            self.params.append(datapoint.embeddings)
            # print("TOKENS", token_ids)

        # Learning rates
        raw_init_distill_lr = torch.tensor(state.distill_lr, device=state.device)
        raw_init_distill_lr = raw_init_distill_lr.repeat(self.T, 1)
        self.raw_distill_lrs = raw_init_distill_lr.expm1_().log_().requires_grad_()
        self.params.append(self.raw_distill_lrs)

        assert len(self.params) > 0, "must have at least 1 parameter"

        # Optimizer and scheduler
        self.optimizer = optim.AdamW(self.params, lr=state.lr, betas=(0.5, 0.999))
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=state.decay_epochs,
                                                   gamma=state.decay_factor)
        for p in self.params:
            p.grad = torch.zeros_like(p)

    def get_steps(self) -> StepsType:
        data_label_iterable = (x for _ in range(self.state.distill_epochs) for x in self.data)
        lrs = F.softplus(self.raw_distill_lrs).unbind()

        steps = []
        for datapoint, lr in zip(data_label_iterable, lrs):
            steps.append((datapoint.embeddings, datapoint.labels, lr))

        return steps

    def forward(self, rdata: torch.Tensor, rlabel: torch.Tensor, steps: StepsType):
        state = self.state

        # forward
        self.model.train()
        w = self.model.get_param()
        params = [w]
        gws = []

        for step_i, (embeddings, labels, lr) in enumerate(steps):
            # print("Step: ", step_i, " of ", len(steps), " steps.")
            with torch.enable_grad():
                outputs = self.model.forward_with_param(w, inputs_embeds=embeddings)
                logits = outputs.logits
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=self.tokenizer.pad_token_id)
            gw, = torch.autograd.grad(loss, w, lr.squeeze(), create_graph=True)
            with torch.no_grad():
                new_w = w.sub(gw).requires_grad_()
                params.append(new_w)
                gws.append(gw)
                w = new_w

        # final L
        self.model.eval()
        outputs = self.model.forward_with_param(params[-1], rdata)
        logits = outputs.logits
        ll = self.loss_fn(logits.view(-1, logits.size(-1)), rlabel.view(-1), ignore_index=self.tokenizer.pad_token_id)
        return ll, (ll, params, gws)

    def backward(self, steps: StepsType, saved_for_backward):
        l, params, gws = saved_for_backward

        datas = []
        gdatas = []
        lrs = []
        glrs = []
        dw, = torch.autograd.grad(l, (params[-1],))

        # backward
        self.model.train()
        # Notation:
        #   math:    \grad is \nabla
        #   symbol:  d* means the gradient of final L w.r.t. *
        #            dw is \d L / \dw
        #            dgw is \d L / \d (\grad_w_t L_t )
        # We fold lr as part of the input to the step-wise loss
        #
        #   gw_t     = \grad_w_t L_t       (1)
        #   w_{t+1}  = w_t - gw_t          (2)
        #
        # Invariants at beginning of each iteration:
        #   ws are BEFORE applying gradient descent in this step
        #   Gradients dw is w.r.t. the updated ws AFTER this step
        #      dw = \d L / d w_{t+1}
        for (embeddings, labels, lr), w, gw in reversed(list(zip(steps, params, gws))):
            # hvp_in are the tensors we need gradients w.r.t. final L:
            #   lr (if learning)
            #   data
            #   ws (PRE-GD) (needed for next step)
            #
            # source of gradients can be from:
            #   gw, the gradient in this step, whose gradients come from:
            #     the POST-GD updated ws
            hvp_in = [w]
            hvp_in.append(embeddings)
            hvp_in.append(lr)
            dgw = dw.neg()  # gw is already weighted by lr, so simple negation
            hvp_grad = torch.autograd.grad(
                outputs=(gw,),
                inputs=hvp_in,
                grad_outputs=(dgw,)
            )
            # Update for next iteration, i.e., previous step
            with torch.no_grad():
                # Save the computed gdata and glrs
                datas.append(embeddings)
                gdatas.append(hvp_grad[1])
                lrs.append(lr)
                glrs.append(hvp_grad[2])

                # Update for next iteration, i.e., previous step
                # Update dw
                # dw becomes the gradients w.r.t. the updated w for previous step
                dw.add_(hvp_grad[0])

        return datas, gdatas, lrs, glrs

    def accumulate_grad(self, grad_infos):
        bwd_out = []
        bwd_grad = []
        for datas, gdatas, lrs, glrs in grad_infos:
            bwd_out += list(lrs)
            bwd_grad += list(glrs)
            for d, g in zip(datas, gdatas):
                d.grad.add_(g)
        if len(bwd_out) > 0:
            torch.autograd.backward(bwd_out, bwd_grad)

    def save_results(self, steps=None, dir=".", subfolder=''):
        with torch.no_grad():
            steps = steps or self.get_steps()
            save_results(dir, steps, subfolder=subfolder)

    def __call__(self):
        return self.train()

    def prefetch_train_loader_iter(self):
        train_iter = iter(self.train_loader)
        for epoch in range(self.state.epochs):
            niter = len(train_iter)
            prefetch_it = max(0, niter - 2)
            for it, val in enumerate(train_iter):
                # Prefetch (start workers) at the end of epoch BEFORE yielding
                if it == prefetch_it and epoch < self.state.epochs - 1:
                    train_iter = iter(self.train_loader)

                yield epoch, it, val

    def train(self):
        state = self.state
        device = state.device
        train_loader = self.train_loader
        ckpt_int = state.checkpoint_interval

        data_t0 = time.time()

        for epoch, it, (rdata, rlabel) in self.prefetch_train_loader_iter():
            data_t = time.time() - data_t0
            if it == 3:
                self.scheduler.step()
            # if it >=5:
            #     break

            if it == 0 and ((ckpt_int >= 0 and epoch % ckpt_int == 0) or epoch == 0):
                with torch.no_grad():
                    steps = self.get_steps()
                self.save_results(steps=steps, subfolder='checkpoints/epoch{:04d}'.format(epoch))
                # evaluate_steps(state, steps, 'Begin of epoch {}'.format(epoch))
            # do_log_this_iter = it == 0 or (state.log_interval >= 0 and it % state.log_interval == 0)
            do_log_this_iter = False
            self.optimizer.zero_grad(set_to_none=False)
            rdata, rlabel = rdata.to(device, non_blocking=True), rlabel.to(device, non_blocking=True)

            t0 = time.time()
            losses = []
            steps = self.get_steps()
            # activate everything needed to run on this process
            grad_infos = []

            l, saved = self.forward(rdata, rlabel, steps)
            losses.append(l.detach())
            grad_infos.append(self.backward(steps, saved))
            del l, saved
            self.accumulate_grad(grad_infos)

            # all reduce if needed
            # average grad
            all_reduce_tensors = [p.grad for p in self.params]
            if do_log_this_iter:
                losses = torch.stack(losses, 0).sum()
                all_reduce_tensors.append(losses)

            losses = torch.stack(losses, 0).sum()
            # print("Losses: ", losses.item())
            # opt step
            self.optimizer.step()
            t = time.time() - t0
            if do_log_this_iter:
                loss = losses.item()
                logging.info((
                    'Epoch: {:4d} [{:7d}/{:7d} ({:2.0f}%)]\tLoss: {:.4f}\t'
                    'Data Time: {:.2f}s\tTrain Time: {:.2f}s'
                ).format(
                    epoch, it * train_loader.batch_size, len(train_loader.dataset),
                    100. * it / len(train_loader), loss, data_t, t,
                ))
                if loss != loss:  # nan
                    raise RuntimeError('loss became NaN')
            del steps, grad_infos, losses, all_reduce_tensors

            data_t0 = time.time()

        with torch.no_grad():
            steps = self.get_steps()
        # self.save_results(steps)
        return steps

    def decode_embeddings(self, token_ids):
        return [self.tokenizer.decode(token_id) for token_id in token_ids]

    # def get_train_data_and_text(self):
    #     resulting_text = []
    #     dataset = []

    #     for i in range(len(self.data)):
    #         self.data[i].update_token_ids(self.embedding_matrix)
    #         self.data[i].update_labels()
    #         resulting_text += [" ".join(self.decode_embeddings(tokens)) for tokens in self.data[i].token_ids]
    #         dataset += [self.data[i].token_ids] #, self.data[i].labels]

    #     return resulting_text, dataset

    def get_train_data_and_text(self):
        resulting_text = []
        token_ids = []
        labels = []
        embeddings = []

        for x in self.data:
            x.update_token_ids(self.embedding_matrix)
            x.update_labels()
            resulting_text += self.decode_embeddings(x.token_ids)
            token_ids += [x.token_ids.cpu().detach()]
            labels += [x.labels.cpu().detach()]
            embeddings += [x.embeddings.cpu().detach()]

        return resulting_text, token_ids, labels, embeddings

    def get_generated_text(self):
        resulting_text = []

        for x in self.data:
            x.update_token_ids(self.embedding_matrix)
            resulting_text += self.decode_embeddings(x.token_ids)

        # If extra space between words:
        # for x in self.data:
        #     x.update_token_ids(self.embedding_matrix)
        #     resulting_text += [" ".join(self.decode_embeddings(tokens)) for tokens in x.token_ids]

        return resulting_text

def distill(state, model, tokenizer, train_loader, loss_fn = torch.nn.functional.cross_entropy):
    return DistillationTrainer(state, model, tokenizer, train_loader, loss_fn).train()
