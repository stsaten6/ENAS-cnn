"""The module for training ENAS."""
import contextlib
#简化with 语句
#TODO understand how to use contextlib  貌似代码里没用到?
import glob
import math
import os
#train controller 类似 dropout 的一个框架
import numpy as np
import scipy.signal
from tensorboard import TensorBoard
import torch
from torch import nn
import torch.nn.parallel
from torch.autograd import Variable

import models
import utils

# logger = utils.get_logger()



def _apply_penalties(extra_out, args):
    """Based on `args`, optionally adds regularization penalty terms for
    activation regularization, temporal activation regularization and/or hidden
    state norm stabilization.

    Args:
        extra_out[*]:
            dropped: Post-dropout activations.
            hiddens: All hidden states for a batch of sequences.
            raw: Pre-dropout activations.

    Returns:
        The penalty term associated with all of the enabled regularizations.

    See:
        Regularizing and Optimizing LSTM Language Models (Merity et al., 2017)
        Regularizing RNNs by Stabilizing Activations (Krueger & Memsevic, 2016)
    """
    penalty = 0

    # Activation regularization.
    if args.activation_regularization:
        penalty += (args.activation_regularization_amount *
                    extra_out['dropped'].pow(2).mean())

    # Temporal activation regularization (slowness)
    if args.temporal_activation_regularization:
        raw = extra_out['raw']
        penalty += (args.temporal_activation_regularization_amount *
                    (raw[1:] - raw[:-1]).pow(2).mean())

    # Norm stabilizer regularization
    if args.norm_stabilizer_regularization:
        penalty += (args.norm_stabilizer_regularization_amount *
                    (extra_out['hiddens'].norm(dim=-1) -
                     args.norm_stabilizer_fixed_point).pow(2).mean())

    return penalty


def discount(x, amount):
    return scipy.signal.lfilter([1], [1, -amount], x[::-1], axis=0)[::-1]


def _get_optimizer(name):
    if name.lower() == 'sgd':
        optim = torch.optim.SGD
    elif name.lower() == 'adam':
        optim = torch.optim.Adam

    return optim


def _get_no_grad_ctx_mgr():
    """Returns a the `torch.no_grad` context manager for PyTorch version >=
    0.4, or a no-op context manager otherwise.
    """
    if float(torch.__version__[0:3]) >= 0.4:
        return torch.no_grad()

    return contextlib.suppress()


def _check_abs_max_grad(abs_max_grad, model):
    """Checks `model` for a new largest gradient for this epoch, in order to
    track gradient explosions.
    """
    finite_grads = [p.grad.data
                    for p in model.parameters()
                    if p.grad is not None]
    new_max_grad = max([grad.max() for grad in finite_grads])
    new_min_grad = max([grad.min() for grad in finite_grads])

    new_abs_max_grad = max([grad.max() for grad in finite_grads])
    if new_abs_max_grad > abs_max_grad:
        print('regularizing:')
        return new_abs_max_grad

    return abs_max_grad


class Trainer(object):
    """A class to wrap training code."""
    def __init__(self, args, dataset):
        """Constructor for training algorithm.

        Args:
            args: From command line, picked up by 'argparse'
            dataset: Currently only `data.text.Corpus` is supported.

        Initializes:
            - Data: train, val and test.
            - Model: shared and controller.
            - Inference: optimizers for shared and controller parameters.
            - Criticism: cross-entropy loss for training the shared model.
        """
        #TODO   加个检查准确率的
        self.args = args
        self.controller_step = 0
        self.cuda = args.cuda
        self.dataset = dataset
        self.epoch = 0
        self.shared_step = 0
        self.start_epoch = 0

        print('regularizing:')
        for regularizer in [('activation regularization',
                             self.args.activation_regularization),
                            ('temporal activation regularization',
                             self.args.temporal_activation_regularization),
                            ('norm stabilizer regularization',
                             self.args.norm_stabilizer_regularization)]:
            if regularizer[1]:
                print(f'{regularizer[0]}')

        # self.train_data = utils.batchify(dataset.train,
        #                                  args.batch_size,
        #                                  self.cuda)
        # NOTE(brendan): The validation set data is batchified twice
        # separately: once for computing rewards during the Train Controller
        # phase (valid_data, batch size == 64), and once for evaluating ppl
        # over the entire validation set (eval_data, batch size == 1)
        self.train_data = dataset.train
        self.valid_data = dataset.valid
        self.test_data = dataset.test
        # self.max_length = self.args.shared_rnn_max_length

        if args.use_tensorboard:
            self.tb = TensorBoard(args.model_dir)
        else:
            self.tb = None
        #TODO initialize controller and shared model
        self.build_model()
        # print("11111111")
        if self.args.load_path:
            print("=======load_path=======")
            self.load_model()

        shared_optimizer = _get_optimizer(self.args.shared_optim)
        controller_optimizer = _get_optimizer(self.args.controller_optim)
        print("=======make optimizer========")
        self.shared_optim = shared_optimizer(
            self.shared.parameters(),
            lr=self.shared_lr,
            weight_decay=self.args.shared_l2_reg)
        print("=======make optimizer========")
        self.controller_optim = controller_optimizer(
            self.controller.parameters(),
            lr=self.args.controller_lr)

        self.ce = nn.CrossEntropyLoss()
        print("finish init")
    def build_model(self):
        """Creates and initializes the shared and controller models."""
        if self.args.network_type == 'rnn':
            self.shared = models.RNN(self.args, self.dataset)
        elif self.args.network_type == 'cnn':
            print("----- begin to init cnn------")
            self.shared = models.CNN(self.args, self.dataset)
            # self.shared = self.shared.cuda()
        else:
            raise NotImplementedError(f'Network type '
                                      f'`{self.args.network_type}` is not '
                                      f'defined')
        print("---- begin to init controller-----")
        self.controller = models.Controller(self.args)
        #self.controller = self.controller.cuda()
        print("===begin to cuda")
        if True:
            print("cuda")
            self.shared.cuda()
            self.controller.cuda()
            print("finish cuda")
        elif self.args.num_gpu > 1:
            raise NotImplementedError('`num_gpu > 1` is in process')


    def train(self):
        """Cycles through alternately training the shared parameters and the
        controller, as described in Section2.4 Training ENAS and deriving
        Architectures, of the paraer.
        """
        if self.args.shared_initial_step > 0:
            self.train_shared(self.args.shared_initial_step)
            self.train_controller()

        for self.epoch in range(self.start_epoch, self.args.max_epoch):
            # 1. Training the shared parameters omega of the child models
            self.train_shared()

            # 2. Training the controller parameters theta
            self.train_controller()

            if self.epoch % self.args.save_epoch == 0:
                with _get_no_grad_ctx_mgr():
                    best_dag = self.derive()
                    self.evaluate(self.eval_data,
                                  best_dag,
                                  'val_best',
                                  max_num=self.args.batch_size*100)
                self.save_model()

            if self.epoch >= self.args.shared_decay_after:
                utils.update_lr(self.shared_optim, self.shared_lr)

    def get_loss(self, inputs, targets, dags):
        """Computes the loss for the same batch for M models.

        This amounts to an estimate of the loss, which is turned into an
        estimate for the gradients of the shared model.
        """
        if not isinstance(dags, list):
            dags = [dags]

        loss = 0
        for dag in dags:
            inputs = Variable(inputs.cuda())
            targets = Variable(targets.cuda())
            # inputs = inputs.cuda()
            #targets = targets.cuda()
            #self.shared = self.shared.cuda()
            output  = self.shared(inputs, dag)
            sample_loss = (self.ce(output, targets) /
                           self.args.shared_num_sample)
            loss += sample_loss

        assert len(dags) == 1, 'there are multiple `hidden` for multiple `dags`'
        return loss

    def train_shared(self, max_step=None):
        """Train the image classification model for 310 steps
        """
        #TODO check if it is right that create a new dag for every batch and may be
        #one epoch one bathc will improve efficient
        model = self.shared
        model.train()
        self.controller.eval()

        if max_step is None:
            max_step = self.args.shared_max_step
        else:
            max_step = min(self.args.shared_max_step, max_step)

        step = 0
        raw_total_loss = 0
        total_loss = 0
        # train_idx = 0
        train_iter = iter(self.train_data)
        #TODO understanding how it train
        while True:
            if step > max_step:
                break
            dags = self.controller.sample(self.args.shared_num_sample)
            #print(dags)
            #TODO use iterator to create batch but need to add StopIteration
            #may be have some method to improve
            try:
                inputs, targets = train_iter.next()
            except StopIteration:
                print("====>train_shared<====== finish one epoch")
                break
                train_iter = iter(self.train_data)
            #print(dags)
            loss = self.get_loss(inputs,
                                targets,
                                dags)
            raw_total_loss += loss.data
            #TODO understand penality
            # loss += _apply_penalties()
            self.shared_optim.zero_grad()
            loss.backward()

            self.shared_optim.step()

            total_loss += loss.data
            if step % 20 == 0:
                print("loss, ", total_loss, step, total_loss /(step+1))
		
            if ((step % self.args.log_step) == 0) and (step > 0):
                self._summarize_shared_train(total_loss, raw_total_loss)
                raw_total_loss = 0
                total_loss = 0

            step += 1
            self.shared_step += 1
                # train_idx += self.max_length
    def get_reward(self, dag, entropies, data_iter):
        """Computes the perplexity of a single sampled model on a minibatch of
        validation data.
        """
        if not isinstance(entropies, np.ndarray):
            entropies = entropies.data.cpu().numpy()
        try:
            inputs, targets = data_iter.next()
        except StopIteration:
            data_iter = iter(self.valid_data)
            inputs, targets = data_iter.next()
        #TODO 怎么做volidate
        valid_loss = self.get_loss(inputs, targets, dag)
        # convert valid_loss to numpy ndarray
        valid_loss = utils.to_item(valid_loss.data)

        valid_ppl = math.exp(valid_loss)

        # TODO we don't knoe reward_c
        if self.args.ppl_square:
            #TODO: but we do know reward_c =80 in the previous paper need to read previous paper
            R = self.args.reward_c / valid_ppl ** 2
        else:
            R = self.args.reward_c / valid_ppl

        if self.args.entropy_mode == 'reward':
            rewards = R + self.args.entropy_coeff * entropies
        elif self.args.entropy_mode == 'regularizer':
            rewards = R * np.ones_like(entropies)
        else:
            raise NotImplementedError(f'Unknown entropy mode: {self.args.entropy_mode}')

        return rewards

    def train_controller(self):
        """Fixes the shared parameters and updates the controller parameters.

        The controller is updated with a score function gradient estimator
        (i.e., REINFORCE), with the reward being c/valid_ppl. where valid_ppl
        is computed on a minibatch of vlaidation data.

        A moving average baseline is used.

        The controller is trained for 2000 steps per epoch (i.e.,
        first (Train Shared) phase -. Second (Train Controller) phase).
        """
        model = self.controller
        model.train()


        avg_reward_base = None
        baseline = None
        adv_history = []
        entropy_history = []
        reward_history = []
        valid_iter = iter(self.valid_data)
        total_loss = 0
        for step in range(self.args.controller_max_step):

            dags, log_probs, entropies = self.controller.sample(
                with_details=True)
            print(dags)
            np_entropies = entropies.data.cpu().numpy()

            with _get_no_grad_ctx_mgr():
                rewards = self.get_reward(dags,
                                          np_entropies,
                                          valid_iter)
            if 1 > self.args.discount > 0:
                rewards = discount(rewards, self.args.discount)

            reward_history.extend(rewards)
            entropy_history.extend(np_entropies)

            # moving average baseline
            if baseline is None:
                baseline = rewards
            else:
                decay = self.args.ema_baseline_decay
                baseline = decay * baseline + (1 - decay) * rewards

            adv = rewards - baseline
            adv_history.extend(adv)

            #policy loss
            loss = -log_probs*utils.get_variable(adv,
                                                self.cuda,
                                                requires_grad=False)
            if self.args.entropy_mode == 'regularizer':
                loss -= self.args.entropy_coeff * np_entropies

            loss = loss.sum()

            self.controller_optim.zero_grad()
            loss.backward()

            if self.args.controller_grad_clip > 0:
                torch.nn.utils.clip_grad_norm(model.parameters(),
                                              self.args.controller_grad_clip)
            self.controller_optim.step()

            total_loss += utils.to_item(loss.data)
            if step%20 ==0:
                print("total loss", total_loss, step, total_loss / (step+1))
            if ((step % self.args.log_step) == 0) and (step > 0):
                self._summarize_controller_train(total_loss,
                                                adv_history,
                                                entropy_history,
                                                reward_history,
                                                avg_reward_base,
                                                dags)
                reward_history, adv_history, entropy_history = [], [], []
                total_loss = 0
            self.controller_step += 1

            # prev_valid_idx = valid_idx
            # valid_idx = ((valid_idx + self.max_length) %
            #             (self.valid_data.size(0) - 1))

            # NOTE(brendan): Whenever we wrap around to the beginning of the
            # validation data, we reset the hidden states.

    def evaluate(self, test_iter, dag, name, batch_size=1, max_num=None):
        """Evaluate on the validation set.
        (lianqing)what is the data of source ?

        NOTE: use validation to check reward but test set is the same as valid set
        """
        self.shared.eval()
        self.controller.eval()

        # data = source[:max_num*self.max_length]
        total_loss = 0
        # pbar = range(0, data.size(0) - 1, self.max_length)
        while True:
            try:
                inputs, targets = next(test_iter)
            except StopIteration:
                print("========> finish evaluate on one epoch<======")
                break
                test_iter = iter(self.test_data)
                inputs, targets = next(test_iter)
                # inputs = Variable(inputs)
            #check if is train the controller will have what difference
            inputs = Variable(inputs.cuda())
            targets = Variable(output.cuda())
            # inputs = inputs.cuda()
            #targets = targets.cuda()
            output = self.shared(inputs,
                                dag,
                                is_train=False)
            # check is self.loss wil work ?:
            total_loss += len(inputs) * self.ce(output, targets).data
            ppl = math.exp(utils.to_item(total_loss) / (count + 1))

        val_loss = utils.to_item(total_loss) / len(data)
        ppl = math.exp(val_loss)
        #TODO it's fix for rnn need to fix for cnn
        self.tb.scalar_summary(f'eval/{name}_loss', val_loss, self.epoch)
        self.tb.scalar_summary(f'eval/{name}_ppl', ppl, self.epoch)
        print(f'eval | loss: {val_loss:8.2f} | ppl: {ppl:8.2f}')

    def derive(self, sample_num=None, valid_iter=None):
        """
        pass sample_num is always to 1 test if batch_size > 1 will work ? for controller.sample
        """
        if sample_num is None:
            sample_num = self.args.derive_num_sample

        dags, _, entropies = self.controller.sample(sample_num,
                                            with_details=True)
        max_R = 0
        best_dag = None
        for dag in dags:
            R, _ = self.get_reward(dag, entropies, valid_iter)
            if R.max() > max_R:
                max_R = R.max()
                best_dag = dag

        print(
        f'derive | max_R: {max_R:8.6f}')
        fname = (f'{self.epoch:03d}-{self.controller_step:06d}-'
                 f'{max_R:6.4}-best.png')
        path = os.path.join(self.args.model_dir, 'networks', fname)
        # utils.draw_network(best_dag, path)
        # self.tb.image_summary('derive/best', [path], self.epoch)

        return best_dag

    @property
    def shared_lr(self):
        degree = max(self.epoch - self.args.shared_decay_after + 1, 0)
        return self.args.shared_lr * (self.args.shared_decay ** degree)

    @property
    def controller_lr(self):
        return self.args.controller_lr
    @property
    def shared_path(self):
        return f'{self.args.model_dir}/shared_epoch{self.epoch}_step{self.shared_step}.pth'

    @property
    def controller_path(self):
        return f'{self.args.model_dir}/controller_epoch{self.epoch}_step{self.controller_step}.pth'

    def get_saved_models_info(self):
        paths = glob.glob(os.path.join(self.args.model_dir, '*.pth'))
        paths.sort()

        def get_numbers(items, delimiter, idx, replace_word, must_contain=''):
            return list(set([int(
                    name.split(delimiter)[idx].replace(replace_word, ''))
                    for name in basenames if must_contain in name]))

        basenames = [os.path.basename(path.rsplit('.', 1)[0]) for path in paths]
        epochs = get_numbers(basenames, '_', 1, 'epoch')
        shared_steps = get_numbers(basenames, '_', 2, 'step', 'shared')
        controller_steps = get_numbers(basenames, '_', 2, 'step', 'controller')

        epochs.sort()
        shared_steps.sort()
        controller_steps.sort()

        return epochs, shared_steps, controller_steps

    def save_model(self):
        torch.save(self.shared.state_dict(), self.shared_path)
        print(f'[*] SAVED: {self.shared_path}')

        torch.save(self.controller.state_dict(), self.controller_path)
        print(f'[*] SAVED: {self.controller_path}')

        epochs, shared_steps, controller_steps = self.get_saved_models_info()

        for epoch in epochs[:-self.args.max_save_num]:
            paths = glob.glob(
                os.path.join(self.args.model_dir, f'*_epoch{epoch}_*.pth'))

            for path in paths:
                utils.remove_file(path)

    def load_model(self):
        epochs, shared_steps, controller_steps = self.get_saved_models_info()

        if len(epochs) == 0:
            print(f'[!] No checkpoint found in {self.args.model_dir}...')
            return

        self.epoch = self.start_epoch = max(epochs)
        self.shared_step = max(shared_steps)
        self.controller_step = max(controller_steps)

        if self.args.num_gpu == 0:
            map_location = lambda storage, loc: storage
        else:
            map_location = None

        self.shared.load_state_dict(
            torch.load(self.shared_path, map_location=map_location))
        print(f'[*] LOADED: {self.shared_path}')

        self.controller.load_state_dict(
            torch.load(self.controller_path, map_location=map_location))
        print(f'[*] LOADED: {self.controller_path}')

    def _summarize_controller_train(self,
                                    total_loss,
                                    adv_history,
                                    entropy_history,
                                    reward_history,
                                    avg_reward_base,
                                    dags):
        """Logs the controller's progress for this training epoch."""
        cur_loss = total_loss / self.args.log_step

        avg_adv = np.mean(adv_history)
        avg_entropy = np.mean(entropy_history)
        avg_reward = np.mean(reward_history)

        if avg_reward_base is None:
            avg_reward_base = avg_reward

        print(
            f'| epoch {self.epoch:3d} | lr {self.controller_lr:.5f} '
            f'| R {avg_reward:.5f} | entropy {avg_entropy:.4f} '
            f'| loss {cur_loss:.5f}')

        # Tensorboard
        if self.tb is not None:
            self.tb.scalar_summary('controller/loss',
                                   cur_loss,
                                   self.controller_step)
            self.tb.scalar_summary('controller/reward',
                                   avg_reward,
                                   self.controller_step)
            self.tb.scalar_summary('controller/reward-B_per_epoch',
                                   avg_reward - avg_reward_base,
                                   self.controller_step)
            self.tb.scalar_summary('controller/entropy',
                                   avg_entropy,
                                   self.controller_step)
            self.tb.scalar_summary('controller/adv',
                                   avg_adv,
                                   self.controller_step)

            paths = []
            for dag in dags:
                fname = (f'{self.epoch:03d}-{self.controller_step:06d}-'
                         f'{avg_reward:6.4f}.png')
                path = os.path.join(self.args.model_dir, 'networks', fname)
                # utils.draw_network(dag, path)
                paths.append(path)

            self.tb.image_summary('controller/sample',
                                  paths,
                                  self.controller_step)

    def _summarize_shared_train(self, total_loss, raw_total_loss):
        """Logs a set of training steps."""
        cur_loss = utils.to_item(total_loss) / self.args.log_step
        # NOTE(brendan): The raw loss, without adding in the activation
        # regularization terms, should be used to compute ppl.
        cur_raw_loss = utils.to_item(raw_total_loss) / self.args.log_step
        ppl = math.exp(cur_raw_loss)

        print(f'| epoch {self.epoch:3d} '
                    f'| lr {self.shared_lr:4.2f} '
                    f'| raw loss {cur_raw_loss:.2f} '
                    f'| loss {cur_loss:.2f} '
                    f'| ppl {ppl:8.2f}')

        # Tensorboard
        if self.tb is not None:
            self.tb.scalar_summary('shared/loss',
                                   cur_loss,
                                   self.shared_step)
            self.tb.scalar_summary('shared/perplexity',
                                   ppl,
                                   self.shared_step)
