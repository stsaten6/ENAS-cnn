"""A module with NAS controller-related code."""
import collections
import os

import torch
import torch.nn.functional as F

import utils


Node = collections.namedtuple('Node', ['id', 'name'])

def _construct_dags(prev_nodes, activations, func_names, num_blocks, args):
    """Construct a set of DAGs based on the action
    """
    if args.network_type == 'cnn':
        dags = []
        for nodes, func_ids in zip(prev_nodes, activations):
            dag = collections.defaultdict(list)
            #0 mean original image
            #node id mean from where
            dag[1] = [Node(0, func_names[func_ids[0]])]
            for jdx, (idx, func_id) in enumerate(zip(nodes, func_ids[1:])):
                 dag[jdx+2].append(Node(idx, func_names[func_id]))
            #the 13th node for avg pooling
            last_node = Node(num_blocks, 'avg')
            # print(dag)
            # print(num_blocks)
            # print(last_node)
            dag[num_blocks+1] = [last_node]
            dags.append(dag)

    return dags

class Controller(torch.nn.Module):
    """Based on
    https://github.com/pytorch/examples/blob/master/word_language_model/model.py

    TODO(brendan): RL controllers do not necessarily have much to do with
    language models

    Base the controller RNN on the GRU from:
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
    """
    def __init__(self, args):
        torch.nn.Module.__init__(self)
        self.args = args

        if self.args.network_type == 'rnn':
            # NOTE(brendan): `num_tokens` here is just the activation function
            # for every even step,
            self.num_tokens = [len(args.shared_rnn_activations)] #[]
            for idx in range(self.args.num_blocks):
                self.num_tokens += [idx + 1,
                                    len(args.shared_rnn_activations)]
            self.func_names = args.shared_rnn_activations
        elif self.args.network_type == 'cnn':
            self.num_tokens = [len(args.shared_cnn_types)] #[4]
            # print("----cnn_num_blocks", args.cnn_num_blocks)
            for idx in range( sum(args.cnn_num_blocks) - 1):
                # [4 2 4 3 4 4 4 5 4 6 4 7 4 8 4 9 4 10 4 11 4 12 4 ]
                self.num_tokens += [idx + 2,
                                    len(args.shared_cnn_types)]
                # larger 1 because can add use original image for skip connection
            self.func_names = args.shared_cnn_types

        num_total_tokens = sum(self.num_tokens)
        # if self.args.network_type == 'rnn':
        self.encoder = torch.nn.Embedding(num_total_tokens,
                                            args.controller_hid)
        self.lstm = torch.nn.LSTMCell(args.controller_hid, args.controller_hid)

        # TODO(brendan): Perhaps these weights in the decoder should be
        # shared? At least for the activation functions, which all have the
        # same size.
        self.decoders = []
        # print("4444444  ")
        for idx, size in enumerate(self.num_tokens):
            # 4 2 4 3 4 4 4 5 4 6 4 7 4 8 4 9 4 10 4 11 4 12 4
            decoder = torch.nn.Linear(args.controller_hid, size)
            self.decoders.append(decoder)

        self._decoders = torch.nn.ModuleList(self.decoders)
        # print("222222")
        self.reset_parameters()
        self.static_init_hidden = utils.keydefaultdict(self.init_hidden)

        def _get_default_hidden(key):
            return utils.get_variable(
                torch.zeros(key, self.args.controller_hid),
                self.args.cuda,
                requires_grad=False)

        self.static_inputs = utils.keydefaultdict(_get_default_hidden)
        # print("======finist init controler=======")
    def reset_parameters(self):
        #how to reset
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        for decoder in self.decoders:
            decoder.bias.data.fill_(0)

    def forward(self,  # pylint:disable=arguments-differ
                inputs,
                hidden,
                block_idx,
                is_embed):
        #The same as rnn's controller
        if not is_embed:
            embed = self.encoder(inputs)
        else:
            embed = inputs

        hx, cx = self.lstm(embed, hidden)
        logits = self.decoders[block_idx](hx)

        logits /= self.args.softmax_temperature

        # exploration
        if self.args.mode == 'train':
            logits = (self.args.tanh_c*F.tanh(logits))

        return logits, (hx, cx)

    def sample(self, batch_size=1, with_details=False, save_dir=None):
        """Samples a set of `args.num_blocks` many computational nodes from the
        controller, where each node is made up of an activation function, and
        each node except the last also includes a previous node.
        """
        #TODO: make it fit both for rnn and cnn
        if self.args.network_type == 'rnn':
            if batch_size < 1:
                raise Exception(f'Wrong batch_size: {batch_size} < 1')

            # [B, L, H]
            inputs = self.static_inputs[batch_size]
            hidden = self.static_init_hidden[batch_size]

            activations = []
            entropies = []
            log_probs = []
            prev_nodes = []
            # NOTE(brendan): The RNN controller alternately outputs an activation,
            # followed by a previous node, for each block except the last one,
            # which only gets an activation function. The last node is the output
            # node, and its previous node is the average of all leaf nodes.
            for block_idx in range(2*(self.args.num_blocks - 1) + 1):
                logits, hidden = self.forward(inputs,
                                              hidden,
                                              block_idx,
                                              is_embed=(block_idx == 0))

                probs = F.softmax(logits, dim=-1)
                log_prob = F.log_softmax(logits, dim=-1)
                # TODO(brendan): .mean() for entropy?
                entropy = -(log_prob * probs).sum(1, keepdim=False)

                action = probs.multinomial(num_samples=1).data
                selected_log_prob = log_prob.gather(
                    1, utils.get_variable(action, requires_grad=False))

                # TODO(brendan): why the [:, 0] here? Should it be .squeeze(), or
                # .view()? Same below with `action`.
                entropies.append(entropy)
                log_probs.append(selected_log_prob[:, 0])

                # 0: function, 1: previous node
                mode = block_idx % 2
                inputs = utils.get_variable(
                    action[:, 0] + sum(self.num_tokens[:mode]),
                    requires_grad=False)

                if mode == 0:
                    activations.append(action[:, 0])
                elif mode == 1:
                    prev_nodes.append(action[:, 0])

            prev_nodes = torch.stack(prev_nodes).transpose(0, 1)  #prev_nodes 是做什么的？   用来index的...
            activations = torch.stack(activations).transpose(0, 1)

            dags = _construct_dags(prev_nodes,
                                   activations,
                                   self.func_names,
                                   sum(self.args.cnn_num_blocks),
                                   self.args)

            if save_dir is not None:
                for idx, dag in enumerate(dags):
                    utils.draw_network(dag,
                                       os.path.join(save_dir, f'graph{idx}.png'))

            if with_details:
                return dags, torch.cat(log_probs), torch.cat(entropies)
        if self.args.network_type == 'cnn':
            if batch_size < 1:
                raise Exception(f'Wrong batch_size: {batch_size} < 1')

            # [B, L, H]
            inputs = self.static_inputs[batch_size]
            hidden = self.static_init_hidden[batch_size]
            cnn_functions = []
            entropies = []
            log_probs = []
            prev_nodes = []
            #NOTE The RNN controller alternately outputs an cnn function,
            #followed by a previous node, for each block except the last one,
            #which only gets an cnn function.
            #11*2 + 1 = 23  the first one not need to chose which index
            for block_idx in range(2*(sum(self.args.cnn_num_blocks) - 1) + 1):
                logits, hidden = self.forward(inputs,
                                              hidden,
                                              block_idx,
                                              is_embed=(block_idx == 0))
                probs = F.softmax(logits, dim=-1)
                log_prob = F.log_softmax(logits, dim=-1)
                #TODO understanding policy gradient and improve the code
                entropy = -(log_prob * probs).sum(1, keepdim=False)
                #use multinomial to chose
                #TODO may be skip connect can choose more than one layer
                action = probs.multinomial(num_samples=1).data
                selected_log_prob = log_prob.gather(
                    1, utils.get_variable(action, requires_grad=False))

                entropies.append(entropy)
                log_probs.append(selected_log_prob[:, 0])

                # 0: function, 1: previous node
                mode = block_idx % 2
                inputs = utils.get_variable(
                    action[:, 0] + sum(self.num_tokens[:mode]),
                    requires_grad=False)

                if mode == 0:
                    cnn_functions.append(action[:, 0])
                elif mode == 1:
                    prev_nodes.append(action[:, 0])
            prev_nodes = torch.stack(prev_nodes).transpose(0, 1)
            cnn_functions = torch.stack(cnn_functions).transpose(0, 1)

            dags = _construct_dags(prev_nodes,
                                   cnn_functions,
                                   self.func_names,
                                   sum(self.args.cnn_num_blocks),
                                   self.args)
            if save_dir is not None:
                for idx, dag in enumerate(dags):
                    utils.drwa_network(dag,
                                    os.path.join(save_dir, f'graph{idx}.png'))
            #TODO when with_details need to check
            if with_details:
                return dags, torch.cat(log_probs), torch.cat(entropies)
        return dags


    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.args.controller_hid)
        return (utils.get_variable(zeros, self.args.cuda, requires_grad=False),
                utils.get_variable(zeros.clone(), self.args.cuda, requires_grad=False))
