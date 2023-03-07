from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import time
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

class Profiling(object):
    def __init__(self, model):
        if isinstance(model, torch.nn.Module) is False:
            raise ValueError("Not a valid model, please provide a 'nn.Module' instance.")

        self.model = model
        self._parameter_names = {v: k for k, v
                                in model.named_parameters()}
        self._seq_keys = [k for k, v in model.named_parameters()]
        self._backward_seq_keys = []
        self._backward_key_sizes = []
        self._grad_accs = []
        self._handles = {}
        self.hook_done = False
        self._start = time.time()
        self._register_hooks()
        self._is_profiling = False

    def _register_hooks(self):
        for name, p in self.model.named_parameters():
            p.register_hook(self._make_hook(name, p))

    def _make_hook(self, name, p):
        def hook(*ignore):
            if not self._is_profiling:
                return
            name = self._parameter_names.get(p)
            if len(self._backward_seq_keys) != len(self._seq_keys):
                self._backward_seq_keys.append(name)
                self._backward_key_sizes.append(p.numel())
            if name not in self._handles:
                self._handles[name] = []
            torch.cuda.synchronize()
            ct = self._timestamp(name)
            self._handles[name].append(ct - self._start)
        return hook

    def reset_start(self):
        self._start = time.time()

    def reset(self):
        self._start = time.time()
        self._handles.clear()

    def stop(self):
        self._is_profiling = False

    def start(self):
        self._is_profiling = True
        self._start = time.time()

    def get_backward_seq_keys(self):
        return self._backward_seq_keys

    def get_backward_key_sizes(self):
        return self._backward_key_sizes

    def get_layerwise_times(self):
        num_trials = len(self._handles[self._seq_keys[0]])
        layerwise_times_multipletest = []
        totals = []
        for j in range(num_trials):
            s = 0
            total = 0.0
            layerwise_times = [] # from the last layer to the first layer
            #for i, k in enumerate(self._seq_keys[::-1]):
            for i, k in enumerate(self._backward_seq_keys):
                t = self._handles[k][j]
                #print('name: ', k, ' diff: ', t-s)
                layerwise_times.append(t-s)
                total += (t-s)
                s = total
            layerwise_times_multipletest.append(layerwise_times)
            totals.append(total)
        array = np.array(layerwise_times_multipletest)
        layerwise_times = np.mean(array, axis=0)
        return layerwise_times, np.mean(totals)

    def _timestamp(self, name):
        return time.time()


def benchmark(trainer):
    # Benchmark to achieve the backward time per layer
    p = Profiling(trainer.net)
    # Warmup
    input_shape, output_shape = trainer.get_data_shape()
    warmup = 5 # warmup should be 0 on some GPUs (e.g., P102-100)
    iteration = 50

    for i in range(iteration+warmup):
        data = trainer.data_iter()

        inputs, labels_cpu = data
        if trainer.is_cuda:
            inputs, labels = inputs.cuda(non_blocking=True), labels_cpu.cuda(non_blocking=True)
        else:
            labels = labels_cpu

        # forward + backward + optimize
        if trainer.dnn == 'inceptionv3':
            _, outputs = trainer.net(inputs)
        else:
            outputs = trainer.net(inputs)
        loss = trainer.criterion(outputs, labels)
        torch.cuda.synchronize()

        if i >= warmup:
            p.start()
        loss.backward()
        if trainer.is_cuda:
            torch.cuda.synchronize()
    layerwise_times, sum_total = p.get_layerwise_times()
    seq_keys = p.get_backward_seq_keys()
    p.stop()
    return seq_keys[::-1], layerwise_times[::-1], p.get_backward_key_sizes()[::-1]

class CommunicationProfiler(object):
    def __init__(self, comm_op, sync_op, sizes=None):
        self.comm_op = comm_op
        self.sync_op = sync_op
        self.sizes = sizes

    def benchmark(self, num_iters=1):
        if self.sizes is None:
            # large_sizes = [4000 for i in range(10)] # 1M to 512M
            sizes = 4000000
        else:
            sizes = self.sizes
        warmup = 10
        size = 1024
        tensor = torch.rand(size).float().cuda()
        stime = time.time()
        for i in range(warmup):
            name = 'warmup-%d' % i
            h = self.comm_op(tensor, average=True, name=name)
            self.sync_op(h)
        etime = time.time()
        elapsed_times = []
        credits = [1, 2, 3, 4, 5]
        for i in credits:
            s = sizes
            tensor = torch.rand(s).float().cuda()
            torch.cuda.synchronize()
            stime = time.time()
            for iter in range(num_iters):
                if i == 1:
                    name1 = 'run-size%d-credit%d-%d'% (s, i, 1)
                    h1 = self.comm_op(tensor, average=True, name=name1)
                    self.sync_op(h1)
                if i == 2:
                    name1 = 'run-size%d-credit%d-%d'% (s, i, 1)
                    name2 = 'run-size%d-credit%d-%d' % (s, i, 2)
                    h1 = self.comm_op(tensor, average=True, name=name1)
                    h2 = self.comm_op(tensor, average=True, name=name2)
                    self.sync_op(h1)
                    self.sync_op(h2)
                if i == 3:
                    name1 = 'run-size%d-credit%d-%d'% (s, i, 1)
                    name2 = 'run-size%d-credit%d-%d' % (s, i, 2)
                    name3 = 'run-size%d-credit%d-%d' % (s, i, 3)
                    h1 = self.comm_op(tensor, average=True, name=name1)
                    h2 = self.comm_op(tensor, average=True, name=name2)
                    h3 = self.comm_op(tensor, average=True, name=name3)
                    self.sync_op(h1)
                    self.sync_op(h2)
                    self.sync_op(h3)
                if i == 4:
                    name1 = 'run-size%d-credit%d-%d'% (s, i, 1)
                    name2 = 'run-size%d-credit%d-%d' % (s, i, 2)
                    name3 = 'run-size%d-credit%d-%d' % (s, i, 3)
                    name4 = 'run-size%d-credit%d-%d' % (s, i, 4)
                    h1 = self.comm_op(tensor, average=True, name=name1)
                    h2 = self.comm_op(tensor, average=True, name=name2)
                    h3 = self.comm_op(tensor, average=True, name=name3)
                    h4 = self.comm_op(tensor, average=True, name=name4)
                    self.sync_op(h1)
                    self.sync_op(h2)
                    self.sync_op(h3)
                    self.sync_op(h4)
                if i == 5:
                    name1 = 'run-size%d-credit%d-%d'% (s, i, 1)
                    name2 = 'run-size%d-credit%d-%d' % (s, i, 2)
                    name3 = 'run-size%d-credit%d-%d' % (s, i, 3)
                    name4 = 'run-size%d-credit%d-%d' % (s, i, 4)
                    name5 = 'run-size%d-credit%d-%d' % (s, i, 5)
                    h1 = self.comm_op(tensor, average=True, name=name1)
                    h2 = self.comm_op(tensor, average=True, name=name2)
                    h3 = self.comm_op(tensor, average=True, name=name3)
                    h4 = self.comm_op(tensor, average=True, name=name4)
                    h5 = self.comm_op(tensor, average=True, name=name5)
                    self.sync_op(h1)
                    self.sync_op(h2)
                    self.sync_op(h3)
                    self.sync_op(h4)
                    self.sync_op(h5)
                # if i == 6:
                #     name1 = 'run-size%d-credit%d-%d'% (s, i, 1)
                #     name2 = 'run-size%d-credit%d-%d' % (s, i, 2)
                #     name3 = 'run-size%d-credit%d-%d' % (s, i, 3)
                #     name4 = 'run-size%d-credit%d-%d' % (s, i, 4)
                #     name5 = 'run-size%d-credit%d-%d' % (s, i, 5)
                #     name6 = 'run-size%d-credit%d-%d' % (s, i, 6)
                #     h1 = self.comm_op(tensor, average=True, name=name1)
                #     h2 = self.comm_op(tensor, average=True, name=name2)
                #     h3 = self.comm_op(tensor, average=True, name=name3)
                #     h4 = self.comm_op(tensor, average=True, name=name4)
                #     h5 = self.comm_op(tensor, average=True, name=name5)
                #     h6 = self.comm_op(tensor, average=True, name=name6)
                #     self.sync_op(h1)
                #     self.sync_op(h2)
                #     self.sync_op(h3)
                #     self.sync_op(h4)
                #     self.sync_op(h5)
                #     self.sync_op(h6)
                # if i == 7:
                #     name1 = 'run-size%d-credit%d-%d'% (s, i, 1)
                #     name2 = 'run-size%d-credit%d-%d' % (s, i, 2)
                #     name3 = 'run-size%d-credit%d-%d' % (s, i, 3)
                #     name4 = 'run-size%d-credit%d-%d' % (s, i, 4)
                #     name5 = 'run-size%d-credit%d-%d' % (s, i, 5)
                #     name6 = 'run-size%d-credit%d-%d' % (s, i, 6)
                #     name7 = 'run-size%d-credit%d-%d' % (s, i, 7)
                #     h1 = self.comm_op(tensor, average=True, name=name1)
                #     h2 = self.comm_op(tensor, average=True, name=name2)
                #     h3 = self.comm_op(tensor, average=True, name=name3)
                #     h4 = self.comm_op(tensor, average=True, name=name4)
                #     h5 = self.comm_op(tensor, average=True, name=name5)
                #     h6 = self.comm_op(tensor, average=True, name=name6)
                #     h7 = self.comm_op(tensor, average=True, name=name7)
                #     self.sync_op(h1)
                #     self.sync_op(h2)
                #     self.sync_op(h3)
                #     self.sync_op(h4)
                #     self.sync_op(h5)
                #     self.sync_op(h6)
                #     self.sync_op(h7)
                # if i == 8:
                #     name1 = 'run-size%d-credit%d-%d'% (s, i, 1)
                #     name2 = 'run-size%d-credit%d-%d' % (s, i, 2)
                #     name3 = 'run-size%d-credit%d-%d' % (s, i, 3)
                #     name4 = 'run-size%d-credit%d-%d' % (s, i, 4)
                #     name5 = 'run-size%d-credit%d-%d' % (s, i, 5)
                #     name6 = 'run-size%d-credit%d-%d' % (s, i, 6)
                #     name7 = 'run-size%d-credit%d-%d' % (s, i, 7)
                #     name8 = 'run-size%d-credit%d-%d' % (s, i, 8)
                #     h1 = self.comm_op(tensor, average=True, name=name1)
                #     h2 = self.comm_op(tensor, average=True, name=name2)
                #     h3 = self.comm_op(tensor, average=True, name=name3)
                #     h4 = self.comm_op(tensor, average=True, name=name4)
                #     h5 = self.comm_op(tensor, average=True, name=name5)
                #     h6 = self.comm_op(tensor, average=True, name=name6)
                #     h7 = self.comm_op(tensor, average=True, name=name7)
                #     h8 = self.comm_op(tensor, average=True, name=name8)
                #     self.sync_op(h1)
                #     self.sync_op(h2)
                #     self.sync_op(h3)
                #     self.sync_op(h4)
                #     self.sync_op(h5)
                #     self.sync_op(h6)
                #     self.sync_op(h7)
                #     self.sync_op(h8)
                # if i == 9:
                #     name1 = 'run-size%d-credit%d-%d'% (s, i, 1)
                #     name2 = 'run-size%d-credit%d-%d' % (s, i, 2)
                #     name3 = 'run-size%d-credit%d-%d' % (s, i, 3)
                #     name4 = 'run-size%d-credit%d-%d' % (s, i, 4)
                #     name5 = 'run-size%d-credit%d-%d' % (s, i, 5)
                #     name6 = 'run-size%d-credit%d-%d' % (s, i, 6)
                #     name7 = 'run-size%d-credit%d-%d' % (s, i, 7)
                #     name8 = 'run-size%d-credit%d-%d' % (s, i, 8)
                #     name9 = 'run-size%d-credit%d-%d' % (s, i, 9)
                #     h1 = self.comm_op(tensor, average=True, name=name1)
                #     h2 = self.comm_op(tensor, average=True, name=name2)
                #     h3 = self.comm_op(tensor, average=True, name=name3)
                #     h4 = self.comm_op(tensor, average=True, name=name4)
                #     h5 = self.comm_op(tensor, average=True, name=name5)
                #     h6 = self.comm_op(tensor, average=True, name=name6)
                #     h7 = self.comm_op(tensor, average=True, name=name7)
                #     h8 = self.comm_op(tensor, average=True, name=name8)
                #     h9 = self.comm_op(tensor, average=True, name=name9)
                #     self.sync_op(h1)
                #     self.sync_op(h2)
                #     self.sync_op(h3)
                #     self.sync_op(h4)
                #     self.sync_op(h5)
                #     self.sync_op(h6)
                #     self.sync_op(h7)
                #     self.sync_op(h8)
                #     self.sync_op(h9)
                # if i == 10:
                #     name1 = 'run-size%d-credit%d-%d'% (s, i, 1)
                #     name2 = 'run-size%d-credit%d-%d' % (s, i, 2)
                #     name3 = 'run-size%d-credit%d-%d' % (s, i, 3)
                #     name4 = 'run-size%d-credit%d-%d' % (s, i, 4)
                #     name5 = 'run-size%d-credit%d-%d' % (s, i, 5)
                #     name6 = 'run-size%d-credit%d-%d' % (s, i, 6)
                #     name7 = 'run-size%d-credit%d-%d' % (s, i, 7)
                #     name8 = 'run-size%d-credit%d-%d' % (s, i, 8)
                #     name9 = 'run-size%d-credit%d-%d' % (s, i, 9)
                #     name10 = 'run-size%d-credit%d-%d' % (s, i, 10)
                #     h1 = self.comm_op(tensor, average=True, name=name1)
                #     h2 = self.comm_op(tensor, average=True, name=name2)
                #     h3 = self.comm_op(tensor, average=True, name=name3)
                #     h4 = self.comm_op(tensor, average=True, name=name4)
                #     h5 = self.comm_op(tensor, average=True, name=name5)
                #     h6 = self.comm_op(tensor, average=True, name=name6)
                #     h7 = self.comm_op(tensor, average=True, name=name7)
                #     h8 = self.comm_op(tensor, average=True, name=name8)
                #     h9 = self.comm_op(tensor, average=True, name=name9)
                #     h10 = self.comm_op(tensor, average=True, name=name10)
                #     self.sync_op(h1)
                #     self.sync_op(h2)
                #     self.sync_op(h3)
                #     self.sync_op(h4)
                #     self.sync_op(h5)
                #     self.sync_op(h6)
                #     self.sync_op(h7)
                #     self.sync_op(h8)
                #     self.sync_op(h9)
                #     self.sync_op(h10)
            etime = time.time()
            elapsed_times.append((etime-stime)/num_iters)
        return credits, elapsed_times