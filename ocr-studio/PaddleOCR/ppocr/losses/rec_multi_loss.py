# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn
import paddle.nn.functional as F

from .rec_ctc_loss import CTCLoss
from .rec_sar_loss import SARLoss
from .rec_nrtr_loss import NRTRLoss


class MultiLoss(nn.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_funcs = {}
        self.loss_list = kwargs.pop('loss_config_list')
        self.weight_1 = kwargs.get('weight_1', 1.0)
        self.weight_2 = kwargs.get('weight_2', 1.0)
        for loss_info in self.loss_list:
            for name, param in loss_info.items():
                if param is not None:
                    kwargs.update(param)
                loss = eval(name)(**kwargs)
                self.loss_funcs[name] = loss

    def forward(self, predicts, batch):
        self.total_loss = {}
        total_loss = 0.0
        # batch [image, label_ctc, label_sar, length, valid_ratio]
        for name, loss_func in self.loss_funcs.items():
            if name == 'CTCLoss':
                loss = loss_func(predicts['ctc'],
                                 batch[:2] + batch[3:])['loss'] * self.weight_1
            elif name == 'SARLoss':
                loss = loss_func(predicts['sar'],
                                 batch[:1] + batch[2:])['loss'] * self.weight_2
            elif name == 'NRTRLoss':
                loss = loss_func(predicts['nrtr'],
                                 batch[:1] + batch[2:])['loss'] * self.weight_2
            else:
                raise NotImplementedError(
                    '{} is not supported in MultiLoss yet'.format(name))
            self.total_loss[name] = loss
            total_loss += loss
        self.total_loss['loss'] = total_loss
        return self.total_loss


class BlankAwareMultiLoss(MultiLoss):
    """
    Blank-Aware Multi-Task Loss
    ===========================
    Feature-gated loss for blank-aware training.
    - CTCLoss: Evaluated on the entire batch (both normal and blank samples).
      Paddle's nn.CTCLoss natively supports target_length=0, providing active
      gradient penalty against character activations on blank inputs.
    - NRTRLoss / SARLoss (GTC): Evaluated ONLY on non-blank samples (length > 0).
      Blank rows are masked out to avoid shape mismatch / spurious token training.
    - All-blank batch edge case:
      If all samples in the batch have length == 0, GTC loss is safely computed
      as 0.0 * paddle.sum(predicts[key]) to maintain gradient graph integrity
      without contributing to total loss.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, predicts, batch):
        self.total_loss = {}
        total_loss = 0.0
        # batch: [image, label_ctc, label_gtc, length, valid_ratio, ...]
        length = batch[3]

        for name, loss_func in self.loss_funcs.items():
            if name == 'CTCLoss':
                pred_ctc = predicts['ctc']
                if length.max() == 0:
                    # All samples in this batch are blank:
                    # In CTC, empty target string aligns only to all-blank tokens (class 0).
                    # Exact NLL = -sum_t(log P(blank_t)).
                    log_probs = F.log_softmax(pred_ctc, axis=-1)
                    loss = -log_probs[:, :, 0].sum(axis=1).mean() * self.weight_1
                else:
                    loss = loss_func(pred_ctc,
                                     batch[:2] + batch[3:])['loss'] * self.weight_1
            elif name in ('NRTRLoss', 'SARLoss'):
                key = 'nrtr' if name == 'NRTRLoss' else 'sar'
                pred = predicts[key]
                non_blank_idx = paddle.nonzero(length > 0)
                if non_blank_idx.numel() == 0:
                    # All samples in this batch are blank: graph-safe 0 loss
                    loss = 0.0 * paddle.sum(pred)
                elif non_blank_idx.numel() == length.shape[0]:
                    # All samples are non-blank: standard path
                    loss = loss_func(pred, batch[:1] + batch[2:])['loss'] * self.weight_2
                else:
                    # Mixed batch: slice only non-blank samples for GTC loss
                    idx = non_blank_idx.squeeze(-1)
                    pred_nb = paddle.gather(pred, idx)
                    batch_nb = [paddle.gather(b, idx) if isinstance(b, paddle.Tensor) else b for b in (batch[:1] + batch[2:])]
                    loss = loss_func(pred_nb, batch_nb)['loss'] * self.weight_2
            else:
                raise NotImplementedError(
                    '{} is not supported in BlankAwareMultiLoss yet'.format(name))
            self.total_loss[name] = loss
            total_loss += loss

        self.total_loss['loss'] = total_loss
        return self.total_loss
