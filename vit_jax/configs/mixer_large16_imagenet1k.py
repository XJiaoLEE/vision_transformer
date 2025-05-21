# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ml_collections

from vit_jax.configs import common
from vit_jax.configs import models


def get_config():
  """Returns config for training Mixer-B/16 on cifar10."""
  config = common.get_config()
  config.unlock()
  config.model_type = 'Mixer'
  config.model = models.get_mixer_l16_config()
  config.dataset = 'imagenet2012'
  # Training Steps (Adjusted for ImageNet-1k)
  config.total_steps = 125_000  # ≈50 epochs for ImageNet-1k (1.28M images / batch=510)
  config.warmup_steps = 12_500  # 10% of total_steps (ViT convention)
  config.base_lr = 0.003

  config.pp = ml_collections.ConfigDict(
      {'train': 'train[:100%]', 'test': 'test', 'crop': 224})
  return config
