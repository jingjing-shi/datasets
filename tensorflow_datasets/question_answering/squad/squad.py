# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
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

"""SQUAD: The Stanford Question Answering Dataset."""

import os

import tensorflow_datasets.public_api as tfds
from tensorflow_datasets.question_answering import qa_utils

_CITATION = """\
@article{2016arXiv160605250R,
       author = {{Rajpurkar}, Pranav and {Zhang}, Jian and {Lopyrev},
                 Konstantin and {Liang}, Percy},
        title = "{SQuAD: 100,000+ Questions for Machine Comprehension of Text}",
      journal = {arXiv e-prints},
         year = 2016,
          eid = {arXiv:1606.05250},
        pages = {arXiv:1606.05250},
archivePrefix = {arXiv},
       eprint = {1606.05250},
}
"""

_DESCRIPTION = """\
Stanford Question Answering Dataset (SQuAD) is a reading comprehension \
dataset, consisting of questions posed by crowdworkers on a set of Wikipedia \
articles, where the answer to every question is a segment of text, or span, \
from the corresponding reading passage, or the question might be unanswerable.
"""

_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
_HOMEPAGE_URL = "https://rajpurkar.github.io/SQuAD-explorer/"


class SquadConfig(tfds.core.BuilderConfig):
  """BuilderConfig for SQUAD."""

  def __init__(self, *, train_file, dev_file, **kwargs):
    super(SquadConfig, self).__init__(**kwargs)
    self.train_file = train_file
    self.dev_file = dev_file


class Squad(tfds.core.GeneratorBasedBuilder):
  """SQUAD: The Stanford Question Answering Dataset."""

  BUILDER_CONFIGS = [
      SquadConfig(
          name="v1",
          version=tfds.core.Version(
              "1.1.0",
              "New split API (https://tensorflow.org/datasets/splits)"),
          description="Version 1.1.0 of SQUAD",
          train_file="train-v1.1.json",
          dev_file="dev-v1.1.json",
      ),

      SquadConfig(
          name="v2",
          version=tfds.core.Version(
              "2.0.0",
              "New split API (https://tensorflow.org/datasets/splits)"),
          description="Version 2.0.0 of SQUAD",
          train_file="train-v2.0.json",
          dev_file="dev-v2.0.json",
      ),
  ]

  def _info(self):

    features_dict = qa_utils.SQUADLIKE_FEATURES if self.builder_config.name == "v1" else qa_utils.SQUADV2LIKE_FEATURES

    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=features_dict,
        # No default supervised_keys (as we have to pass both question
        # and context as input).
        supervised_keys=None,
        homepage=_HOMEPAGE_URL,
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    urls_to_download = {
        "train": os.path.join(_URL, self.builder_config.train_file),
        "dev": os.path.join(_URL, self.builder_config.dev_file)
    }
    downloaded_files = dl_manager.download_and_extract(urls_to_download)

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={"filepath": downloaded_files["train"]}),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={"filepath": downloaded_files["dev"]}),
    ]

  def _generate_examples(self, filepath):
    if self.builder_config.name == "v1":
      return qa_utils.generate_squadlike_examples(filepath)
    return qa_utils.generate_squadv2like_examples(filepath)
