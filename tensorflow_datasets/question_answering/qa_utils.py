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

"""Shared utilities for QA datasets."""
import json

from absl import logging
import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds


SQUADLIKE_FEATURES = tfds.features.FeaturesDict({
    "id":
        tf.string,
    "title":
        tfds.features.Text(),
    "context":
        tfds.features.Text(),
    "question":
        tfds.features.Text(),
    "answers":
        tfds.features.Sequence({
            "text": tfds.features.Text(),
            "answer_start": tf.int32,
        }),
})

SQUADV2LIKE_FEATURES = tfds.features.FeaturesDict({
    "id":
        tf.string,
    "title":
        tfds.features.Text(),
    "context":
        tfds.features.Text(),
    "plausible_answers":
        tfds.features.Sequence({
            "text": tfds.features.Text(),
            "answer_start": tf.int32,
        }),
    "question":
        tfds.features.Text(),
    "answers":
        tfds.features.Sequence({
            "text": tfds.features.Text(),
            "answer_start": tf.int32,
        }),
    "is_impossible": tf.bool,
})


def generate_squadlike_examples(filepath):
  """Parses a SQuAD-like JSON, yielding examples with `SQUADLIKE_FEATURES`."""
  logging.info("generating examples from = %s", filepath)

  # We first re-group the answers, which may be flattened (e.g., by XTREME).
  qas = {}
  with tf.io.gfile.GFile(filepath) as f:
    squad = json.load(f)
    for article in squad["data"]:
      title = article.get("title", "").strip()
      for paragraph in article["paragraphs"]:
        context = paragraph["context"].strip()
        for qa in paragraph["qas"]:
          id_ = qa["id"]
          if id_ in qas:
            qas[id_]["answers"].extend(qa["answers"])
          else:
            qas[id_] = qa

    for id_, qa in qas.items():
      question = qa["question"].strip()
      answer_starts = [answer["answer_start"] for answer in qa["answers"]]
      answers = [answer["text"].strip() for answer in qa["answers"]]
      # Features currently used are "context", "question", and "answers".
      # Others are extracted here for the ease of future expansions.
      yield id_, {
          "title": title,
          "context": context,
          "question": question,
          "id": id_,
          "answers": {
              "answer_start": answer_starts,
              "text": answers,
          },
      }


def generate_squadv2like_examples(filepath):
  """Parses a SQuADV2-like JSON, yielding examples with `SQUADV2LIKE_FEATURES`."""
  logging.info("generating examples from = %s", filepath)

  # We first re-group the answers, which may be flattened (e.g., by XTREME).
  qas = {}
  with tf.io.gfile.GFile(filepath) as f:
    squad = json.load(f)
    for article in squad["data"]:
      title = article.get("title", "").strip()
      for paragraph in article["paragraphs"]:
        context = paragraph["context"].strip()
        for qa in paragraph["qas"]:
          id_ = qa["id"]
          if id_ in qas:
            qas[id_]["answers"].extend(qa["answers"])
          else:
            qas[id_] = qa

    for id_, qa in qas.items():
      #  Not all examples have plausible answers
      if "plausible_answers" not in qa:
        qa["plausible_answers"] = []

      question = qa["question"].strip()
      is_impossible = qa["is_impossible"]

      plausible_answer_starts = [
          plausible_answer["answer_start"]
          for plausible_answer in qa["plausible_answers"]
      ]
      plausible_answers = [
          plausible_answer["text"]
          for plausible_answer in qa["plausible_answers"]
      ]

      answer_starts = [answer["answer_start"] for answer in qa["answers"]]
      answers = [answer["text"].strip() for answer in qa["answers"]]

      # Features currently used are "context", "question", and "answers".
      # Others are extracted here for the ease of future expansions.
      yield id_, {
          "title": title,
          "context": context,
          "question": question,
          "id": id_,
          "plausible_answers": {
              "answer_start": plausible_answer_starts,
              "text": plausible_answers,
          },
          "answers": {
              "answer_start": answer_starts,
              "text": answers,
          },
          "is_impossible": is_impossible,
      }
