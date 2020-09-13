"""CT_lymph_nodes dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow_datasets.core import utils
import numpy as np
import os
import io
import pydicom

# BibTeX citation
_CITATION = """
\@misc{CT_Lymph_Nodes_Citation,
  doi = {10.1007/978-3-319-10404-1_65},
  url = {https://wiki.cancerimagingarchive.net/display/Public/CT+Lymph+Nodes#12d41e510fe547b59000cd90afb8dbf2},
  author = {Roth, Holger R., Lu, Le, Seff, Ari, Cherry, Kevin M., Hoffman, Joanne, Wang, Shijun, Liu, Jiamin, Turkbey, Evrim and Summers, Ronald M.},
  title = {A New 2.5D Representation for Lymph Node Detection Using Random Sets of Deep Convolutional Neural Network Observations},
  publisher = {Springer International Publishing},
  year = {2014},
}
@article{TCIA_Citation,
  author = {
    K. Clark and B. Vendt and K. Smith and J. Freymann and J. Kirby and
    P. Koppel and S. Moore and S. Phillips and D. Maffitt and M. Pringle and
    L. Tarbox and F. Prior
  },
  title = {{The Cancer Imaging Archive (TCIA): Maintaining and Operating a
  Public Information Repository}},
  journal = {Journal of Digital Imaging},
  volume = {26},
  month = {December},
  year = {2013},
  pages = {1045-1057},
}
"""
# TODO(CT_lymph_nodes):
_DESCRIPTION = """
"""


class CtLymphNodes(tfds.core.GeneratorBasedBuilder):
  """TODO(CT_lymph_nodes): Short description of my dataset."""

  # TODO(CT_lymph_nodes): Set up version.
  VERSION = tfds.core.Version('0.1.0')

  def _info(self):
    # TODO(CT_lymph_nodes): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        # This is the description that will appear on the datasets page.
        description=_DESCRIPTION,
        # tfds.features.FeatureConnectors
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
        }),
        # If there's a common (input, target) tuple from the features,
        # specify them here. They'll be used if as_supervised=True in
        # builder.as_dataset.
        supervised_keys=(),
        # Homepage of the dataset for documentation
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    # TODO(CT_lymph_nodes): Downloads the data and defines the splits
    # dl_manager is a tfds.download.DownloadManager that can be used to
    # download and extract URLs
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={},
        ),
    ]

  def _generate_examples(self):
    """Yields examples."""
    # TODO(CT_lymph_nodes): Yields (key, example) tuples from the dataset
    yield 'key', {}
