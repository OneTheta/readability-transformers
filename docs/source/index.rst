ReadabilityTransformers Documentation
====================================================

ReadabilityTransformers is a Python framework optimized for text readability-related tasks. This project was created out of competing in the `CommonLit Kaggle competition<https://www.kaggle.com/c/commonlitreadabilityprize>`_., and most examples will follow this use-case.


Installation
====================================================

You can install this package using pip:

.. code-block:: python

   pip install readability-transformers

This package offers coverage from **Python 3.6** and **PyTorch 1.6.0** or higher.

Usage
====================================================

This is an example inference code for the CommonLit model.

.. code-block:: python

   from readability_transformers import ReadabilityTransformer

   model = ReadabilityTransformer(
      "commonlit-bert-base-twostep",
      device="cpu",
      double=True
   )

   passages = ["This is a sample text that we want to extract readability scores from."]
  
   predictions = model.predict(batch, batch_size=batch_size)
   
   for passage, prediction in zip(passages, predictions):
      print("Passage:", passage)
      print("Prediction:, prediction)
      print()


Contact
=========================

Contact person: Chan Woo Kim, chanwkim01@gmail.com

https://chanwookim.com

https://1theta.com


Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

*This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.*



.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
