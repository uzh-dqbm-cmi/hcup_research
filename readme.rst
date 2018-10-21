README
======

This repository includes analysis scripts and modules for working with the `HCUP readmission dataset <https://www.hcup-us.ahrq.gov/db/nation/nrd/nrddbdocumentation.jsp>`__.

Dataset folder
==============
Place the NRD databases unzipped in the ``dataset`` folder.

Src folder
==========
The ``src`` folder includes neural models implementation using ``PyTorch`` (version 0.3.x), in addition to training/evaluation workflow and utilities found in the ``train_eval.py`` and ``utilities.py`` modules respectively.

Notebooks
=========
The ``notebooks`` folder includes jupyter notebooks to perform the following:

Data processing for the HCUP NRD databases
------------------------------------------

    - ``data_processing.ipynb``: Go through the cells of the notebook sequentially to process `HCUP readmission dataset <https://www.hcup-us.ahrq.gov/db/nation/nrd/nrddbdocumentation.jsp>`__.

Neural models
-------------
    - ``neural_models_dataset_gen.ipynb``: To prepare and generate the dataset in a suitable format for training/evaluating the neural models using 5-folds cross validation.
    - ``optimize_hyperparams_neural_models.ipynb``: Run this notebook to identify **quasi-best** hyperparams for neural models.
    - ``train_eval_neural_models.ipynb``: Train and evaluate the neural models.

Baseline models
---------------
    - ``train_eval_baseline_models.ipynb``: Train and evaluate baseline models.

Performance evaluation
----------------------
    - ``performance_evaluation_decoded_output.ipynb``: To post-process decoded output (i.e. predicted outcomes of the models) for generating performance evaluation.
    - ``run_cvAUC_R.ipynb``: Run the ``cvAUC`` package to analyze AUC (area under the ROC curve) performance of the models on the 5-folds (R kernel should be installed).
