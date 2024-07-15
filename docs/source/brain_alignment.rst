Calculating Brain Alignment
===========================
Brain alignment is a measure of how well a model's representation aligns with the brain's representation. Specifically, it measures how well a model's hidden representation can linearly predict brain activity. Herein, we briefly how to 
compute this score. Note that this overview only applies to language models and brain activity gathered from a linguistic stimuli. 

#. Prepare the brain activity data

#. Prepare the model's hidden representations

#. Train an encoder model that predicts brain activity from the model's hidden representations

#. Evaluating the encoder model on held-out data.

Each of these steps are crucial and depending on the application, there exists a variety of ways to perform each step. In the sections that follow, we detail their trade-offs and considerations, respectively. 


.. contents:: Table of Contents


Preprocessing Brain Activity Data
----------------------------------

Most of the datasets that we work will measure brain activity with an fMRI. fMRI measures the blood oxygenation level dependent (BOLD) signal, which is a proxy for neural activity. A stimulus is presented to each subject while they are in the scanner and measurements are taken at fixed time intervals. This process is shown in :numref:`fmri_measurement`.

.. _fmri_measurement:
.. figure:: _static/fmri.jpeg
    :align: center
    :width: 500px
    :alt: fMRI Measurement 

    Depending on the modality of the stimulus, the correspondance between voxel-level brain activity and the stimulus can vary greatly. For example, if the stimulus is given in the form of an audio recording, depending on the syllables in each word, between any two fMRI measurements there could be a variable number of words. On the other hand, if the stimulus is presented as text, generally researchers will present the text word-by-word at fixed time intervals (e.g. 0.5 seconds) to the participants. 

The fMRI data is collected by voxels. Each voxel is generally a :math:`2\text{mm}\times 2\text{mm}\times 2\text{mm}` cube in the brain. Intuitively, **the data collected is autoregressive** in the sense that the fMRI measurement taken at :math:`t=4` should be highly correlated with the measurements at :math:`t=0,t=2,t=6,\ldots`. Therefore, one needs to be extremely careful when training/evaluating a regressor that predicts new fMRI measurements. 

The timing of the stimulus is, as mentioned before, crucial. Thus, the time step of each event is generally provided with the fMRI measurements. Note that since each subject's brain is different in size and shape, the number of voxels 
and their relative locations can vary. Therefore, it almost never makes sense to average voxel data across subjects. 

With these considerations in mdin, the general pipeline for preprocessing this data is as follows:

#. *(Optional)* For each fMRI measurement, average neighboring voxels together according to a pre-defined atlas (what brain region they are a part of). This can speed up the calculation of alignment scores without losing much interpretability (unlike PCA). 

#. Separate the fMRI data into folds for cross-validation across the time dimension. More about this in :ref:`Training the Encoder Model`.

#. Normalize the fMRI data for each subject across all voxels. 


Since each dataset has its own quirks with how they're presenting the stimulus and performing the fMRI measurements, please refer to the `dataset-specific documentation <circuit_brain.dproc.html>`_.

.. important:: 
    Normalization across the voxel dimension should always be performed after the data is separated into training and test folds. Otherwise, there will be data leakage as the model has access to the test mean during training. 


Computing the Model's Hidden Representations
---------------------------------------------
We are mostly going to be working with Transformer models. Thus, we define a hidden layer of the model as the residual connection after a self-attention layer. The activations of this hidden layer should have dimensions :math:`\text{batch_size}\times \text{seq_length}\times \text{embedding_dim}`. The general pipeline for computing these hidden representations is displayed in :numref:`model_repr_pipeline`. 

#. (Only for reading stimuli) Join the words presented to the participants into a single string with spaces. Also remove/replace any special formatting characters. For example, in the :ref:`Harry Potter dataset <harrypotter>`, the authors of the dataset replace all newlines and italics with "+" and "@", respectively. 

#. Tokenize the entire stimuli. Keep track of where each word starts and ends relative to the tokens (and punctuation). 

#. Fix some context length, :math:`n_{\text{ctx}}`, and slide a window of this length over the tokens. For each window, compute the hidden representations of the model. Thus we should yield a tensor of shape :math:`\text{num_windows}\times \text{batch_size}\times n_{\text{ctx}}\times \text{embedding_dim}`. For more information on how to choose this context length, refer to [#mt]_ and [#kw]_.

#. Partition the hidden representations into folds for cross-validation across the sequence dimension (more on this in :ref:`Training the Encoder Model`). 

#. Aggregate the hidden representations across the sequence dimension. 

#. (Optional) Perform dimensionality reduction on the hidden representations. 

#. Normalize the aggregated representations across the embedding dimension separately for the training and test fold. 


.. _model_repr_pipeline:
.. figure:: _static/pipeline.jpg
    :align: center
    :width: 500px
    :alt: Model Pipeline

    Pipeline for computing the model's hidden representations, then using these representations to derive a linear model that predicts brain activity. Note that we evaluate the fit of this model on held-out data with the Pearson correlation metric.

.. note:: 
    It is common practice to reduce the dimension of the aggregated hidden representations using PCA before training the encoder model. Otherwise, the linear model is underdetermined, as there are so few fMRI measurements. 


Partitioning the Token-Level Representations into Words
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Depending on the tokenization scheme, we may require averaging the tokens across the sequence dimension with respect to which words they belong to first (see :numref:`model_repr_aggregation`). This is somewhat tricky because during tokenization, we need to keep track of which word each token belongs to. 


.. caution::
    White space matters!!! Though it may seem easier to first separate the stimuli into individual words, tokenize these words and keep track of their indices, the resulting embeddings are non-trivially different from tokenizing the entire input with the original whitespace. For example,

    .. code-block:: python

        >>> from transformers import AutoTokenizer
        >>> tok = AutoTokenizer.from_pretrained("gpt2")
        >>> text = "Malfoy certainly did talk a lot about flying. He"
        >>> tok(text)

        ["M", "alf", "oy", " certainly", " did", " talk", " a", " lot", " about", " flying", ".", " He"]
        [44, 1604, 726, 3729, 750, 1561, 257, 1256, 546, 7348, 13, 679]

        >> tok(text.split(" "))

        ["M", "alf", "oy", "certain", "ly", "did" "talk", "a", "lot", "about", "flying", ".", "He"]
        [44, 1604, 726, 39239, 306, 20839, 16620, 64, 26487, 10755,45928, 1544]

To get the token-word correspondance, we apply a greedy strategy for a given word we continuously de-tokenize until we have recovered the word entirely. Then, the set of tokens that correspond to this word are the tokens we had to de-tokenize set minus the tokens we have already assigned. 


Aggregating the Hidden Representations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _model_repr_aggregation:
.. figure:: _static/hidden-repr.jpeg
    :align: center
    :width: 500px
    :alt: Hidden Representations

    Various schemes to aggregate the token-level hidden representations into a single tensor. Here, we introduce and implement four main strategies found in the literature: direct concatenation of word-level embeddings, convolution of the word-level embeddings with the canonical hemodynamic response function (HRF), mean-pooling of the word-level embeddings, and mean-pooling of the token-level embeddings. 

In this subsection, we describe the four main strategies to aggregate the token-level hidden representations into a single tensor. That is, how to transform :math:`\text{batch_size}\times \text{seq_length}\times \text{embedding_dim}`-dimension tensor into a :math:`\text{batch_size}\times d_{\text{agg}}`-dimension tensor.

To generate the word-level hidden representations, we need to keep track of the set of tokens that correspond to each word. Then, we can average the token-level hidden representations according to this set to yield the word-level hidden representations. This process is shown in :numref:`model_repr_aggregation`.

.. glossary::

    Direct concatenation

        After computing the word-level hidden representations from the token-level hidden representations, we simply concatenate all of these word-level tensors together. If this is the chosen aggregation strategy, some sort of dimensionality reduction needs to be performed before training the encoder model since the number of features will be too large and the linear system could be underdetermined. 

    Convolution with HRF

        The canonical hemodynamic response function (HRF) describes the relationship between neural activity and the BOLD signal. Since the fMRI measures blood oxygenation, there is a time delay between the actual neural activity and the oxygenation of that region. This time delay is captured by the HRF. Thus, instead of performing a uniform average across the word dimension, we can convolute the word-level hidden representations with the HRF, then concatenating the resulting tensors. This strategy is significantly more complicated since each brain region has its own HRF. For simplicity, one may also convolve with a single simplified HRF across all regions. It should also be noted that between subjects the actual HRF can vary greatly due to differences in physiology and other external factors [#hrf]_.


    Word-Level Average

        After computing the word-level hidden representations from the token-level hidden representations, we average across the word dimension. 

    Token-Level Average

        This is the simplest aggregation strategy. We directly average all hidden representations across the sequence dimension. The resulting tensor is most faithful to the original hidden representation, since we are not injecting any additional structural or synatical information by averaging across words beforehand, nor are we assuming a distribution of temporal correlation between words. 

All four of these strategies are implemented within `our library <circuit_brain.model.html>`_. The choice of aggregation strategy is crucial, as it can greatly affect the alignment score.


.. caution::
    These aggregation strategies vary in their faithfulness to the model's mechanisms as well as their inherent alignment with physiological properties of the brain. For example, though direct concatenation preserves the information from all tokens the resulting tensor lies in the direct sum of the individual embedding spaces. 

    On the other hand, when performing a word-level average, we are injecting additional information into the model (i.e. what a word is) that is not encoded in the model itself. The same issue exists with convolution with the HRF, except now we are injecting artificial attention information. 

    In this sense, I believe the token-level average to be the most faithful to the model's mechanisms, however, it lies the furthest away from the physiological properties of the brain. 


.. caution:: 
    In most cases where one uses either direct concatenation or convolution the dimension of the resulting tensor is too large (in the sense that any linear system which predicts brain activity from these features may be underdetermined). Therefore, a common fix for this is to reduce the dimension of this tensor through PCA. However, one needs to be careful that additional confounding factors are not introduced through this process. Say we have two models A, B, if PCA preserves 87%, 85% of the variance of model A and B's hidden representations respectively, it could be that if model A has higher brain alignment it is due purely to its ability to sparsity of its representation. 



Training the Encoder Model
--------------------------
Ultimately, we wish to train a linear encoder model :math:`f` such that :math:`f` maps the preprocessed model's hidden representation into brain activity. For the most part, we perform voxel-level prediction. In this subsection, we describe in detail how to train and evaluate this encoder model.  


Creating the Labeled Data
~~~~~~~~~~~~~~~~~~~~~~~~~
As mentioned previously, there is a delay between the stimulus and the BOLD response. This delay generally ranges from 0-12 seconds. Therefore, for each fMRI measurement, we sample the previous ~20 words depending on the rate the words are shown to the participants. Note that we perform this word-level sampling **after** we pass the entire context into the language model. Thus, for each ~20 word context and layer in the model, we have a single tensor extracted from the model's hidden representation (aggregated through one of the aggregation methods described above). We then train a linear regressor to predict the each voxel-level fMRI measurement using this context. 

The encoder model is a Ridge regressor with :math:`\alpha` determined through cross validation. 

Cross Validation
~~~~~~~~~~~~~~~~
Two layers of cross validation is performed. In the "outer loop", we split all of the data into :math:`n` folds, such that we train the model on :math:`n-1` folds and evaluate on the last fold. During each iteration of this "outer loop", we perform cross validation again to determine the optimal :math:`\alpha` for the ridge regression. Note that we essentially training a separate regression for each voxel. Therefore, we also choose a specific :math:`alpha` for each voxel. 

.. caution::
    Since the fMRI signal is autoregressive, we need to be careful not to leak test data into the training folds. Thus, when we partition the data into training and test folds, we trim off the edges of the training fold (the number of samples we trim should match the BOLD response delay). Also need to make sure that the context of the language model for the test set does not contain any tokens in the training set. 


Evaluating the Encoder Model
----------------------------
The model is evaluated using Pearson correlation on the held-out test set in each fold. Then, we average this score across folds to yield the final brain-alignment score. 


.. rubric:: References

.. [#mt] Toneva, Mariya, and Leila Wehbe. "Interpreting and improving natural-language processing (in machines) with natural language-processing (in the brain)." Advances in neural information processing systems 32 (2019).

.. [#kw] Aw, Khai Loong, and Mariya Toneva. ‘Training Language Models to Summarize Narratives Improves Brain Alignment’. The Eleventh International Conference on Learning Representations, 2023, https://openreview.net/forum?id=KzkLAE49H9b.

.. [#hrf] "Since HRF is influenced by non-neural factors, to date it has largely been considered as a confound or has been ignored in many analyses." --- Rangaprakash D, Tadayonnejad R, Deshpande G, O'Neill J, Feusner JD. FMRI hemodynamic response function (HRF) as a novel marker of brain function: applications for understanding obsessive-compulsive disorder pathology and treatment response. Brain Imaging Behav. 2021 Jun;15(3):1622-1640. doi: 10.1007/s11682-020-00358-8. PMID: 32761566; PMCID: PMC7865013.