# Line of work

All that is written here can be changed, redacted, removed, and so on, this is just a basis to have something to discuss.

### Understanding the instrument

What exactly are we measuring, and what are we missing compared to classical methods. Probably we can find almost all informations in (the paper)[https://www.researchgate.net/publication/343143113_Dreem_Open_Datasets_Multi-Scored_Sleep_Datasets_to_Compare_Human_and_Automated_Sleep_Staging] about the dataset

p.s. I think that the absence of EOG should be taken into account, and I didn't get where the reference electrode is placed.

### Understanding sleep

What features distinguish each phase of sleep, and what features are in common, for every parameter we are measuring.

### Looking for methods

A very useful thing to do for Kaggle challenges is to look at previous solutions. If there are recent challenges or studyies about sleep classification, we should definetely look into that and make a list of the methods used.

### Exploratory Data Analysis

Here we should look into how time series analysis is carried on usually, plot something useful and stuff.  
This will guide the next steps.

### Baseline

We will see at this stage how well a basic and established algorithm performs, so we can compare future results and understand what is an improvement and what is not.  
We'll record the performance on both our data and the test data, submitting the results, so that we can evaluate how much further models are overfitting.

### Feature extraction / Cleaning the data

Usually data in kaggle competitions is clean, but we could want to filter our electric potentials from eye/head movements (EEG signal is very low, and gets very easily dirty.

### Model selection

Now we'll try many models, and decide which to keep for further improvement. We will apply cross-validation and see which ones performs better.

### Parameters exploration

Here we'll basically iterate over ranges of parameters for every selected model, and see which parameter space works well.

### Extra

Since the dataset is public, check if also the test set is public. We don't want to cheat, but we also want to be sure other people don't.
