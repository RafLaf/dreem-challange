### Understanding the instrument

[Dreem website with all the articles releated](https://dreem.com/research)
[Paper on the Dreem Headband](https://www.biorxiv.org/content/10.1101/662734v1.full)

What exactly are we measuring, and what are we missing compared to classical methods. Probably we can find almost all informations in p.s. I think that the absence of EOG should be taken into account, and I didn't get where the reference electrode is placed.

### Understanding sleep

What features distinguish each phase of sleep, and what features are in common, for every parameter we are measuring.

### Looking for methods

A very useful thing to do for Kaggle challenges is to look at previous solutions. If there are recent challenges or studies about sleep classification, we should definitely look into that and make a list of the methods used.

### Exploratory Data Analysis

Here we should look into how time series analysis is carried on usually, plot something useful and stuff.  
This will guide the next steps.

### Baseline

We will see at this stage how well a basic and established algorithm performs, so we can compare future results and understand what is an improvement and what is not.  
We'll record the performance on both our data and the test data, submitting the results, so that we can evaluate how much further models are overfitting.

### Feature extraction / Cleaning the data

Usually data in kaggle competitions is clean, but we could want to filter our electric potentials from eye/head movements (EEG signal is very low, and gets very easily dirty.

### Model selection

Now we'll try many models, and decide which to keep for further improvement. We will apply cross-validation and see which ones perform better.

### Parameters exploration

Here we'll basically iterate over ranges of parameters for every selected model, and see which parameter space works well.
