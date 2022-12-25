# Generate-fake-tasks-using-CGAN

### This project uses and explain how to use GANs with tabular data not only with images

## Overview
For Mobile Crowdsensing System (MCS), a fake task attack is essential because it aims to overload the sensing servers in the MCS platform and use more energy from users' smart devices. Typically, an empirical model like the CrowdSenSim programme is used to generate fictitious tasks. Cybercriminals now use more sophisticated tools to create assaults. One of the most effective methods for creating synthetic samples is the Generative Adversarial Network (GAN). To generate comparable samples, GAN takes into account all of the data in the training dataset. This project uses GAN to construct fake tasks and test the effectiveness of fake task detection.<br>

## Important dependencies 
```python
pip install numpy 
pip install tensorflow-gpu==2.9.1
pip install pandas 
pip install seaborn 
pip install matplotlib 
pip install sklearn
pip install imbalanced-learn
```

## Project Main steps
1. Download MCS dataset which and Split the dataset into training dataset (80%) and test dataset (20%)
2. Implement classic classifiers (Adaboost and RF) and train them
3. Verify detection performance using test dataset and present results comparison in bar chart
4. Implement a CGAN model and train it.
5. Generate synthetic fake tasks via Generator network in CGAN after the training procedure
6. Mix the generated fake tasks with the original test dataset to obtain a new test dataset
7. Obtain Adaboost and RF detection performance using the new test dataset and present results in
bar chart
8. Consider the Discriminator to as the first level classifier
and RF/Adaboost as the second level classifier
![Project steps](https://drive.google.com/uc?export=view&id=1YuHxXGr96Zgg2h5zbb_uO4ejviF8cEo2)

## Results 
Machine learning models obtain accepted accuracies during training and testing phase before
the GAN phase.
After generate new data with generator the model was very confused to classify the data samples for which
class they belong to, where the accuracy was very low as the model was trying to classify generated data that
could be acceptable sample or not.
The discriminator tries to distinguish real data from the data created by the generator. After we apply that
output comes from using the generated data from generator with discriminator. I tried to map values to class
using average value but I got result totally different where the result has been improved from low accuracy
as shown in fig.2 to high accuracy.

References.
[1] Jupyter Notebook Viewer.
https://nbviewer.org/github/codyznash/GANs_for_Credit_Card_Data/blob/master/GAN_comparisons.ipynb.
[2] Santhanam, Sivasurya. “GANs from Scratch.” Medium, 11 Jan. 2020,https://towardsdatascience.com/gans-from-scratch-8f5da17b3fb4.

