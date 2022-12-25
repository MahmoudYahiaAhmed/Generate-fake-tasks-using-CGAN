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
**The generated tasks from the generator are robust and succussed to fault the
classic ML algorithms because it is tried to generate tasks very close to the real
one, so the models can’t determine it and the accuracies has been decreased
from 0.92 to 0.575 in the Adaboost model and has been decreased from 0.993 to
0.590 in the Random Forest model.<br>
In the cascade approach the discriminator helped the models because it can filter
the fake tasks, so after the filtering it out the accuracies increased again to 0.926
in Adaboost and to 0.993 in the Random Forest model and this result is
approximately one before mixing which means that the discriminator filtered**



