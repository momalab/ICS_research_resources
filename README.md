# ICS Research Resources
This repository contains the constructed datasets as well as trained models for the paper: "I came, I saw, I hacked: Automated Generation of Process-independent Attacks for Industrial Control Systems"  
Here, we demonstrate fingerprinting of various processes and sectors of an Industrial Control System (ICS) to gain automatically contruct attack vector that has intelligence to perform meaningful damage  
**Part 1: ICS Sector fingerprinting using HMI images**  
*Collected Images*  
In the current version of the dataset, we classify an HMI image to belong to one of the three sectors:  
1. Chemical  
2. Energy  
3. Water and wastewater  

The constructed dataset (each containing the sectors):  
Training images: /HMI Images/Images/Train.zip  
Test images: /HMI Images/Images/Test.zip  
  
*Text extracted from images*  
The raw data extracted from the images using OCR: /HMI Images/HMI_text/TextResults_(Test/Train).txt  
The translated data from the images: /HMI Images/HMI_text/TranslatedText_(Test/Train).txt  
The cleaned data from the images: /HMI Images/HMI_text/TransTextClean_(Test/Train).txt  

*Models*  
Final trained model using VGG16 architecture to be applied on images: /HMI Images/Models/NDSS_VGG16_078.7z  
Final trained model using Multinomial Naive Bayes to be applied on cleaned text: /HMI Images/Models/MNB.sav  

*Training Scripts*  
All the training scripts used to train the architectures discussed in the paper  /HMI Images/Training Scripts/  

*Sample Evaluation script*  
A sample script to utilize the models to predict accuracy: Evaluation_final_script.py
Please note, there might be a difference in test accuracy of ~2%. We encourage advancement of other architectures and models to train the constructed dataset.  

# Cite us!
E. Sarkar, H. Benkraouda, M. Maniatakos  
"I came, I saw, I hacked: Automated Generation of Process-independent Attacks for Industrial Control Systems"  
ACM Asia Conference on Computer and Communications Security (AsiaCCS 2020)


  
