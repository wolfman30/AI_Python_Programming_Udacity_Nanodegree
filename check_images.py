#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND/intropylab-classifying-images/check_images.py
#                                                                             
# TODO: 0. Fill in your information in the programming header below
# PROGRAMMER: Andrew Wolf   
# DATE CREATED: 08/16/2018
# REVISED DATE:             <=(Date Revised - if any)
# REVISED DATE: 05/14/2018 - added import statement that imports the print 
#                           functions that can be used to check the lab
# PURPOSE: Check images & report results: read them in, predict their
#          content (classifier), compare prediction to actual value labels
#          and output results
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
##

# Imports python modules
import argparse
from time import time, sleep
from os import listdir
import pandas as pd
from pandas import DataFrame as df 

# Imports classifier function for using CNN to classify images 
from classifier import classifier 

# Imports print functions that check the lab
from print_functions_for_lab_checks import *

# Main program function defined below
def main():
    start_time = time()
    
    # Creates & retrieves command line arguments
    in_arg = get_input_args()
    
    # Creates pet image labels by creating a dictionary with key=filename and value=file label to be used
    # to check the accuracy of the classifier function
    answers_dic = get_pet_labels(in_arg.dir)
    
    # TODO: 4. Define classify_images() function to create the classifier 
    # labels with the classifier function uisng in_arg.arch, comparing the 
    # labels, and creating a dictionary of results (result_dic)
    results_dic = classify_images(in_arg.dir, answers_dic, in_arg.arch)
    
    # TODO: 5. Define adjust_results4_isadog() function to adjust the results
    # dictionary(result_dic) to determine if classifier correctly classified
    # images as 'a dog' or 'not a dog'. This demonstrates if the model can
    # correctly classify dog images as dogs (regardless of breed)
    adjust_results4_isadog(results_dic, in_arg.dogfile)
    
    # TODO: 6. Define calculates_results_stats() function to calculate
    # results of run and puts statistics in a results statistics
    # dictionary (results_stats_dic)
    results_stats_dic = calculates_results_stats(results_dic)

    # TODO: 7. Define print_results() function to print summary results, 
    # incorrect classifications of dogs and breeds if requested.
    print_results(results_dic, results_stats_dic, in_arg.arch, print_incorrect_dogs=True, print_incorrect_breed=True)

    # TODO: 1. Define end_time to measure total program runtime
    # by collecting end time
    end_time = time() - start_time

    # TODO: 1. Define tot_time to computes overall runtime in
    # seconds & prints it in hh:mm:ss format
    tot_time = '%s hours %s minutes %.2f seconds' % (end_time//3600, end_time%3600//60, end_time%3600%60)
    print("\n** Total Elapsed Runtime:", tot_time)



# TODO: 2.-to-7. Define all the function below. Notice that the input 
# paramaters and return values have been left in the function's docstrings. 
# This is to provide guidance for acheiving a solution similar to the 
# instructor provided solution. Feel free to ignore this guidance as long as 
# you are able to acheive the desired outcomes with this lab.

def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object. 
     3 command line arguments are created:
       dir - Path to the pet image files(default- 'pet_images/')
       arch - CNN model architecture to use for image classification(default-
              pick any of the following vgg, alexnet, resnet)
       dogfile - Text file that contains all labels associated to dogs(default-
                'dognames.txt'
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir','-d', type = str, default='pet_images/', help= 'directory containing all pet images')
    ap.add_argument('--arch','-a', type = str, default='resnet', help='One of three CNN architectures to choose from', 
                    choices=['vgg', 'alexnet', 'resnet'])
    ap.add_argument('--dogfile','-l', type = str, default='dognames.txt', help='Labels given to each image')
    return ap.parse_args()



def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels based upon the filenames of the image 
    files. Reads in pet filenames and extracts the pet image labels from the 
    filenames and returns these label as petlabel_dic. This is used to check 
    the accuracy of the image classifier model.
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by pretrained CNN models (string)
    Returns:
     pet_dic - Dictionary storing image filename (as key) and Pet Image
                     Labels (as value)  
    """
    filenames = listdir(image_dir) #captures all filenames from directory into a list
     #lower cases and splits each filename based on '_'
    filenames_lower = [filename.lower().split('_') for filename in filenames]
    
    labels = []
    for word_list in filenames_lower:
        word_list.pop(-1)
        labels.append(" ".join(word_list))
        
    pet_dic = dict(zip(filenames, labels))#combines two lists filenames and labels together into dictionary
    return pet_dic

def classify_images(images_dir, pet_dic, model):
    """
    Creates classifier labels with classifier function, compares labels, and 
    creates a dictionary containing both labels and comparison of them to be
    returned.
     PLEASE NOTE: This function uses the classifier() function defined in 
     classifier.py within this function. The proper use of this function is
     in test_classifier.py Please refer to this program prior to using the 
     classifier() function to classify images in this function. 
     Parameters: 
      images_dir - The (full) path to the folder of images that are to be
                   classified by pretrained CNN models (string)
      petlabel_dic - Dictionary that contains the pet image(true) labels
                     that classify what's in the image, where its' key is the
                     pet image filename & it's value is pet image label where
                     label is lowercase with space between each word in label 
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
     Returns:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)   where 1 = match between pet image and 
                    classifer labels and 0 = no match between labels
    """
    filenames = list(pet_dic.keys())#creates list of filenames from the keys of pet_dic
    
    pet_labels = list(pet_dic.values())#creates list of pet labels from the values of pet_dic
    
    #creates list of labels from classifier 
    classifier_labels = [classifier(images_dir + image, model).lower().strip() for image in filenames]
    
    #creates a list of indexes where the pet labels are found in the classifier labels accounting for the condition
    #of a pet label being within a longer classifier label that does not match
    matches = []
    for pet_label, classifier_label in zip(pet_labels, classifier_labels):
        found_idx = classifier_label.find(pet_label)
        matches.append(found_idx)
        #ensures that the found index is of a matching pair
        #if the classifier label and pet label do not match based on the following condition
        #the found index is changed to -1 for a non-match
        if ( ( found_idx == 0 and len(pet_label) == len(classifier_label)) or ( (found_idx == 0) or 
                (classifier_label[found_idx - 1] == " ") ) and (  ( found_idx + len(pet_label) == len(classifier_label) ) or                                           (classifier_label[found_idx + len(pet_label) :found_idx + len(pet_label) +1] in (" ",",") )  )): 
            found_idx = 0
        else:
            found_idx = -1
            
    #creates a list of match_scores of 0 and 1 centered on where the indexes are greater than or equal to 0
    match_scores = [1 if item>=0 else 0 for item in matches]
    
    #creates the dictionary with filenames as the keys and the values as lists comprised of pet labels, classifier labels, and match scores
    results_dic = dict(zip(filenames, [list((a, b, c)) for a, b, c in zip(pet_labels, classifier_labels, match_scores)]))
    
    return results_dic

def adjust_results4_isadog(results_dic, dogsfile):
    """
    Adjusts the results dictionary to determine if classifier correctly 
    classified images 'as a dog' or 'not a dog' especially when not a match. 
    Demonstrates if model architecture correctly classifies dog images even if
    it gets dog breed wrong (not a match).
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    --- where idx 3 & idx 4 are added by this function ---
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
     dogsfile - A text file that contains names of all dogs from ImageNet 
                1000 labels (used by classifier model) and dog names from
                the pet image files. This file has one dog name per line
                dog names are all in lowercase with spaces separating the 
                distinct words of the dogname. This file should have been
                passed in as a command line argument. (string - indicates 
                text file's name)
    Returns:
           None - results_dic is mutable data type so no return needed.
    """
    classifier_labels = [i[1] for i in list(results_dic.values())]
    pet_labels = [i[0] for i in list(results_dic.values())]
    dognames = []
    with open(dogsfile, 'r') as f: 
        for line in f:
            dognames.append(line.rstrip())
            
    pet_matches = [1 if boolean else 0 for boolean in [pet_label in dognames for pet_label in pet_labels]]
    classifier_matches = [1 if boolean else 0 for boolean in [classifier_label in dognames for classifier_label in classifier_labels]]
    combo= [list((a, b)) for a, b in zip(pet_matches, classifier_matches)]
        
    for a, b in zip(results_dic.keys(), combo):
        results_dic[a].extend(b)
    #check_classifying_labels_as_dogs(results_dic)
    #print(results_dic)
   

def calculates_results_stats(results_dic):
    """
    Calculates statistics of the results of the run using classifier's model 
    architecture on classifying images. Then puts the results statistics in a 
    dictionary (results_stats) so that it's returned for printing as to help
    the user to determine the 'best' model for classifying images. Note that 
    the statistics calculated as the results are either percentages or counts.
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
    Returns:
     results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's 
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value 
    """
    n_correctly_classified_as_dogs = [i[3]==1 and i[4]==1 for i in results_dic.values()].count(True)
    n_correctly_classified_not_dogs = [i[3]==0 and i[4]==0 for i in results_dic.values()].count(True)
    n_correctly_classified_breed = [i[2]==1 and i[3]==1 for i in results_dic.values()].count(True)
  
    results_stats = {}
    results_stats['n_images'] = len(results_dic.keys())
    results_stats['n_dog_imgs'] = [i[3] for i in list(results_dic.values())].count(1)
    results_stats['n_not_dog'] = results_stats['n_images'] - results_stats['n_dog_imgs']
    
    results_stats['pct_correctly_classified_as_dogs'] = (n_correctly_classified_as_dogs/results_stats['n_dog_imgs']) * 100
    results_stats['pct_correctly_classified_not_dogs'] = (n_correctly_classified_not_dogs/results_stats['n_not_dog']) * 100
    results_stats['pct_correctly_classified_breed'] = (n_correctly_classified_breed/n_correctly_classified_as_dogs) * 100
    results_stats['pct_pet_class_match'] = [results_dic[key][2] for key in results_dic].count(1)/results_stats['n_images'] * 100
    results_stats = pd.Series(results_stats)
    print()
    print(results_stats)
    print()
    return results_stats
    
    


def print_results(results_dic, results_stats, model, print_incorrect_dogs=False, print_incorrect_breed=False):
    """
    Prints summary results on the classification and then prints incorrectly 
    classified dogs and incorrectly classified dog breeds if user indicates 
    they want those printouts (use non-default values)
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
      results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's 
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value 
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
      print_incorrect_dogs - True prints incorrectly classified dog images and 
                             False doesn't print anything(default) (bool)  
      print_incorrect_breed - True prints incorrectly classified dog breeds and 
                              False doesn't print anything(default) (bool) 
    Returns:
           None - simply printing results.
    """ 
    if print_incorrect_dogs:
        for i in results_dic.values():
            if (i[2] == 0 and i[3]==0 and i[4]==1):
                print('Model:', model, '|','Images MISCLASSIFIED as Dogs --', 
                     'Pet label:',i[0], '|', 'Classified label:', i[1])
    if print_incorrect_breed:
        for i in results_dic.values():
            if (i[2]==0 and i[3]==1 and i[4]==1):
                print('\n', 'Model:', model, '|', 'Classified Breed:', i[1],'|', 'Actual Breed:', i[0])
        

                
                
# Call to main function to run the program
if __name__ == "__main__":
    main()
