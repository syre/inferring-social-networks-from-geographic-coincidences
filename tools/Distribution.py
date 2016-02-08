#!/usr/bin/env python3.5
from collections import defaultdict
import json

import matplotlib.pyplot as plt  
import numpy as np


class Distribution:

    def __init__(self):
        pass


    def open_file(self, filename):
        """Open a JSON file and return the dict
        
        Arguments:
            filename {string} -- Name of the JSON file to open
        
        Returns:
            dict -- The corresponding dicts of data from the JSON file
        """
        with open(filename) as json_file:
            d = json.load(json_file)
        return d

    def normalize(self, numbers):
        """Normalizing input numbers
        
        Gets the total sum using numpy. Calculates the normalized numbers usin a simple list comprehension
        
        Arguments:
            numbers {list of ints or floats} -- Numbers to normalize
        
        Returns:
            list of ints or floats -- The normalized numbers
        """
        total = np.sum(numbers)
        return [i/total for i in numbers]


    def calc_cdf(self, numbers):
        """Calculates the Cumulative Distribution Function (CDF) for some numbers
        
        Calculates the CDF using Numpy's 'cumsum' function and local 'normalize' function
        
        Arguments:
            numbers {list of ints or floats} -- Numbers for which the CDF should be calculated
        
        Returns:
            Numpy array -- The CDF
        """
        return np.cumsum(self.normalize(numbers))  #Return the cumulative numbers



    def plot_cdf(self, plot_type, numbers, titles, x_labels, y_labels, x_ticks=[[]], y_ticks=[[]]):
        """Plots a list of CDF's in a single figure
        
        
        Arguments:
            plot_type {string} -- String that tells what type of plot there should be used: Bar-plot or XY-plot
            numbers {list of list of CDF values/numbers} -- List of list of values which comes from 'calc_cdf' function
            titles {list of list of strings} -- Title for each subplot
            x_labels {list of list of strings} -- Label for x axis for each subplot
            y_labels {list of list of strings} -- Label for y axis for each subplot
        
        Keyword Arguments:
            x_ticks {list of list of values} -- Specified ticks for x axis instead of default values (default: {[[]]})
            y_ticks {list of list of values} -- Specified ticks for y axis instead of default values (default: {[[]]})
        """
        
        # You typically want your plot to be ~1.33x wider than tall. This plot is a rare    
        # exception because of the number of lines being plotted on it.    
        # Common sizes: (10, 7.5) and (12, 9)    
        plt.figure(figsize=(12, 14))    
          
        # Remove the tick marks; they are unnecessary with the tick lines we just plotted.    
        plt.tick_params(axis="both", which="both", bottom="off", top="off",    
                        labelbottom="on", left="off", right="off", labelleft="on") 

        for i in range(0,len(numbers)):
            if plot_type == "barplot":
                   
                if i==0:
                    ax = plt.subplot(len(numbers)-1, 1, i+1)    
                else:
                    print("i+1+1 = {}".format(i+1+1))
                    ax = plt.subplot(len(numbers)-1,len(numbers)-1,i+1+1)


                # Remove the plot frame lines. They are unnecessary chartjunk.      
                ax.spines["top"].set_visible(False)    
                ax.spines["bottom"].set_visible(False)    
                ax.spines["right"].set_visible(False)    
                ax.spines["left"].set_visible(False)    
                  
                # Ensure that the axis ticks only show up on the bottom and left of the plot.    
                # Ticks on the right and top of the plot are generally unnecessary chartjunk.    
                ax.get_xaxis().tick_bottom()    
                ax.get_yaxis().tick_left() 


                ind = np.arange(len(numbers[i]))  # the x locations for the groups
                width = 0.35       # the width of the bars
                #rects2 = ax.bar(ind + width, womenMeans, width, color='y', yerr=womenStd)
                plt.bar(ind+width, numbers[i], width)
                plt.title(titles[i])
                plt.xlabel(x_labels[i])
                plt.ylabel(y_labels[i])
                if x_ticks != []:
                    plt.xticks(ind + width*1.5, x_ticks[i], rotation='vertical')
                if y_ticks != []:
                    plt.yticks(ind + width*1.5, y_ticks[i], rotation='vertical')
                #plt.show()
            
            elif plot_type == "xy":

                if i==0:
                    ax = plt.subplot(len(numbers)-1, 1, i+1)    
                else:
                    #print("i+1+1 = {}".format(i+1+1))
                    ax = plt.subplot(len(numbers)-1,len(numbers)-1,i+1+1)

                # Remove the plot frame lines. They are unnecessary chartjunk.      
                ax.spines["top"].set_visible(False)    
                ax.spines["bottom"].set_visible(False)    
                ax.spines["right"].set_visible(False)    
                ax.spines["left"].set_visible(False)    
                  
                # Ensure that the axis ticks only show up on the bottom and left of the plot.    
                # Ticks on the right and top of the plot are generally unnecessary chartjunk.    
                ax.get_xaxis().tick_bottom()    
                ax.get_yaxis().tick_left() 

                x = np.arange(len(numbers[i]))
                #print(numbers[i][:-10])
                #print("len(x) = {0}, len(numbers[{1}]) = {2}".format(len(x), i, len(numbers[i])))
                ind = np.arange(len(numbers[i]))  # the x locations for the groups
                width = 0.35       # the width of the bars
                plt.plot(x, numbers[i])
                plt.title(titles[i])
                plt.xlabel(x_labels[i])
                plt.ylabel(y_labels[i])
                #print(y_ticks == [[]])
                #print(x_ticks[0] != [])
                #print(x_ticks)
                if x_ticks[0] != []:
                    #print("xticks her!!")
                    plt.xticks(ind + width*1.5, x_ticks[i], rotation='vertical')
                if y_ticks[0] != []:
                    plt.yticks(ind + width*1.5, y_ticks[i], rotation='vertical')
                

            else:
                print("Unknown plot-type!!!") 
        plt.show()               


    def plot_barplot(self, numbers, title, x_label, y_label, x_ticks=[], y_ticks=[]):
        
        ind = np.arange(len(x_ticks))  # the x locations for the groups
        width = 0.35       # the width of the bars

        # You typically want your plot to be ~1.33x wider than tall. This plot is a rare    
        # exception because of the number of lines being plotted on it.    
        # Common sizes: (10, 7.5) and (12, 9)    
        plt.figure(figsize=(12, 14))    
          
        # Remove the plot frame lines. They are unnecessary chartjunk.    
        ax = plt.subplot(111)    
        ax.spines["top"].set_visible(False)    
        ax.spines["bottom"].set_visible(False)    
        ax.spines["right"].set_visible(False)    
        ax.spines["left"].set_visible(False)    
          
        # Ensure that the axis ticks only show up on the bottom and left of the plot.    
        # Ticks on the right and top of the plot are generally unnecessary chartjunk.    
        ax.get_xaxis().tick_bottom()    
        ax.get_yaxis().tick_left()    


        # Remove the tick marks; they are unnecessary with the tick lines we just plotted.    
        plt.tick_params(axis="both", which="both", bottom="off", top="off",    
                        labelbottom="on", left="off", right="off", labelleft="on") 

        #rects2 = ax.bar(ind + width, womenMeans, width, color='y', yerr=womenStd)
        plt.bar(ind + width, numbers, width)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if x_ticks != []:
            plt.xticks(ind + width*1.5, x_ticks, rotation='vertical')
        if y_ticks != []:
            plt.yticks(ind + width*1.5, y_ticks, rotation='vertical')
        plt.show()          


    def plot_barplot_subplot(self, numbers, title, x_label, y_label, x_ticks=[[]], y_ticks=[[]]): ##each_figure=False,
        """Plots a list of barplots in a single figure
        
        Arguments:
            numbers {list of list of CDF values/numbers} -- List of list of values for the barplots
            titles {list of list of strings} -- Title for each subplot
            x_labels {list of list of strings} -- Label for x axis for each subplot
            y_labels {list of list of strings} -- Label for y axis for each subplot
        
        Keyword Arguments:
            x_ticks {list of list of values} -- Specified ticks for x axis instead of default values (default: {[[]]})
            y_ticks {list of list of values} -- Specified ticks for y axis instead of default values (default: {[[]]})

        """

        # You typically want your plot to be ~1.33x wider than tall. This plot is a rare    
        # exception because of the number of lines being plotted on it.    
        # Common sizes: (10, 7.5) and (12, 9)    
        plt.figure(figsize=(12, 14))    
          
        for i in range(0,len(numbers)):
            #print("i = {}".format(i))
            #print(len(x_ticks[i]))

            ind = np.arange(len(numbers[i]))  # the x locations for the groups
            width = 0.35       # the width of the bars
            # Remove the plot frame lines. They are unnecessary chartjunk.    
            #print("len(numbers)-1 = {}".format(len(numbers)-1))
            if i==0:
                ax = plt.subplot(len(numbers)-1, 1, i+1)    
            else:
                #print("i+1+1 = {}".format(i+1+1))
                ax = plt.subplot(len(numbers)-1,len(numbers)-1,i+1+1)
            ax.spines["top"].set_visible(False)    
            ax.spines["bottom"].set_visible(False)    
            ax.spines["right"].set_visible(False)    
            ax.spines["left"].set_visible(False)    
              
            # Ensure that the axis ticks only show up on the bottom and left of the plot.    
            # Ticks on the right and top of the plot are generally unnecessary chartjunk.    
            ax.get_xaxis().tick_bottom()    
            ax.get_yaxis().tick_left()    


            # Remove the tick marks; they are unnecessary with the tick lines we just plotted.    
            plt.tick_params(axis="both", which="both", bottom="off", top="off",    
                            labelbottom="on", left="off", right="off", labelleft="on") 

            #rects2 = ax.bar(ind + width, womenMeans, width, color='y', yerr=womenStd)
            plt.bar(ind + width, numbers[i], width)
            plt.title(title[i])
            plt.xlabel(x_label[i])
            plt.ylabel(y_label[i])
            if x_ticks != []:
                plt.xticks(ind + width*1.5, x_ticks[i], rotation='vertical')
            if y_ticks != []:
                plt.yticks(ind + width*1.5, y_ticks[i], rotation='vertical')
        plt.show() 


    def get_procentile_of_data(self, data_numbers, labels, procent):
        """Finds those numbers and labels which represent a minimum procent of data
        
        Calculates the CDF for the data and normalize the numbers. Loops over the data, until the given procentage
        is reached, and then returns the appropriated slices of numbers and labels
        
        Arguments:
            data_numbers {list of int or floats} -- Given data/numbers
            labels {list of values} -- The corresponding labels for the data/numbers
            procent {float} -- Procentage (between 0.0 and 1.0)
        
        Returns:
            [list of ints or floats and list of values] -- The sliced list og numbers/values which represent minimum the given procentage
        """
        i = 0;
        data_numbers_cdf = self.calc_cdf(data_numbers)
        data_numbers_norm = self.normalize(data_numbers)
        for n in data_numbers_cdf:
            if n>=procent:
                break
            i+=1
        return data_numbers_norm[:i+1], labels[:i+1]


    
    def fetch_data(self, file_list, key_to_fetch, requirement_keys=[], requirement_values=[]):
        """Fetch data based on a key from (JSON) files, and return the data sorted in different ways
        
        Arguments:
            file_list {list of strings} -- list of filenames to fetch data from
            key_to_fetch {string or list of strings} -- The key the wanted data is under in the JSON files
            requirement_keys {list of strings} -- Keys where other requirements are needed
            requirement_values {list of values} -- Values for requirements to the keys
        
        Keyword Arguments:
            requirement_keys {list of strings} -- Keys where other requirements are needed (default: {[]})
            requirement_values {list of values} -- Values for requirements to the keys (default: {[]})

        Returns:
            lists -- The list of data, sorted in different ways
        """

        key_data = defaultdict(dict)
        if isinstance(key_to_fetch, list):
            key_data[key_to_fetch[-1]] = set()
        else:
            key_data[key_to_fetch] = set()

        list_of_key_data_list_asc = []    #Ascending
        list_of_numbers_asc = []           #Ascending
        list_of_key_data_list_desc = []   #Descending
        list_of_numbers_desc = []          #Descending
        list_of_key_data_list_alph = []   #Alphabetic (countries)
        list_of_numbers_alph = []          #Alphabetic (countries)

        print("Summarization:\n----------")
        for f in file_list:
            print("For {0}:\n-------".format(f))
            all_data = self.open_file(f)

            #print(len(all_data))
            for data in all_data: 
                #print(data)
                if isinstance(key_to_fetch, list):
                    data_value = None
                    temp = None
                    index = 0
                    for key in key_to_fetch:
                        #print("key in key_to_fetch = {0}".format(key))
                        if len(key_to_fetch) == 1: #only 1 key!
                            temp = data[key]
                            index += 1
                            if temp != '':
                                exit_flag = False
                                index2 = 0
                                for k in requirement_keys:
                                    #print("k = {0}".format(k))
                                    if data[k] != requirement_values[index2]:
                                        exit_flag = True
                                        break
                                    index2 += 1
                                if exit_flag:
                                    #print("Krav ikke opfyldt!")
                                    #print(data)
                                    break
                                #print("\n-----------\nKrav ER opfyldt!\n----------")
                                #print(data)
                                # print("temp = {0}".format(temp))
                                # print("key_to_fetch[-1] = {0}".format(key_to_fetch[-1]))
                                # print("(key_to_fetch[-1]+'_numbers') = {0}".format((key_to_fetch[-1]+'_numbers')))
                                #print("key_data[(temp+'_numbers')] = {0}".format(key_data[(temp+'_numbers')]))
                                key_data[key_to_fetch[0]].add(temp)
                                if temp in key_data[(key_to_fetch[-1]+'_numbers')]:
                                    #print("+1")
                                    key_data[(key_to_fetch[0]+'_numbers')][temp] += 1
                                else: 
                                    #print("Ikke +1")
                                    key_data[(key_to_fetch[0]+'_numbers')][temp] = 1
                            index += 1
                        else:
                            if temp is None:
                                temp = data[key]
                                index += 1
                            else:
                                if isinstance(temp, list):
                                    temp = temp[0][key]
                                else:
                                    temp = temp[key]

                                if index==(len(key_to_fetch)-1) and temp != '':
                                    exit_flag = False
                                    index2 = 0
                                    for k in requirement_keys:
                                        #print("k = {0}".format(k))
                                        if data[k] != requirement_values[index2]:
                                            exit_flag = True
                                            break
                                        index2 += 1
                                    if exit_flag:
                                        #print("Krav ikke opfyldt!")
                                        #print(data)
                                        break
                                    #print("\n-----------\nKrav ER opfyldt!\n----------")
                                    #print(data)
                                    # print("temp = {0}".format(temp))
                                    # print("key_to_fetch[-1] = {0}".format(key_to_fetch[-1]))
                                    # print("(key_to_fetch[-1]+'_numbers') = {0}".format((key_to_fetch[-1]+'_numbers')))
                                    #print("key_data[(temp+'_numbers')] = {0}".format(key_data[(temp+'_numbers')]))
                                    key_data[key_to_fetch[-1]].add(temp)
                                    if temp in key_data[(key_to_fetch[-1]+'_numbers')]:
                                        #print("+1")
                                        key_data[(key_to_fetch[-1]+'_numbers')][temp] += 1
                                    else: 
                                        #print("Ikke +1")
                                        key_data[(key_to_fetch[-1]+'_numbers')][temp] = 1
                                index += 1
                else:
                    if data[key_to_fetch] != '': #
                        key_data[key_to_fetch].add(data[key_to_fetch])
                        if data[key_to_fetch] in key_data[(key_to_fetch+'_numbers')]:
                            key_data[(key_to_fetch+'_numbers')][data[key_to_fetch]] += 1
                        else: 
                            key_data[(key_to_fetch+'_numbers')][data[key_to_fetch]] = 1



            print("Number of {0}: {1}".format(key_to_fetch, len(key_data[key_to_fetch[-1]])))

            key_data_list = []
            numbers = []
            if isinstance(key_to_fetch, list):
                for label in key_data[key_to_fetch[-1]]:
                    #print(label)
                    key_data_list.append(label)
                    #print(key_data[(key_to_fetch[-1]+'_numbers')][label])
                    numbers.append(key_data[(key_to_fetch[-1]+'_numbers')][label])
            else:
                for label in key_data[key_to_fetch]:
                    key_data_list.append(label)
                    numbers.append(key_data[(key_to_fetch+'_numbers')][label])

            key_data_list_alph, numbers_alph = zip(*sorted(zip(key_data_list, numbers)))
            numbers_asc, key_data_list_asc = zip(*sorted(zip(numbers, key_data_list)))
            numbers_desc, key_data_list_desc = zip(*sorted(zip(numbers, key_data_list), reverse=True))

            list_of_key_data_list_asc.append(list(key_data_list_asc))    #Ascending
            list_of_numbers_asc.append(list(numbers_asc))                  #Ascending
            list_of_key_data_list_desc.append(list(key_data_list_desc))  #Descending
            list_of_numbers_desc.append(list(numbers_desc))                #Descending
            list_of_key_data_list_alph.append(list(key_data_list_alph))  #Alphabetic (countries)
            list_of_numbers_alph.append(list(numbers_alph))                #Alphabetic (countries)

            total_geotags = np.sum(numbers)
            print("total_geotags = {0}, numbers_desc[0] = {1}, key_data_list_desc[0] = {2}".format(total_geotags, numbers_desc[0],key_data_list_desc[0]))
            print("{0} stands for {1}% of all geotagging".format(key_data_list_desc[0], ((numbers_desc[0]/total_geotags)*100)))
            print("\n")
        print("Længde: {0}".format(len(list_of_numbers_alph)))
        return list_of_key_data_list_alph, list_of_numbers_alph, list_of_key_data_list_asc, list_of_numbers_asc, list_of_key_data_list_desc, list_of_numbers_desc
