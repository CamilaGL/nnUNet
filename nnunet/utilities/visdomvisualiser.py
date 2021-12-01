import shutil
import numpy as np
import matplotlib.pyplot as plt

from os import path, makedirs
from abc import ABC, abstractmethod

from visdom import Visdom

def get_plotter(config):
    '''
    Setup the right plotter
    '''

    # setup the visualizer according to the backend library
    # if config['visualization']['library'] == 'tensorboard':
    #     if config['experiment']['type'] == 'image-classification':
    #         plotter = ImageClassificationTensorboardPlotter(config)
    # elif config['visualization']['library'] == 'visdom':
    #     if config['experiment']['type'] == 'image-classification':
    #         plotter = ImageClassificationVisdomPlotter(config)
    #     elif config['experiment']['type'] == 'image-segmentation-2d':
    plotter = GenericVisdomPlotter(config)

    return plotter




class GenericPlotter(ABC):
    '''
    Generic abstract class to plot training statistics
    '''

    def __init__(self, config):
        '''
        Generic initializer
        '''
        super(GenericPlotter, self).__init__()


    @abstractmethod
    def plot_multiple_statistics(self, plot_name, x, y_values):
        '''
        Plot line plots in the same plot
        '''
        pass

    @abstractmethod
    def plot_scalar(self, plot_name, x, y, legend):
        '''
        Plot a line plot
        '''
        pass

    @abstractmethod
    def display_image(self, image_key, image, caption=''):
        '''
        Display given images in the plot
        '''
        pass



class GenericVisdomPlotter(GenericPlotter):
    '''
    Visdom based generic plotter implementation
    '''

    def __init__(self, config):
        '''
        Initializer
        '''
        super(GenericVisdomPlotter, self).__init__(config)

        # prepare the environment name
        self.env = 'Cami_' + config
        # default host and port
        hostname = 'deep' # "deep"
        baseurl = "/visdom"
        myport = 80
        # replace host and port by the one provided in the config file
        # if 'hostname' in config['visualization']:
        #     hostname = config['visualization']['hostname']
        # if 'port' in config['visualization']:
        #     port = int(config['visualization']['port'])

        # initialize the object for visualization
        self.viz = Visdom(server=hostname,base_url=baseurl, port=myport, use_incoming_socket=False, use_polling=True)
        # the dictionary of plots and figures
        self.figures = dict()
        self.plots = dict()
        
        # initialize the current epoch in 0
        self.current_epoch = 0
        

    def plot(self, plot_name, split_name, x, y, x_label='Epochs'):
        '''
        Plot a line plot
        '''

        # if the plot is not in the dictionary, initialize one
        if (plot_name not in self.plots):
            self.plots[plot_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, 
                                                  opts=dict(legend=[str(split_name)],
                                                            title=plot_name,
                                                            xlabel=x_label,
                                                            ylabel=plot_name))
        # if the plot is already there, update
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, 
                          update='append', win=self.plots[plot_name], name=str(split_name))

    def plot_matplotlib(self, plt, plot_name="matplotlib_progreso"):
        self.viz.matplot(plt, env=self.env, win=plot_name)

    def plot_multiple_statistics(self, plot_name, x, y_values):
        '''
        Plot multiple statistics within the same plot
        '''
        # get the split names
        split_names = y_values.keys()
        # iterate for each of them
        for split in split_names:
            # plot the values
            self.plot(plot_name, split, x, y_values[split])

    def plot_text(self, info, plot_name="some_text"):
        '''
        Plot some text
        '''
        self.viz.text(info, env=self.env, win=plot_name)


    def plot_scalar(self, plot_name, x, y, legend):
        '''
        Plot a line plot
        '''
        self.plot(plot_name, legend, x, y)


    def display_image(self, image_key, image, caption=''):
        '''
        Display given image in the plot
        '''
        # if the image is already in the plot, remove it to replace it for the new one
        if image_key in self.figures:
            self.viz.close(win=self.figures[image_key], env=self.env)
            del self.figures[image_key]
        # plot the image
        self.figures[image_key] = self.viz.images(image, env=self.env, opts=dict(title=caption))

    def close_client(self):

        self.viz.use_socket = False



