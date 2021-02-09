## CoNNear: A convolutional neural-network model of human cochlear mechanics and filter tuning for real-time applications.

If you use this code, please cite (bibtex given below [1]):    
Baby, D., Van Den Broucke, A. & Verhulst, S. A convolutional neural-network model of human cochlear mechanics and filter tuning for real-time applications. Nat Mach Intell (2021). https://doi.org/10.1038/s42256-020-00286-8

**The supporting paper can be found at [https://www.nature.com/articles/s42256-020-00286-8] with DOI 10.1038/s42256-020-00286-8 (https://arxiv.org/abs/2004.14832).**

> This work was funded with support from the EU Horizon 2020 programme under grant agreement No 678120 (RobSpear).


This repository contains two notebooks made for running and testing the CoNNear model. The full version `connear_notebook.ipynb` holds both the CoNNear model and the reference transmission line (TL) model (Verhulst et al.), the latter can be used as a validation tool. The `connear_notebook_light.ipynb` only runs the CoNNear model, hence decreases significantly the computational time of the full notebook. Both notebooks consist of different blocks corresponding to different sections of the paper. Each block can be adapted by the reader to run variations on the simulations that were described in the paper. 

Besides the notebooks (both in `.html` and `.ipynb` format) the repository contains the trained model (in the connear folder), a `helper_ops.py` file, this `README.md` document, a license file, a speech fragment (`dutch_sentence.wav`), and the folder that contains the reference TL model (tlmodel). 

## How to test the CoNNear model

1. If it's the first time you run the full notebook, you'll have to compile the cochlea_utils.c file that is used for solving the transmission line (TL) model of the cochlea. This requires some C++ compiler which should be installed beforehand. Go the  tlmodel-folder from the terminal and run:
	```
	gcc -shared -fpic -O3 -ffast-math -o tridiag.so cochlea_utils.c
	```
> If running on google colab: add the following as a code block and run it to compile cochlea_utils.c in the runtime machine.

>	!gcc -shared -fpic -O3 -ffast-math -o tridiag.so cochlea_utils.c

2. Install numpy, scipy, keras and tensorflow in a code environment. We opted for Anaconda v2020.02 and used the following versions: 
	+ Python 3.6
	+ Numpy v1.18.1
	+ Scipy v1.4.1 
	+ Keras v2.3.1
	+ Tensorflow v1.13.2

3. Run the code blocks from the "Import required python packages and functions" section of the notebook. All other code blocks are independent and can be run in an order of your choice. 

4. Run the desired code blocks and/or adapt the various parameters to look at the performance for similar tasks. All of the code blocks do hold extra comments to clarify the steps or to define the choice of a parameter value. 
    
## CoNNear model specifications

The CoNNear model is an 8 layer, tanh, 64 filter length - deep convolutional neural network model,
trained on a training set containing 2310 speech sentences of the TIMIT speech dataset. It predicts the basilar membrane displacements for 201 cochlear channels, resembling a frequency range from 100Hz (channel 0) to 12kHz (channel 200) based on the Greenwood map.
		
During the CoNNear simulations, 256 context samples are added at both sides to account for possible loss of information due to the slicing of full speech fragments. 

The CoNNear model can take a stimulus with a variable sample lengths as an input, however, due to the convolutional character of CoNNear, this sample length has to be a multiple of 16. 

## System test

The system was tested on a MacBook Pro 2015 (macOS Sierra v10.12.6) with 3.1 GHz Intel Corei7, 16 GB RAM, and on a MacBook Air 2017 (macOS Catalina v10.15.3) with 1.8 GHz Dual-Core Intel Core i5, 8 GB RAM. 

## CoNNear model specifications

The installation time for the Anaconda environment and dependencies is approximately 20 min. The run-time of the first code-blocks in the connear_notebooks is very fast, only the DPOAE simulation code-block is slow due to the specific nature of how to present the stimuli and analyse the results for the DPOAE analysis. This code-block will automatically calculate the ETA (can be up to 25 mins).

----
For questions, please reach out to one of the corresponding authors

* Deepak Baby: deepakbabycet@gmail.com
* Arthur Van Den Broucke: arthur.vandenbroucke@ugent.be
* Sarah Verhulst: s.verhulst@ugent.be

----
[1] Bibtex
```
@Article{Baby2021,
author={Baby, Deepak
and Van Den Broucke, Arthur
and Verhulst, Sarah},
title={A convolutional neural-network model of human cochlear mechanics and filter tuning for real-time applications},
journal={Nature Machine Intelligence},
year={2021},
month={Feb},
day={08},
abstract={Auditory models are commonly used as feature extractors for automatic speech-recognition systems or as front-ends for robotics, machine-hearing and hearing-aid applications. Although auditory models can capture the biophysical and nonlinear properties of human hearing in great detail, these biophysical models are computationally expensive and cannot be used in real-time applications. We present a hybrid approach where convolutional neural networks are combined with computational neuroscience to yield a real-time end-to-end model for human cochlear mechanics, including level-dependent filter tuning (CoNNear). The CoNNear model was trained on acoustic speech material and its performance and applicability were evaluated using (unseen) sound stimuli commonly employed in cochlear mechanics research. The CoNNear model accurately simulates human cochlear frequency selectivity and its dependence on sound intensity, an essential quality for robust speech intelligibility at negative speech-to-background-noise ratios. The CoNNear architecture is based on parallel and differentiable computations and has the power to achieve real-time human performance. These unique CoNNear features will enable the next generation of human-like machine-hearing applications.},
issn={2522-5839},
doi={10.1038/s42256-020-00286-8},
url={https://doi.org/10.1038/s42256-020-00286-8}
}
```
