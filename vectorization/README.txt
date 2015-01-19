This is my implementation of autoencoder from Deep Learning tutorial. I wrote code in computeNumericalGradient.m, sparseAutoEncoder.m and sampleIMAGES.m. The dataset and the rest of the code are provided by Deep Learning Tutorial website at http://deeplearning.stanford.edu/wiki/index.php/UFLDL_Tutorial. 

Summary (from the Deep Learning Tutorial website) The algorithm attempts to learn a set of edge detectors from a pool of random sample patches from a provided list of images. 

The algorithm tells the neural network to learn the identity function (h_(W, b) (x) is similar to x). It constrains the number of hidden units to be smaller than output unit, so the network needs to learn a compressed version of representation and can potentially discover correlation among inputs. In this tutorial, using a random set of images helps to discover set of edge detectors in weights and bias representation.

Downloading all the files and running train.m in Matlab may give you something like this attached image.
