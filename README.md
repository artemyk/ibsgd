# Code for On the Information Bottleneck Theory of Deep Learning

* `SaveActivations.ipynb` is a jupyter notebook that trains on MNIST and  saves (in a data directory) activations when run on test set inputs (as well as weight norms, &c.) for each epoch.

* `ComputeMI.ipynb` is a jupyter notebook that loads the data files, computes MI values, and does the infoplane plots and SNR plots.

* `demo.py` is a simple script showing how to compute MI between X and Y, where Y = f(X) + Noise.

Andrew Michael Saxe, Yamini Bansal, Joel Dapello, Madhu Advani, Artemy Kolchinsky, Brendan Daniel Tracey, David Daniel Cox, On the Information Bottleneck Theory of Deep Learning, *ICLR 2018*.
