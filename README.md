# Caffemodel_Compress# <<Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding>>

Song Han, Huizi Mao, William J. Dally
(Submitted on 1 Oct 2015 (v1), last revised 15 Feb 2016 (this version, v5))

This C++ project implements CNN channel pruning, an idea I've been considering for a while. You can prune channels in your trained Caffe model using this tool. First, ensure you have the XML file relocated, regardless of whether you’re working with depthwise or convolutional layers. When setting up the layers for pruning, the corresponding channel pruning arrangement is implicitly configured. You’ll also need to make a minor change in main.cpp and build the project with your local IDE.

Updated 18/6/25: Eltwise pruning on Mac is now supported.
