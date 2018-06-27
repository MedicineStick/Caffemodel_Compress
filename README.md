# Caffemodel_Compress
A c++ project what performing CNN channel pruning
It's an idea I've had floating around for a while , You can perform channel pruning on your trained caffemodel by using this tool, A XML file you should recollocate first,irrespective of depthwise convolution layer, when convolution layer which need to prune has been setting, a reciprocal arrangement of channel pruning also has been setting during processing of progress implicitly, then do a little path change in main.cpp ,and build it with your local IDE.


updated 18/6/25 Already supported to eltwise pruning
