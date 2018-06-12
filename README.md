# Caffemodel_Compress
A c++ project what performing CNN channel pruning

You can perform channel pruning on your trained caffemodel by using this tool, A XML file you should rewrite first, then do a little path change in main.cpp ,build it.

note: We do not support to prune a convolution layer which subsequent to a eltwise layer for now, I know it's a little non-reboost , but I still working on it .
