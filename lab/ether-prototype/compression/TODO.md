Thoughts after generating this folder:

You'd want the transformation's to be provided by the library? Or just have an extensive list of things to do; basically this is the program which will be generated, so needs to be very dynamic. Later that would have to translate into equivalent C++ (or something else) code, that code would have to be compiled as well to calculate cost of certain transformations.

Probably the different compression algorithms aren't that useful for estimating entropy; since we're after a new algorithm, or you'd have to very neatly order things in such a way that the compression algorithms can properly take advantage of their strengths - unlikely though. --> You'd probably want them as starting points to start traversing from though? (But some of the same design principles would apply) -> Basically they're specific points in the equivalence graph we're traversing -> So we want to compile them to something we can interpret in this search for algorithms.

This is at least a setup for how to include WebGL/C++ code if we need it. And the idea of accelerating with the GPU is essential for the project to succeed - we need quick feedback on the compression algorithms.

So what we want, a compiler which provides backends .wgsl & .cpp (for equivalent on 1 cpu).
