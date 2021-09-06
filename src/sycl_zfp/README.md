# TODO

1. Figure out why float and double decompression in 3D fails on host & cpu device
2. Figure out why all compressions in 3D fail on host & cpu device (undefined behaviour unsigned integer to signed ?)
3. Find a way to select the device from the CLI
5. Find a reliable solution to detect if a pointer is usable or not
6. Oddness: sometimes cuda benchmark is faster while SYCL global execution time is smaller than CUDA's
7. Implement variable rate.
