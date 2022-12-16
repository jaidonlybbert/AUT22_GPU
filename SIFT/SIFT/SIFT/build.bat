nvcc -arch=sm_86 -rdc=true -lineinfo -o sift_dev sift_device.cu sift_host.cpp -lcudadevrt
