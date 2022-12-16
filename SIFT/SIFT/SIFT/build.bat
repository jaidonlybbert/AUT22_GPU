nvcc -arch=sm_86 -rdc=true -lineinfo -o sift sift_main.cu sift_device.cu sift_host.cpp -lcudadevrt
