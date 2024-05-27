# Build main image using dockerfile
docker build -f Dockerfile --network=host -t $1 .

# Install tiny-cuda-nn into the image file using runtime. 
# This is a workaround to avoid having to reconfigure docker to build images using nvidia runtime: https://github.com/NVIDIA/nvidia-docker/wiki/Advanced-topics#default-runtime.
# Changing docker settings require root privileges.
rm ./tmp.cid
nvidia-docker run --cidfile tmp.cid -it $1 bash -c 'pip install --global-option="--no-networks" git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch && mkdir -p /root/.imageio/ && wget https://cloud.tsinghua.edu.cn/f/34b0e26916574bd9b3d1/\?dl\=1 -O /root/.imageio/libfreeimage-3.16.0-linux64.so'
nvidia-docker commit $(< tmp.cid) $1
rm ./tmp.cid