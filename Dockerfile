# Define Base Image
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel
# Install dependencies

ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get update && apt-get install -y git cmake build-essential ninja-build libboost-program-options-dev libboost-graph-dev libboost-system-dev libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev libgoogle-glog-dev libgtest-dev libgmock-dev libsqlite3-dev libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev curl ffmpeg pkg-config python3 python3-dev rsync software-properties-common unzip libopencv-dev imagemagick

#RUN cd /opt
WORKDIR /opt
RUN git clone https://github.com/colmap/colmap.git
#RUN cd colmap
WORKDIR /opt/colmap
RUN git checkout tags/3.11.0
RUN mkdir build
#RUN cd build
WORKDIR /opt/colmap/build

RUN cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=89
RUN ninja install
RUN python -m pip install --upgrade pip

RUN pip install opencv-python pyyaml plyfile tqdm termcolor kornia imgaug lpips tensorboardX ipdb scikit-image imageio imageio[ffmpeg] imageio[pyav] mmcv==1.6.0 argparse pytorch_msssim open3d torch_efficient_distloss einops
COPY ./dependencies /opt/dependencies
WORKDIR /opt/dependencies/segment-anything
RUN pip install -e .
WORKDIR /opt/dependencies/GroundingDINO
RUN pip install -e .

RUN pip install yapf==0.40.1
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
RUN pip install ultralytics
RUN pip install numpy==1.26.4

ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"

ENV TORCH_EXTENSIONS_DIR=/opt/torch_extensions_cache
COPY ./lib /opt/lib
WORKDIR /opt
RUN python lib/build_module.py

RUN cp /opt/torch_extensions_cache/adam_upd_cuda/adam_upd_cuda.so /opt/conda/lib/python3.10/site-packages
RUN cp /opt/torch_extensions_cache/render_utils_cuda/render_utils_cuda.so /opt/conda/lib/python3.10/site-packages
RUN cp /opt/torch_extensions_cache/total_variation_cuda/total_variation_cuda.so /opt/conda/lib/python3.10/site-packages
RUN cp /opt/torch_extensions_cache/ub360_utils_cuda/ub360_utils_cuda.so /opt/conda/lib/python3.10/site-packages

RUN echo 'ln /dev/null /dev/raw1394' >> ~/.bashrc
