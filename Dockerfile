FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
WORKDIR /app
COPY . ./
ENV DEBIAN_FRONTEND=noninteractive 
#RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-get update && apt install -y ffmpeg libsm6 libxext6 cmake git ninja-build build-essential python3-pip python-dev gcc libboost-all-dev
RUN pip install -r requirements.txt && pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html
ENV CUDA_HOME="/usr/local/cuda-11.3"
ENV TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
RUN git clone https://github.com/open-mmlab/mmcv.git && cd mmcv && git reset --hard 91a7fee && MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install -e .
#RUN git clone https://github.com/traveller59/spconv && cd spconv && git reset --hard f22dd9 && git submodule update --recursive && python3 setup.py bdist_wheel && pip install ./dist/spconv-1.2.1-cp37-cp37m-linux_x86_64.whl && cd ..
RUN if [ -d "/app/mmdetection_kitti" ]; then rm -rf /app/mmdetection_kitti ; fi && git clone https://github.com/xy-guo/mmdetection_kitti && cd /app/mmdetection_kitti && pip install -v . && cd ..
RUN if [ -d "/app/spconv" ]; then rm -rf /app/spconv ; fi && git clone https://github.com/traveller59/spconv && cd spconv && git reset --hard f22dd9 && git submodule update --recursive && cd third_party && git clone https://github.com/pybind/pybind11.git && cd pybind11 && git reset --hard 085a294  && cd ../.. &&  python3 setup.py bdist_wheel && export whl=$(ls dist/) && pip install ./dist/$whl && cd .. 
RUN  export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX" && python3 setup.py develop
EXPOSE 5000
ENTRYPOINT python3 tools/infer.py --cfg_file ./configs/stereo/kitti_models/liga.3d-and-bev.yaml --ckpt released.final.liga.3d-and-bev.ep53.pth