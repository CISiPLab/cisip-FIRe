FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 libgl1-mesa-glx -y

RUN pip install jupyterlab
RUN pip install opencv-python
RUN pip install scikit-image
RUN pip install scikit-learn
RUN pip install seaborn
RUN pip install pandas
RUN pip install imutils
RUN pip install craft-text-detector
RUN pip install tensorflow
RUN pip install kornia
RUN pip install pytorch_memlab

RUN apt-get install bc unzip git -y

# if use accimage
RUN pip install --prefix=/opt/intel/ipp ipp-devel
RUN pip install git+https://github.com/pytorch/accimage
ENV LD_LIBRARY_PATH=/opt/intel/ipp/lib:$LD_LIBRARY_PATH

# if use pillow-simd
RUN pip uninstall -y pillow
RUN CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

RUN pip install umap-learn

RUN pip install faiss-gpu

# setup user [ to avoid -u $(id -u):$(id -g) ]
RUN groupadd -g 1000 user
RUN useradd -g 1000 -u 1000 -ms /bin/bash user
EXPOSE 6006
EXPOSE 8123

USER user
WORKDIR /workspace
