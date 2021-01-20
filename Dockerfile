FROM continuumio/miniconda
WORKDIR /app
ARG src="D:/Github/VRFrameGAN/"
ADD ${src} ./code
RUN conda env create -f env37_jan19.yml && echo "source activate NNenv37vr" > ~/.bashrc && cd code/
ENV PATH /opt/conda/envs/env/bin:$PATH
