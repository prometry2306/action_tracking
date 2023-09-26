# Ubuntu22.04 + CUDA11.8 + mmpose のGPUコンテナ
FROM public.ecr.aws/d4g9d0l2/ubuntu22-cuda11.8-mmpose
LABEL maintainer="mshinoda"
ENV PROJECT_ROOTDIR /opt/app

# install application materials
USER root
WORKDIR ${PROJECT_ROOTDIR}
COPY requirements.txt ${PROJECT_ROOTDIR}/requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install --trusted-host pypi.python.org --no-cache-dir -r ${PROJECT_ROOTDIR}/requirements.txt

WORKDIR ${PROJECT_ROOTDIR}/src
EXPOSE 8501

