FROM nvidia/cuda:11.0.3-base-ubuntu20.04 as fhmap_runtime
ARG INSTALL_DIRECTORY=/home/fhmap_user
ARG PYTHON_VERSION=3.9
# Restrict log 
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y tzdata && \
	apt-get install --no-install-recommends -y \
	git curl ssh openssh-client \
	python${PYTHON_VERSION} python3-pip \
	&& pip3 install poetry

# Add user. Without this, following process is executed as admin (This will lead file permission problem.). 
RUN useradd -ms /bin/sh fhmap_user
USER fhmap_user

RUN mkdir -p ${INSTALL_DIRECTORY} \
	&& git clone https://github.com/ultimatemagic79/FourierHeatmap.git ${INSTALL_DIRECTORY}/FourierHeatmap
WORKDIR ${INSTALL_DIRECTORY}/FourierHeatmap
RUN poetry install

ENTRYPOINT ["poetry", "run", "python", "fhmap/apps/eval_fhmap.py"]
CMD ["--help"]
