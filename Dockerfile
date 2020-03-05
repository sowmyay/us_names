#Ubuntu is the base docker image that will be pulled from  bockerhub. It is equivalent to having an ubuntu machine
FROM ubuntu:18.04

WORKDIR /usr/src/app

ENV LANG="C.UTF-8" LC_ALL="C.UTF-8" PATH="/opt/venv/bin:$PATH" PIP_NO_CACHE_DIR="false"

# Run executes commands
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python3 -m venv /opt/venv && \
    python3 -m pip install pip==19.3.1 pip-tools==4.2.0

RUN python3 -m piptools sync

RUN jupyter nbextension enable --py widgetsnbextension

# COPY or ADD commands are use to copy files from host to container directories
COPY . .

EXPOSE 8080

EXPOSE 6006

# When docker container is built, this is the first command that will be executed
ENTRYPOINT ["/usr/src/app/bin/names"]
#Cmd command can't be used to build the docker image
CMD ["-h"]