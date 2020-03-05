dockerimage ?= rideos/babynames
dockerfile ?= Dockerfile
srcdir ?= $(shell pwd)
datadir ?= $(shell pwd)

install:
	@docker build -t $(dockerimage) -f Dockerfile .

i: install

update:
	@docker build -t $(dockerimage) -f $(dockerfile) . --pull --no-cache

u: update

run:
	@docker run --ipc=host -it --rm -p 8080:8080 -v $(srcdir):/usr/src/app/ -v $(datadir):/data --entrypoint=/bin/bash $(dockerimage)

r: run

tensorboard:
	@docker run -it --rm -p 6006:6006 -v $(datadir):/data tensorflow/tensorflow:2.0.1-py3 tensorboard --bind_all --logdir /data/runs

t:tensorboard

.PHONY: install i run r update u tensorboard t
