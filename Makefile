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

.PHONY: install i run r update u
