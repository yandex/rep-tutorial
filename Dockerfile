FROM yandex/rep:0.6.5
MAINTAINER Andrey Ustyuzhanin <anaderi@yandex-team.ru>

RUN /bin/bash --login -c " \
  pip install hep_ml \
"

COPY . /notebooks/
