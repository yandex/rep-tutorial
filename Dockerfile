FROM yandex/rep
MAINTAINER Andrey Ustyuzhanin <anaderi@yandex-team.ru>

RUN /bin/bash --login -c " \
  pip install hep_ml \
"
