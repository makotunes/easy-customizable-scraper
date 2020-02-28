##FROM centos
FROM ubuntu
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip
RUN python --version
#FROM python:3.7.4
#FROM continuumio/anaconda3
USER root

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

#Ubuntu
#RUN apt-get upgrade
 #apt-get install -y expect wget bzip2 unzip vim git gcc make git psmisc net-tools build-essential \
ENV DEBIAN_FRONTEND=noninteractive

RUN set -x \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends wget git gcc make build-essential mecab mecab-ipadic-utf8 libmecab-dev swig xz-utils libcurl4 curl mongodb sudo \
    && rm -rf /var/lib/apt/lists/*

RUN cat /etc/issue
#RUN set -x \
#    #&& apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv E52529D4 \
#    #&& bash -c 'echo "deb [arch=amd64] http://repo.mongodb.org/apt/ubuntu bionic/mongodb-org/4.0 multiverse" > /etc/apt/sources.list.d/mongodb-org-4.0.list' \
#    #&& apt-get update -y \
#    #&& apt-get install -y mongodb-org
#    && wget -qO - https://www.mongodb.org/static/pgp/server-4.2.asc | sudo apt-key add - \
#    && echo "deb http://repo.mongodb.org/apt/debian stretch/mongodb-org/4.2 main" | sudo tee /etc/apt/sources.list.d/mongodb-org-4.2.list\
#    && apt-get update -y \
#    && apt-get install -y mongodb-org

#RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
#    wget --quiet https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh -O ~/anaconda.sh && \
#    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
#    rm ~/anaconda.sh
#ENV PATH="/opt/conda/bin:$PATH"

COPY requirements.txt /usr/src/app/

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

#RUN python -m nltk.downloader -d /root/nltk_data all 
RUN python -m nltk.downloader -d /root/nltk_data book

RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git


RUN ./mecab-ipadic-neologd/bin/install-mecab-ipadic-neologd -n -a -y && rm -rf ./mecab-ipadic-neologd

RUN mkdir -p /data/db && chown -R mongodb:mongodb /data/db

ENV LANG="en_US.UTF-8"
ENV LANGUAGE="en_US.UTF-8"

RUN pip install pandas

COPY ./src /usr/src/app/src
COPY ./custom /usr/src/app/custom
RUN mkdir ./workspace
COPY entrypoint.sh /usr/src/app/

ENV PYTHONPATH="/usr/src/app/"
ENTRYPOINT /usr/src/app/entrypoint.sh