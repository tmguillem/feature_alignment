FROM python:2.7
MAINTAINER Guillem Torrente <guillemtorrente@hotmail.com>

RUN apt-get update
RUN pip install -r requirements.txt

