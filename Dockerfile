FROM python:2.7

RUN mkdir /opt/workspace

WORKDIR /opt/workspace

COPY .  /opt/workspace

RUN pip install -r env_requirements.txt

RUN python setup.py install
