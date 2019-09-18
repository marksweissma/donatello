FROM python:2.7

RUN mkdir /opt/workspace

WORKDIR /opt/workspace

COPY .  /opt/workspace

RUN cd /opt/workspace && pip install -r env_requirements.txt && python setup.py install
