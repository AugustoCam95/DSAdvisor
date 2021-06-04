FROM ubuntu:20.04

# Instalar Python
RUN set -xe \
    && apt-get update \
    && apt-get install python3-pip -y \
    && apt-get install curl -y

# Copiar Requirements
WORKDIR /DSAdvisor
COPY ./requirements.txt /DSAdvisor/
COPY ./DSAdvisor /DSAdvisor/

RUN pip3 install -r ./requirements.txt

# Configuracao
EXPOSE 5000
CMD python3 app.py
