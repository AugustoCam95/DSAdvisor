FROM ubuntu:20.04

# Instalar Python
RUN set -xe \
    && apt-get update \
    && apt-get install python3-pip -y \
    && apt-get install curl -y \
    && mkdir /DSAdvisor

# Copiar Requirements
WORKDIR /DSAdvisor
COPY . .

RUN pip3 install -r ./requirements.txt

# Configuracao
EXPOSE 5000
CMD python3 application.py