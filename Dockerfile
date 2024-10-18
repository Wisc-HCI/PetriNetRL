FROM python:3.11.4

WORKDIR /scratch

COPY requirements.txt ./

RUN pip3 install -r requirements.txt

RUN rm ./requirements.txt