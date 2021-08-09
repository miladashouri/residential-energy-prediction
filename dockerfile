FROM python:3.9.6
EXPOSE 8080
LABEL Milad Ashouri "ashouri dot milad68 at gmail.com"
ADD . /python-flask
WORKDIR /python-flask
RUN pip install -r requirements.txt
