FROM golang:1.12.7-stretch

WORKDIR /mix_net

COPY . /mix_net

CMD ["/mix_net/start.sh"]
