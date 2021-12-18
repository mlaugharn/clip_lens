FROM postgres:latest
RUN apt update -y && apt upgrade -y
RUN apt install git build-essential postgresql-server-dev-13 -y
RUN git clone --branch v0.1.7 https://github.com/ankane/pgvector.git && cd pgvector && make && make install
