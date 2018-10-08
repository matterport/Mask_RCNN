FROM tensorflow/tensorflow:latest-gpu-py3

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxtst6 \
    libfontconfig1 \
    libxrender1 \
    libsm6 && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/env && \
    echo '#!/bin/bash\npip install -e . \nexec "$@"' > /app/env/entrypoint.bash && \
    chmod +x /app/env/entrypoint.bash

ADD docker/requirements.txt /app/env/requirements.txt

RUN pip install cython && pip install -r /app/env/requirements.txt

WORKDIR /app/src

CMD [ "bash" ]
ENTRYPOINT [ "bash", "/app/env/entrypoint.bash" ]
