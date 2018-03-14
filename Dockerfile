from jupyter/tensorflow-notebook

USER $NB_UID

RUN git clone https://github.com/cocodataset/cocoapi.git /home/jovyan/cocoapi
RUN cd /home/jovyan/cocoapi/PythonAPI && make install

