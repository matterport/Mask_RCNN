# Pull in the AI for Earth Base Image, so we can extract necessary libraries.
FROM mcr.microsoft.com/aiforearth/base-py:latest as ai4e_base

# Use any compatible Ubuntu-based image as your selected base image.
FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
# Copy the AI4E tools and libraries to our container.
COPY --from=ai4e_base /ai4e_api_tools /ai4e_api_tools

# Add the AI4E API source directory to the PATH.
ENV PATH /usr/local/envs/ai4e_py_api/bin:$PATH
# Add the AI4E tools directory to the PYTHONPATH.
ENV PYTHONPATH="${PYTHONPATH}:/ai4e_api_tools"

# Install Miniconda, Flask, Supervisor, uwsgi
RUN ./ai4e_api_tools/requirements/install-api-hosting-reqs.sh

# Install Azure Blob SDK
RUN ./ai4e_api_tools/requirements/install-azure-blob.sh

# Install Application Insights
RUN ./ai4e_api_tools/requirements/install-appinsights.sh

# Install Opencensus
RUN ./ai4e_api_tools/requirements/install-opencensus.sh
EXPOSE 55678 50001

# Now add our custom image commands ---------------------------------------------------
# installing environement, replaed 3 lines in ordiginal repo
RUN apt-get update -y && apt-get install -y git-core

RUN echo "source activate ai4e_py_api" >> ~/.bashrc \
    && git clone https://github.com/ecohydro/CropMask_RCNN ~/CropMask_RCNN

WORKDIR ~/CropMask_RCNN
    
RUN conda env update -f ~/CropMask_RCNN/app/app-env.yml -n ai4e_py_api \
    && . activate ai4e_py_api \
    && python ~/CropMask_RCNN/setup.py install

# Note: supervisor.conf reflects the location and name of your api code.
# If the default (./my_api/runserver.py) is renamed, you must change supervisor.conf
COPY ./supervisord.conf /etc/supervisord.conf
# startup.sh is a helper script
COPY ./startup.sh /
RUN chmod +x /startup.sh

COPY ./LocalForwarder.config /lf/

# Copy your API code
COPY ./keras_iNat_api /app/keras_iNat_api/

# Application Insights keys and trace configuration
ENV APPINSIGHTS_INSTRUMENTATIONKEY= \
    APPINSIGHTS_LIVEMETRICSSTREAMAUTHENTICATIONAPIKEY= \
    LOCALAPPDATA=/app_insights_data \
    OCAGENT_TRACE_EXPORTER_ENDPOINT=localhost:55678

# The following variables will allow you to filter logs in AppInsights
ENV SERVICE_OWNER=AI4E_PyTorch_Example \
    SERVICE_CLUSTER=Local\ Docker \
    SERVICE_MODEL_NAME=AI4E_PyTorch_Example \
    SERVICE_MODEL_FRAMEWORK=Python \
    SERVICE_MODEL_FRAMEOWRK_VERSION=3.6.6 \
    SERVICE_MODEL_VERSION=1.0

ENV API_PREFIX=/v1/keras_iNat_api

ENV STORAGE_ACCOUNT_NAME= \
    STORAGE_ACCOUNT_KEY=

# Expose the port that is to be used when calling your API
EXPOSE 80
HEALTHCHECK --interval=1m --timeout=3s --start-period=20s \
  CMD curl -f http://localhost/${API_PREFIX}/  || exit 1
ENTRYPOINT [ "/startup.sh" ]
# Use the following entrypoint to debug bridge issues.
#ENTRYPOINT ["uwsgi", "--chdir", "/app/keras_iNat_api/", "--virtualenv", "/usr/local/envs/ai4e_py_api", "--http", "0.0.0.0:80", "--callable", "app", "--enable-threads", "--wsgi-file", "/app/tf_iNat_api/runserver.py", "--wsgi-disable-file-wrapper"]

       
