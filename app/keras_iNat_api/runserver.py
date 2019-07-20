# /ai4e_api_tools has been added to the PYTHONPATH, so we can reference those
# libraries directly.
from flask import Flask, request
from ai4e_app_insights_wrapper import AI4EAppInsights
from ai4e_service import APIService
from sas_blob import SasBlob
from PIL import Image
import keras_detector
from io import BytesIO
from os import getenv
import uuid
import sys
import numpy as np
import traceback

print("Creating Application")

ACCEPTED_CONTENT_TYPES = ['image/png', 'application/octet-stream', 'image/jpeg', 'image/tiff']
blob_access_duration_hrs = 1

app = Flask(__name__)

# Use the AI4EAppInsights library to send log messages.
log = AI4EAppInsights()

# Use the APIService to executes your functions within a logging trace, supports long-running/async functions,
# handles SIGTERM signals from AKS, etc., and handles concurrent requests.
with app.app_context():
    ai4e_service = APIService(app, log)

# Define a function for processing request data, if appliciable.  This function loads data or files into
# a dictionary for access in your API function.  We pass this function as a parameter to your API setup.
def process_request_data(request):
    return_values = {'image_bytes': None}
    try:
        # Attempt to load the body
        return_values['image_bytes'] = BytesIO(request.data)
    except:
        log.log_error('Unable to load the request data')   # Log to Application Insights
    return return_values

# POST, async API endpoint example
@ai4e_service.api_async_func(
    api_path = '/detect', 
    methods = ['POST'], 
    request_processing_function = process_request_data, # This is the data process function that you created above.
    maximum_concurrent_requests = 5, # If the number of requests exceed this limit, a 503 is returned to the caller.
    content_types = ACCEPTED_CONTENT_TYPES,
    content_max_length = 10000000000, # In bytes
    trace_name = 'post:detect')
def detect(*args, **kwargs):
    print('runserver.py: detect() called, generating detections...')
    image_bytes = kwargs.get('image_bytes')
    taskId = kwargs.get('taskId')

    # Update the task status, so the caller knows it has been accepted and is running.
    ai4e_service.api_task_manager.UpdateTaskStatus(taskId, 'running - generate_detections')

    try:
        arr_for_detection, image_for_drawing = keras_detector.open_image(image_bytes)
        ai4e_service.api_task_manager.UpdateTaskStatus(taskId, 'running - image opened, generating detections')
        #TO DO visualize masks
        rois, clsses, scores, masks = keras_detector.generate_detections(arr_for_detection)

        ai4e_service.api_task_manager.UpdateTaskStatus(taskId, 'rendering boxes')

        # image is modified in place
        # here confidence_threshold is hardcoded, but you can ask that as a input from the request
        keras_detector.render_bounding_boxes(
            rois, scores, clsses, image_for_drawing, confidence_threshold=0.5)

        print('runserver.py: detect(), rendering and saving result image...')
        # save the PIL Image object to a ByteIO stream so that it can be written to blob storage
        output_img_stream = BytesIO()
        image.save(output_img_stream, format='jpeg')
        output_img_stream.seek(0)

        sas_blob_helper = SasBlob()
        # Create a unique name for a blob container
        container_name = str(uuid.uuid4()).replace('-','')

        # Create a writable sas container and return its url
        sas_url = sas_blob_helper.create_writable_container_sas(
            getenv('STORAGE_ACCOUNT_NAME'), getenv('STORAGE_ACCOUNT_KEY'), container_name, blob_access_duration_hrs)

        # Write the image to the blob
        sas_blob_helper.write_blob(sas_url, 'detect_output.jpg', output_img_stream)
        
        ai4e_service.api_task_manager.CompleteTask(taskId, 'completed - output written to: ' + sas_url)
        print('runserver.py: detect() finished.')
    except:
        log.log_exception(sys.exc_info()[0], taskId)
        ai4e_service.api_task_manager.FailTask(taskId, 'failed: ' + str(sys.exc_info()[0])+'\n'+str(traceback.format_exc()))

if __name__ == '__main__':
    app.run()
