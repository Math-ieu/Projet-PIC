import json
import base64
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
import io
import os

# Load model once
MODEL_PATH = os.environ.get('MODEL_PATH', 'model.tflite')
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((256, 256))
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def lambda_handler(event, context):
    try:
        body = json.loads(event['body'])
        image_b64 = body.get('image')
        
        if not image_b64:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No image provided'})
            }
            
        image_bytes = base64.b64decode(image_b64)
        input_data = preprocess_image(image_bytes)
        
        # Inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Post-processing
        # Assuming classification output
        prediction = output_data.tolist()
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'prediction': prediction
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
