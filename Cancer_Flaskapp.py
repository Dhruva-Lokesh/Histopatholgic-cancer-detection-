from flask import Flask, request, render_template, url_for
import boto3
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# S3 Configuration
s3_client = boto3.client('s3')
UPLOAD_BUCKET_NAME = 'userinputimages'
RESULT_BUCKET_NAME = 'predictionresults'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Upload image to S3 and trigger prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        # Upload file to S3
        s3_client.upload_fileobj(file, UPLOAD_BUCKET_NAME, filename)
        
        # Send the image S3 path to EMR for prediction
        image_s3_path = f"s3://{UPLOAD_BUCKET_NAME}/{filename}"
        prediction_result = send_to_emr(image_s3_path)
        
        if prediction_result:
            # Generate S3 URL for the uploaded image
            image_url = f"https://{UPLOAD_BUCKET_NAME}.s3.amazonaws.com/{filename}"
            
            # Render the results page
            return render_template(
                'result.html', 
                prediction=prediction_result, 
                image_url=image_url
            )
        else:
            return "Error during prediction", 500
    return "Invalid file format", 400

# Sends image S3 path to EMR for prediction
def send_to_emr(image_s3_path):
    emr_client = boto3.client('emr')
    cluster_id = "j-mbfm01utuo"  

    # EMR Step to run prediction
    step_response = emr_client.add_job_flow_steps(
        JobFlowId=cluster_id,
        Steps=[
            {
                'Name': 'Run Prediction',
                'ActionOnFailure': 'TERMINATE_CLUSTER',
                'HadoopJarStep': {
                    'Jar': 'command-runner.jar',
                    'Args': [
                        'spark-submit',
                        '--deploy-mode', 'cluster',
                        's3://<your-script-bucket>/predict_script.py',
                        image_s3_path,
                        RESULT_BUCKET_NAME
                    ]
                }
            }
        ]
    )

    step_id = step_response['StepIds'][0]
    waiter = emr_client.get_waiter('step_complete')
    waiter.wait(ClusterId=cluster_id, StepId=step_id)

    # Fetch the result from the result bucket
    prediction_result = fetch_prediction_result(image_s3_path.split('/')[-1])
    return prediction_result

def fetch_prediction_result(filename):
    result_key = f"{filename.split('.')[0]}_result.txt"
    try:
        obj = s3_client.get_object(Bucket=RESULT_BUCKET_NAME, Key=result_key)
        prediction = obj['Body'].read().decode('utf-8')
        return prediction
    except Exception as e:
        print(f"Error fetching prediction result: {e}")
        return None

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8010)
