from flask import Flask, request, jsonify
import boto3
import json

app = Flask(__name__)

# Initialize SageMaker runtime client
sagemaker_runtime = boto3.client('sagemaker-runtime')

# Define your SageMaker endpoint name
ENDPOINT_NAME = "xgboost-2025-03-17-23-45-40-937"  # Replace with your actual SageMaker endpoint name

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        input_data = request.get_json()

        # Convert the input data into JSON string format for SageMaker
        csv_data = ",".join(map(str, input_data["input"]))  # Convert list to CSV string

        # Invoke SageMaker endpoint
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="text/csv",  # Change Content-Type to text/csv
    Body=csv_data  # Send CSV string, not JSON
        )

        # Parse the response
        result = json.loads(response['Body'].read().decode())

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
