import requests

# Specify the URL of your API
url = 'http://127.0.0.1:5000/transcribe'

# Open your WAV file in binary mode
with open(r"C:\Users\admin\Downloads\audio.wav", 'rb') as f:
    # Create a dictionary with your file under the key 'file'
    files = {'file': f}
    
    # Make the POST request and get the response
    response = requests.post(url, files=files)

# Print the response
print(response.json())
