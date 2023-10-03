import base64
from gradio_client import Client
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the full path to the file
file_path = os.path.join(script_dir, 'result.txt')

client = Client("http://semantic-sam.xyzou.net:6090/")

# Convert image to base64
with open("crossfar.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

# Form the data URL for the image
image_data_url = "data:image/png;base64," + encoded_string

try:
    result = client.predict(
				"https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png",	# str (filepath or URL to image)
				["Stroke"],	# List[str] in 'Interative Mode' Checkboxgroup component
				"https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png",	# str (filepath or URL to image)
				"Howdy!",	# str in '[Text] Referring Text' Textbox component
				"https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav",	# str (filepath or URL to file)
				"https://github.com/gradio-app/gradio/raw/main/test/test_files/video_sample.mp4",	# str (filepath or URL to file)
				api_name="/predict"
)
except Exception as e:
    print(f"An error occurred while making the prediction: {e}")
    with open(file_path, 'w') as f:
        f.write(str(e))
else:
    # print(result)
    with open(file_path, 'w') as f:
        f.write(str("good_result.txt"))