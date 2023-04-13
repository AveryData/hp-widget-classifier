"# HP-Manufacturing-Image-Detection" 


# Instructions 
1. Copy the files locally to your machine. Probably easiest to just use: Be sure to get the `"casting_production_detection.hdf5"` file from Google Drive. This is the saved ML model which is used for predictions.

I reccomend using: `git clone https://github.com/AveryData/hp-widget-classifier`

Put the .hdf5 file in the "hp-widget-classifier" folder

2. Open to the correct folder "hp-widget-classifier" in a command prompt 
3. Create a virtual environment for the project using the following command:`conda create -n myenv python=3.10`. Note the 3.10 Python is needed to keep the newer versions of TensorFlow & Streamlit happy. 
4. Activate the venv by typing `activate myenv`
5. Install the required packages using pip by running the following command: `pip install -r requirements.txt`
6. Launch the Streamlit app by running the following command: `streamlit run app.py`
