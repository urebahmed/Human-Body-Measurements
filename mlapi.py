# bring lightweight libraries

import subprocess
from vidFeed import main
import skimage.io as im
import json
import numpy

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
# import vidFeed
import papa
app = FastAPI()

# define a function to be called when the api is called
path = './sample_data/input/user_image_.jpg'



@app.route('/', methods=['GET'])
def predict(request: Request):
    measurements_dict = papa.main() # gives measurement in ndarray form
    measurements = json.dumps(measurements_dict) 
    
    return JSONResponse(status_code=200, content=measurements_dict)


# run the app
if __name__ == '__main__':
    app.run(debug=True)
