from flask import Flask, render_template, request

import requests


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/summary", methods=["get", "post"])
def summary():
    result = ""
    data = ""
    # if request.method == "post":
    API_URL = "https://api-inference.huggingface.co/models/pszemraj/pegasus-x-large-book-summary"
    headers = {"Authorization": "Bearer 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"}
    input_data = request.form.get("data")
    print("data",input_data)
    maxL = len(input_data.split())
    minL =  int(maxL *0.3) 
    print("original length:",maxL)
    print("Summary length:",minL)
    if(input_data == ""):
        return render_template("index.html",result=input_data,input_data=input_data )
        # return input_data
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    output = query({"inputs": input_data,"parameters": {"min_length": minL},})
    result = str(output[0]["summary_text"])+"."
    print("Summary:")
    print(result)
    # print(result)
    return render_template("index.html",result=result,input_data=input_data )
    # return result


if __name__ == "__main__":
    app.run(debug=True)
