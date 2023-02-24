from flask import Flask, request
from flask_cors import cross_origin
from ast import literal_eval
from recommend_image import recommend_image
from recommend_text import recommend_text

app = Flask(__name__)


@app.route("/")
@cross_origin()
def main():
    return "Use /recommend/text or /recommend/image to get Fashion recommendations based on text or image inputs respectively"


@app.route("/recommend/text", methods=["GET", "POST"])
@cross_origin()
def recommendText():
    if request.method == "POST":
        data = request.data
        #search_item = literal_eval(data.decode('utf8'))
        #search_item = data
        return recommend_text(data["search"])


@app.route("/recommend/image", methods=["GET", "POST"])
@cross_origin()
def recommendImg():
    if request.method == "POST":
        data = request.data
        search_item = literal_eval(data.decode('utf8'))
        return recommend_image(search_item["search"])


# run the api
class FashionApi:
    def start(self):
        app.run(debug=True, use_reloader=False)