from quotator import app
from flask import render_template
import torch
import quotator.model_utils as utils

quote = ""

@app.route("/")
@app.route("/home")
@app.route("/index")
def index():
    quote = utils.generate()
    return render_template("index.html", quote=quote)

@app.errorhandler(404)
def page_not_found(e):
    quote = "Page Not Found 🥺 Click Here !"
    return render_template("index.html", quote=quote)