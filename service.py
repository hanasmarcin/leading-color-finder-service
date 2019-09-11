from flask import Flask, request, abort, jsonify
from ColorFinder import ColorFinder


app = Flask(__name__)


@app.route("/")
def hello():
    return 'hello from flask'


@app.route("/colors", methods=["GET"])
def get_colors_for_image():
    url = request.args.get("url")
    if url is None:
        abort(400, "Url was not provided.")
    try:
        color_finder = ColorFinder(url, 5)
        colors, pixel_count_for_colors = color_finder.get_colors()
        response = [{"color": colors[i], "pixel_count": pixel_count_for_colors[i]}
                    for i in range(len(pixel_count_for_colors))]
        return jsonify(response)
    except OSError:
        abort(400, "Wrong url was provided.")


if __name__ == '__main__':
    app.run(debug=True)
