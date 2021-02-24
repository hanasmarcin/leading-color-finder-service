from io import BytesIO

from flask import Flask, request, abort, jsonify, send_file
from ColorFinder import ColorFinder
import WallpaperCreator
import resource

app = Flask(__name__)
resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 1000 * 1024 * 1024))


@app.route("/")
def hello():
    return 'hello from flask'


@app.route("/colors", methods=["GET"])
def get_colors_for_image():
    url = request.args.get("url")
    if url is None:
        abort(400, "Url was not provided.")
    try:
        color_finder = ColorFinder(url, 6)
        colors, pixel_count_for_colors = color_finder.get_colors()
        response = [{"color": colors[i], "pixel_count": pixel_count_for_colors[i]}
                    for i in range(len(pixel_count_for_colors))]
        return jsonify(response)
    except OSError:
        abort(400, "Wrong url was provided.")


@app.route("/wallpaper", methods=["GET"])
def get_wallpaper_for_colors():
    colors = request.json
    if colors is None:
        abort(400, "Colors were not provided.")
    try:
        colors = list(colors)
        # file_obj = BytesIO()
        pil_img = WallpaperCreator.create_img(2340, 1080, colors)
        # pil_img.save(file_obj, 'PNG')
        # file_obj.seek(0)
        abort(400, "Oki ;)")
        # return send_file(file_obj, mimetype='image/png')
    except OSError:
        abort(400, "Wrong url was provided.")


if __name__ == '__main__':
    app.run(debug=False)
