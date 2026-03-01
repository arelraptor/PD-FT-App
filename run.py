import os
import sys
import platform


def add_ffmpeg_to_path():
    env_path = os.path.dirname(sys.executable)

    if sys.platform == "win32":
        ffmpeg_bin = os.path.join(env_path, 'Library', 'bin')
    else:
        ffmpeg_bin = os.path.join(env_path, 'bin')

    if ffmpeg_bin not in os.environ["PATH"]:
        os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ["PATH"]

add_ffmpeg_to_path()

from app import create_app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000)
    #app.run(host='0.0.0.0', port=5000, ssl_context='adhoc')