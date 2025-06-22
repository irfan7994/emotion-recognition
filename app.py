from flask import Flask, render_template
import threading
import camera

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start_video():
    thread = threading.Thread(target=camera.detect_emotion)
    thread.start()
    return render_template('video.html')

if __name__ == '__main__':
    app.run(debug=True)