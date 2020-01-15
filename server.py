from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return "<h1> Hello </h1>"

host_addr = "0.0.0.0"
port_num = "8000"

if __name__=="__main__":
    app.run(host=host_addr, port=port_num)