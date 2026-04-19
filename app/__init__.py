from flask import Flask
from flask_bootstrap import Bootstrap5
from ybjy_config import APP_TITLE, SECRET_KEY, MAX_CONTENT_LENGTH, BOOTSTRAP_SERVE_LOCAL
from app.ybjy_database import initialize_yangben_yujian_database
from app.ybjy_routes import register_yangben_yujian_routes

bootstrap = Bootstrap5()


def create_yangben_yujian_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = SECRET_KEY
    app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
    app.config['BOOTSTRAP_SERVE_LOCAL'] = BOOTSTRAP_SERVE_LOCAL
    app.config['APP_TITLE'] = APP_TITLE
    bootstrap.init_app(app)
    initialize_yangben_yujian_database()
    register_yangben_yujian_routes(app)
    return app
