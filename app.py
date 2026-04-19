from app import create_yangben_yujian_app

app = create_yangben_yujian_app()

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
