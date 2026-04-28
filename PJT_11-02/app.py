from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# 가상의 유저 데이터
USER_ID = "admin"
USER_PW = "1234"

@app.route('/')
def home():
    return "<h2>홈페이지 - <a href='/login'>로그인</a></h2>"

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = request.form['username']
        user_pw = request.form['password']

        # if문을 이용한 로그인 검증
        if user_id == USER_ID and user_pw == USER_PW:
            return f"<h3>안녕하세요, {user_id}님!</h3>"
        else:
            # 실패했을 경우 다시 로그인 페이지로, 에러 메시지 포함
            return render_template('login.html', error="아이디 또는 비밀번호가 틀렸습니다.")

    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True)
