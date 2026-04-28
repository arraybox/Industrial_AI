from flask import Flask, request, url_for, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return '🌐 홈 페이지입니다!'

@app.route('/user/<username>')
def user_profile(username):
    return f'👤 {username}님의 프로필 페이지입니다.'

@app.route('/post/<int:post_id>')
def post_detail(post_id):
    return f'📝 {post_id}번 게시글입니다.'

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        return '✅ POST 요청으로 제출되었습니다.'
    else:
        return """
            <form method='post'>
                <input type='submit' value='제출'>
            </form>
        """

@app.route('/goto_home')
def goto_home():
    return f'<a href="{url_for("home")}">홈으로 가기</a>'

@app.errorhandler(404)
def page_not_found(e):
    return render_template("error.html"), 404

if __name__ == '__main__':
    app.run(debug=True)
