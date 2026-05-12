# PJT_11-03

Flask 라우팅 기능을 연습하는 예제 폴더입니다. 정적 경로, 동적 URL 파라미터, 정수형 라우트 변수, GET/POST 처리, `url_for`, 404 에러 핸들러를 한 파일에서 확인할 수 있습니다.

## 파일 구성

| 파일 | 설명 |
|---|---|
| `app.py` | Flask 라우팅 예제 애플리케이션 |
| `templates/error.html` | 존재하지 않는 페이지 요청 시 표시되는 404 템플릿 |

## 라우트 구성

| 경로 | 설명 |
|---|---|
| `/` | 홈 페이지 문자열 반환 |
| `/user/<username>` | URL의 사용자 이름을 받아 프로필 문구 반환 |
| `/post/<int:post_id>` | 정수형 게시글 번호를 받아 상세 문구 반환 |
| `/submit` | GET 요청이면 제출 폼 표시, POST 요청이면 제출 완료 문구 반환 |
| `/goto-home` | `url_for('home')`로 생성한 홈 경로 반환 |
| 404 handler | 등록되지 않은 경로 요청 시 `error.html` 렌더링 |

## 실행 방법

```bash
pip install flask
python app.py
```

실행 후 아래 주소들을 브라우저에서 확인합니다.

```text
http://127.0.0.1:5000/
http://127.0.0.1:5000/user/alice
http://127.0.0.1:5000/post/3
http://127.0.0.1:5000/submit
```

## 학습 포인트

- Flask의 동적 라우팅
- URL 변수 타입 지정
- `request.method`를 이용한 GET/POST 분기
- `url_for()` 사용
- 커스텀 404 에러 페이지 작성
## 작성자 정보

| 항목 | 내용 |
|---|---|
| 작성자 | arraybox |
| 이름 | 이일주 |
| 이메일 | arraybox@chungbuk.ac.kr |
| 학번 | 2025254015 |


