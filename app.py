from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    source_text = data.get('text', '')
    source_lang = data.get('source_lang', 'ko')
    target_lang = data.get('target_lang', 'ja')
    
    # TODO: 실제 번역 로직 구현
    translated_text = source_text  # 임시로 원본 텍스트 반환
    
    return jsonify({
        'translated_text': translated_text,
        'furigana': ''  # TODO: 후리가나 정보 추가
    })

if __name__ == '__main__':
    app.run(debug=True)
