from flask import Flask, render_template, request, jsonify
import torch
from transformers import EncoderDecoderModel, BertJapaneseTokenizer, PreTrainedTokenizerFast

app = Flask(__name__)

# 모델과 토크나이저 로드
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "fine_tuned_ja_ko"

# 토크나이저 로드 - 원본 모델 구조에 맞게 수정
try:
    # 원본 모델에서 토크나이저 로드
    src_tok = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-v2")
    tgt_tok = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2")
    
    # 타겟 토크나이저 특수 토큰 설정
    if tgt_tok.pad_token is None:
        tgt_tok.add_special_tokens({"pad_token": "<pad>"})
    
    # 특수 토큰 ID 확인
    print(f"[DEBUG] 타겟 토크나이저 특수 토큰:")
    print(f"  - pad_token: {tgt_tok.pad_token} (ID: {tgt_tok.pad_token_id})")
    print(f"  - eos_token: {tgt_tok.eos_token} (ID: {tgt_tok.eos_token_id})")
    print(f"  - bos_token: {tgt_tok.bos_token} (ID: {tgt_tok.bos_token_id})")
        
except Exception as e:
    print(f"[ERROR] 토크나이저 로딩 실패: {e}")
    # 폴백: 저장된 모델에서 로드
    src_tok = PreTrainedTokenizerFast.from_pretrained(MODEL_PATH)
    tgt_tok = PreTrainedTokenizerFast.from_pretrained(MODEL_PATH)
    if tgt_tok.pad_token is None:
        tgt_tok.add_special_tokens({"pad_token": "<pad>"})

# 모델 로드
model = EncoderDecoderModel.from_pretrained(MODEL_PATH)

# 모델 설정을 config.json에 맞게 수정
model.config.pad_token_id = 3  # config.json에서 확인한 값
model.config.eos_token_id = 1  # config.json에서 확인한 값
model.config.decoder_start_token_id = 0  # config.json에서 확인한 값

# 토크나이저 임베딩 크기 조정
model.decoder.resize_token_embeddings(len(tgt_tok))

model.to(DEVICE)
model.eval()

print(f"[INFO] 모델이 {DEVICE}에 로드되었습니다.")
print(f"[INFO] 모델 파라미터: {sum(p.numel() for p in model.parameters())/1e6:.1f} M")
print(f"[INFO] 소스 토크나이저: {type(src_tok).__name__}")
print(f"[INFO] 타겟 토크나이저: {type(tgt_tok).__name__}")
print(f"[INFO] 모델 설정:")
print(f"  - pad_token_id: {model.config.pad_token_id}")
print(f"  - eos_token_id: {model.config.eos_token_id}")
print(f"  - decoder_start_token_id: {model.config.decoder_start_token_id}")

@app.route('/')
def index():
    return render_template('index.html')

@torch.inference_mode()
def translate_lyrics(japanese_text):
    """일본어 가사를 한국어로 번안"""
    try:
        print(f"[DEBUG] 입력 텍스트: {japanese_text}")
        
        # 입력 텍스트 토크나이징
        enc = src_tok(
            japanese_text, 
            truncation=True, 
            padding="max_length", 
            max_length=64,  # 길이 줄임
            return_tensors="pt"
        )
        
        print(f"[DEBUG] 토크나이징 완료: {enc['input_ids'].shape}")
        
        # GPU로 이동
        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)
        
        # 번안 생성 - 모델 설정 사용
        gen = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=64,  # 최대 길이 줄임
            min_length=1,   # 최소 길이 설정
            num_beams=1,    # greedy decoding으로 변경
            early_stopping=True,
            pad_token_id=model.config.pad_token_id,  # 모델 설정 사용
            eos_token_id=model.config.eos_token_id,  # 모델 설정 사용
            decoder_start_token_id=model.config.decoder_start_token_id,  # 모델 설정 사용
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.2,  # 반복 방지
            length_penalty=1.0,      # 길이 페널티
            no_repeat_ngram_size=0   # n-gram 제한 해제
        )
        
        print(f"[DEBUG] 생성 완료: {gen.shape}")
        print(f"[DEBUG] 생성된 토큰 ID: {gen[0].tolist()}")
        
        # 결과 디코딩
        translated_text = tgt_tok.batch_decode(gen, skip_special_tokens=True)[0]
        
        print(f"[DEBUG] 디코딩 결과: {translated_text}")
        
        # EOS 토큰 이후 텍스트 제거
        if tgt_tok.eos_token and tgt_tok.eos_token in translated_text:
            translated_text = translated_text.split(tgt_tok.eos_token)[0]
        
        return translated_text.strip()
        
    except Exception as e:
        print(f"[ERROR] 번안 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return japanese_text  # 오류 시 원본 반환

@torch.inference_mode()
def simple_translate(japanese_text):
    """간단한 번안 (기본 파라미터만 사용)"""
    try:
        print(f"[SIMPLE] 입력: {japanese_text}")
        
        # 토크나이징
        enc = src_tok(japanese_text, return_tensors="pt")
        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)
        
        # 기본 생성 - 모델 설정 사용
        gen = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=32,
            num_beams=1,
            early_stopping=True,
            pad_token_id=model.config.pad_token_id,
            eos_token_id=model.config.eos_token_id,
            decoder_start_token_id=model.config.decoder_start_token_id
        )
        
        print(f"[SIMPLE] 생성된 토큰 ID: {gen[0].tolist()}")
        
        # 디코딩
        result = tgt_tok.batch_decode(gen, skip_special_tokens=True)[0]
        print(f"[SIMPLE] 결과: {result}")
        
        return result.strip()
        
    except Exception as e:
        print(f"[SIMPLE] 오류: {e}")
        return japanese_text

@app.route('/test', methods=['GET'])
def test_translation():
    """간단한 테스트용 번안"""
    test_text = "こんにちは"
    try:
        # 간단한 번안 먼저 시도
        simple_result = simple_translate(test_text)
        
        # 복잡한 번안도 시도
        complex_result = translate_lyrics(test_text)
        
        return jsonify({
            'input': test_text,
            'simple_output': simple_result,
            'complex_output': complex_result,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'input': test_text,
            'error': str(e),
            'status': 'error'
        })

@app.route('/translate', methods=['POST'])
def translate():
    print("[INFO] 번안 요청 받음")
    
    try:
        data = request.get_json()
        print(f"[INFO] 요청 데이터: {data}")
        
        source_text = data.get('text', '')
        print(f"[INFO] 입력 텍스트: {source_text}")
        
        if not source_text.strip():
            print("[INFO] 빈 텍스트 요청")
            return jsonify({
                'translated_text': '',
                'furigana': ''
            })
        
        # 일본어 → 한국어 번안 수행
        print("[INFO] 번안 시작...")
        translated_text = translate_lyrics(source_text)
        print(f"[INFO] 번안 결과: {translated_text}")
        
        response_data = {
            'translated_text': translated_text,
            'furigana': ''  # TODO: 후리가나 정보 추가
        }
        
        print(f"[INFO] 응답 데이터: {response_data}")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"[ERROR] 번안 API 오류: {e}")
        return jsonify({
            'translated_text': f'오류 발생: {str(e)}',
            'furigana': ''
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
