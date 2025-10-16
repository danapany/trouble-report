# 빠른 시작 가이드 (Quick Start Guide)

## 🚀 5분 안에 시작하기

### 1단계: 환경 설정
```bash
# 가상환경 생성 (선택사항)
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

### 2단계: API 키 설정
`.env` 파일을 열고 OpenAI API 키를 입력하세요:
```
OPENAI_API_KEY=sk-your-api-key-here
DB_BASE_PATH=./vectordb
```

💡 OpenAI API 키 발급: https://platform.openai.com/api-keys

### 3단계: 문서 추가
`data/documents/` 폴더에 Word 문서(.docx)를 추가하세요.

### 4단계: 앱 실행
```bash
streamlit run src/app.py
```

브라우저에서 자동으로 열립니다 (보통 http://localhost:8501)

### 5단계: 문서 인덱싱
1. 사이드바에서 "문서 인덱싱 시작" 버튼 클릭
2. 인덱싱 완료까지 대기 (문서 수에 따라 시간 소요)

### 6단계: 질문하기!
이제 질문을 입력하고 답변을 받아보세요! 🎉

---

## ❓ 문제 해결

### 패키지 설치 오류
```bash
# pip 업그레이드
python -m pip install --upgrade pip

# 패키지 재설치
pip install -r requirements.txt --upgrade
```

### OpenAI API 오류
- API 키가 올바른지 확인
- 계정에 크레딧이 있는지 확인
- 인터넷 연결 확인

### OCR 느림
- GPU 옵션 활성화 (CUDA 설치 필요)
- 또는 OCR 비활성화하고 텍스트만 사용

### 메모리 부족
- `vector_store.py`의 `batch_size` 줄이기
- 한 번에 처리하는 문서 수 줄이기

---

## 💡 팁

### 효율적인 질문 방법
✅ 좋은 질문:
- "회사의 연차 휴가 정책은 어떻게 되나요?"
- "신입사원 온보딩 절차를 알려주세요."
- "보안 정책에 대해 설명해주세요."

❌ 피해야 할 질문:
- "이거" (너무 모호함)
- "다 알려줘" (범위가 너무 넓음)

### 검색 설정 조정
- 일반적인 질문: 검색 문서 수 3-5개
- 복잡한 질문: 검색 문서 수 7-10개
- 정확한 답변 필요: OCR 활성화

### 성능 최적화
- 문서를 주제별로 구분하여 저장
- 정기적으로 DB 최적화
- 불필요한 문서는 제거

---

## 📞 지원

문제가 있으면 다음을 확인하세요:
1. README.md - 전체 문서
2. 사용 가이드 탭 - 앱 내 가이드
3. 에러 메시지 - 콘솔 로그 확인
