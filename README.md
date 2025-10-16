# 사내 지식문서 RAG 챗봇 시스템

## 📌 프로젝트 개요
Word 문서 1000개를 기반으로 한 LLM 기반 RAG(Retrieval-Augmented Generation) 챗봇 시스템입니다.
문서 내 텍스트와 이미지(OCR)를 모두 벡터화하여 의미 기반 검색을 제공합니다.

## 🏗️ 시스템 아키텍처
- **벡터 DB**: ChromaDB (오픈소스)
- **OCR**: EasyOCR
- **임베딩**: OpenAI text-embedding-3-small
- **LLM**: OpenAI GPT-4
- **프론트엔드**: Streamlit

## 📁 프로젝트 구조
```
rag_chatbot_project/
├── src/
│   ├── document_processor.py   # Word 문서 처리 및 텍스트 추출
│   ├── ocr_processor.py        # 이미지 OCR 처리
│   ├── vector_store.py         # ChromaDB 벡터 저장소 관리
│   ├── rag_engine.py           # RAG 엔진 (검색 + 생성)
│   ├── chatbot.py              # 챗봇 로직
│   └── app.py                  # Streamlit 메인 앱
├── data/
│   └── documents/              # Word 문서 저장 폴더 (여기에 .docx 파일 추가)
├── .env                        # 환경 변수
├── requirements.txt            # Python 패키지 목록
└── README.md                   # 프로젝트 설명
```

## 🚀 설치 및 실행

### 1. 환경 설정
```bash
# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정
`.env` 파일에 다음 내용을 설정하세요:
```
OPENAI_API_KEY=your_openai_api_key_here
DB_BASE_PATH=./vectordb
```

### 3. Word 문서 추가
`data/documents/` 폴더에 Word 문서(.docx)를 추가하세요.

### 4. 문서 인덱싱 (최초 1회)
```bash
streamlit run src/app.py
```
- 웹 UI에서 "문서 인덱싱 시작" 버튼을 클릭하여 문서를 벡터화합니다.
- 1000개 문서는 시간이 걸릴 수 있습니다 (약 30분~1시간)

### 5. 챗봇 사용
- 인덱싱 완료 후 질문을 입력하면 의미 기반 검색으로 답변을 생성합니다.

## 🎯 주요 기능
1. **문서 처리**: Word 문서 텍스트 및 이미지 자동 추출
2. **OCR 처리**: 이미지 내 한글/영문 텍스트 인식
3. **청킹 전략**: 500자 단위로 문서 분할 (오버랩 100자)
4. **벡터 검색**: 의미 기반 유사도 검색 (Top-5)
5. **컨텍스트 기반 답변**: 검색된 문서 기반 LLM 답변 생성

## 📊 성능 최적화
- **배치 처리**: 문서를 배치로 처리하여 API 호출 최소화
- **캐싱**: 처리된 문서는 벡터 DB에 저장되어 재처리 불필요
- **청킹**: 적절한 크기로 문서를 분할하여 검색 정확도 향상

## 🔧 문제 해결
- **OCR 느림**: GPU 사용 시 속도 향상 가능 (CUDA 설치 필요)
- **메모리 부족**: 배치 크기 줄이기 (vector_store.py의 batch_size 조정)
- **API 비용**: 로컬 임베딩 모델로 교체 가능 (sentence-transformers)

## 📝 라이선스
MIT License
