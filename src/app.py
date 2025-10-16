"""
RAG 챗봇 Streamlit 앱
"""

import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
import sys

# 환경 변수 로드
load_dotenv()

# 모듈 임포트
from chatbot import RAGChatbot

# 페이지 설정
st.set_page_config(
    page_title="사내 지식문서 RAG 챗봇",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #616161;
        margin-bottom: 2rem;
    }
    .stat-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .source-box {
        background-color: #e3f2fd;
        padding: 0.8rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
        border-left: 3px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_chatbot():
    """챗봇 초기화 (캐시)"""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    db_path = os.getenv('DB_BASE_PATH', './vectordb')
    
    return RAGChatbot(data_dir, db_path)


def main():
    # 헤더
    st.markdown('<div class="main-header">🤖 사내 지식문서 RAG 챗봇</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Word 문서 기반 의미 검색 & 답변 생성</div>', unsafe_allow_html=True)
    
    # 챗봇 초기화
    try:
        chatbot = initialize_chatbot()
    except Exception as e:
        st.error(f"❌ 챗봇 초기화 오류: {str(e)}")
        st.info("💡 .env 파일에 OPENAI_API_KEY가 설정되어 있는지 확인하세요.")
        return
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 시스템 통계
        stats = chatbot.get_stats()
        
        st.subheader("📊 시스템 통계")
        st.markdown(f"""
        <div class="stat-box">
            <strong>📁 Word 문서:</strong> {stats['document_count']}개<br>
            <strong>🔢 벡터 DB 항목:</strong> {stats['vector_count']}개<br>
            <strong>📂 문서 경로:</strong> {stats['documents_dir']}<br>
            <strong>💾 DB 경로:</strong> {stats['db_path']}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 인덱싱 설정
        st.subheader("🔄 문서 인덱싱")
        
        enable_ocr = st.checkbox("OCR 활성화 (이미지 텍스트 추출)", value=True)
        use_gpu = st.checkbox("GPU 사용 (OCR 가속)", value=False)
        
        if st.button("🚀 문서 인덱싱 시작", type="primary", use_container_width=True):
            with st.spinner("문서 인덱싱 중... 시간이 걸릴 수 있습니다."):
                result = chatbot.index_documents(enable_ocr=enable_ocr, gpu=use_gpu)
                
                if result['status'] == 'success':
                    st.success("✅ 인덱싱 완료!")
                    st.info(f"""
                    - 처리된 문서: {result['total_docs']}개
                    - 텍스트 청크: {result['total_chunks']}개
                    - 추출된 이미지: {result['total_images']}개
                    - OCR 텍스트: {result['ocr_texts']}개
                    """)
                    st.rerun()
                elif result['status'] == 'no_documents':
                    st.warning("⚠️ 처리할 문서가 없습니다. data/documents/ 폴더에 .docx 파일을 추가하세요.")
                else:
                    st.error("❌ 인덱싱 중 오류가 발생했습니다.")
        
        st.markdown("---")
        
        # 데이터베이스 초기화
        st.subheader("🗑️ 데이터베이스 관리")
        if st.button("⚠️ DB 초기화", use_container_width=True):
            if st.session_state.get('confirm_reset', False):
                chatbot.reset_database()
                st.success("✅ 데이터베이스가 초기화되었습니다.")
                st.session_state['confirm_reset'] = False
                st.rerun()
            else:
                st.session_state['confirm_reset'] = True
                st.warning("⚠️ 다시 한 번 클릭하면 모든 데이터가 삭제됩니다!")
        
        st.markdown("---")
        
        # 검색 설정
        st.subheader("🔍 검색 설정")
        top_k = st.slider("검색할 문서 수", min_value=1, max_value=10, value=5)
    
    # 메인 영역
    # 탭 구성
    tab1, tab2, tab3 = st.tabs(["💬 채팅", "📚 사용 가이드", "ℹ️ 정보"])
    
    with tab1:
        # 채팅 인터페이스
        st.subheader("💬 질문하기")
        
        # 세션 상태 초기화
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # 질문 입력
        query = st.text_input(
            "질문을 입력하세요:",
            placeholder="예: 회사의 복지 제도에 대해 알려주세요.",
            key="query_input"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            search_button = st.button("🔍 검색", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ 대화 기록 삭제", use_container_width=True)
        
        if clear_button:
            st.session_state.chat_history = []
            st.rerun()
        
        # 검색 실행
        if search_button and query:
            with st.spinner("답변 생성 중..."):
                result = chatbot.chat(query, top_k=top_k)
                
                # 대화 기록에 추가
                st.session_state.chat_history.append({
                    'query': query,
                    'answer': result['answer'],
                    'sources': result['sources']
                })
        
        # 대화 기록 표시
        if st.session_state.chat_history:
            st.markdown("---")
            st.subheader("📝 대화 기록")
            
            for i, item in enumerate(reversed(st.session_state.chat_history)):
                with st.container():
                    # 질문
                    st.markdown(f"**👤 질문 #{len(st.session_state.chat_history) - i}:**")
                    st.info(item['query'])
                    
                    # 답변
                    st.markdown("**🤖 답변:**")
                    st.success(item['answer'])
                    
                    # 출처
                    if item['sources']:
                        with st.expander("📚 참고 문서"):
                            for source in item['sources']:
                                st.markdown(f"""
                                <div class="source-box">
                                    <strong>📄 {source['file_name']}</strong><br>
                                    유형: {source['type']}<br>
                                    유사도: {source['score']}
                                </div>
                                """, unsafe_allow_html=True)
                    
                    st.markdown("---")
        else:
            st.info("👋 질문을 입력하고 검색 버튼을 클릭하세요!")
    
    with tab2:
        st.markdown("""
        ## 📚 사용 가이드
        
        ### 1️⃣ 문서 준비
        - `data/documents/` 폴더에 Word 문서(.docx)를 추가하세요.
        - 문서는 자동으로 텍스트와 이미지로 분리됩니다.
        
        ### 2️⃣ 문서 인덱싱
        - 사이드바에서 "문서 인덱싱 시작" 버튼을 클릭하세요.
        - OCR을 활성화하면 이미지 내 텍스트도 검색 가능합니다.
        - 인덱싱은 최초 1회만 수행하면 됩니다.
        
        ### 3️⃣ 질문하기
        - 자연어로 질문을 입력하세요.
        - 시스템이 관련 문서를 검색하고 답변을 생성합니다.
        - 답변의 출처 문서를 확인할 수 있습니다.
        
        ### 4️⃣ 고급 설정
        - **검색할 문서 수**: 더 많은 문서를 검색하면 답변이 더 포괄적이지만 시간이 더 걸립니다.
        - **OCR 활성화**: 이미지 내 텍스트도 검색하려면 활성화하세요.
        - **GPU 사용**: OCR 속도를 높이려면 GPU를 활성화하세요 (CUDA 필요).
        
        ### 💡 팁
        - 구체적인 질문일수록 정확한 답변을 얻을 수 있습니다.
        - 문서가 업데이트되면 재인덱싱이 필요합니다.
        - 검색 결과가 만족스럽지 않으면 질문을 다시 표현해보세요.
        """)
    
    with tab3:
        st.markdown("""
        ## ℹ️ 시스템 정보
        
        ### 🏗️ 아키텍처
        - **벡터 DB**: ChromaDB (오픈소스)
        - **임베딩**: OpenAI text-embedding-3-small
        - **LLM**: OpenAI GPT-4o-mini
        - **OCR**: EasyOCR (한글/영문)
        - **프론트엔드**: Streamlit
        
        ### 📁 프로젝트 구조
        ```
        rag_chatbot_project/
        ├── src/
        │   ├── document_processor.py   # 문서 처리
        │   ├── ocr_processor.py        # OCR 처리
        │   ├── vector_store.py         # 벡터 DB
        │   ├── rag_engine.py           # RAG 엔진
        │   ├── chatbot.py              # 챗봇 로직
        │   └── app.py                  # Streamlit 앱
        ├── data/
        │   ├── documents/              # Word 문서
        │   └── images/                 # 추출된 이미지
        ├── vectordb/                   # 벡터 DB
        ├── .env                        # 환경 변수
        └── requirements.txt            # 패키지 목록
        ```
        
        ### 🔧 기술 스택
        - Python 3.8+
        - Streamlit
        - ChromaDB
        - OpenAI API
        - python-docx
        - EasyOCR
        - Pillow
        
        ### 📝 버전
        - Version: 1.0.0
        - Last Updated: 2024
        
        ### 📄 라이선스
        MIT License
        """)


if __name__ == "__main__":
    main()
