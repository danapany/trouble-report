"""
RAG 엔진 모듈
- 질의 처리
- 벡터 검색
- LLM 답변 생성
"""

from typing import List, Dict, Optional
from openai import OpenAI
import os


class RAGEngine:
    def __init__(self, vector_store, openai_api_key: Optional[str] = None, 
                 model: str = "gpt-4o-mini"):
        """
        Args:
            vector_store: VectorStore 인스턴스
            openai_api_key: OpenAI API 키
            model: 사용할 GPT 모델
        """
        self.vector_store = vector_store
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        
        # OpenAI 클라이언트 초기화
        self.openai_client = OpenAI(api_key=self.openai_api_key)
    
    def retrieve_relevant_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        쿼리와 관련된 문서 검색
        
        Args:
            query: 사용자 질의
            top_k: 반환할 문서 수
        
        Returns:
            [{'text': str, 'metadata': dict, 'score': float}, ...]
        """
        # 벡터 검색
        results = self.vector_store.search(query, n_results=top_k)
        
        # 결과 포맷팅
        documents = []
        for i, doc in enumerate(results['documents']):
            documents.append({
                'text': doc,
                'metadata': results['metadatas'][i],
                'score': 1 - results['distances'][i]  # 거리를 유사도로 변환
            })
        
        return documents
    
    def generate_answer(self, query: str, context_documents: List[Dict], 
                       max_tokens: int = 1000) -> Dict:
        """
        검색된 문서를 기반으로 답변 생성
        
        Args:
            query: 사용자 질의
            context_documents: 검색된 문서 리스트
            max_tokens: 최대 토큰 수
        
        Returns:
            {
                'answer': str,
                'sources': List[dict]
            }
        """
        # 컨텍스트 구성
        context = self._build_context(context_documents)
        
        # 프롬프트 구성
        system_prompt = """당신은 사내 지식문서를 기반으로 답변하는 전문 AI 어시스턴트입니다.

다음 규칙을 따라주세요:
1. 제공된 문서 내용을 기반으로만 답변하세요.
2. 문서에 없는 내용은 추측하지 말고 "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 답하세요.
3. 답변은 명확하고 구조화되게 작성하세요.
4. 가능한 경우 출처 문서를 언급하세요.
5. 한국어로 답변하세요."""

        user_prompt = f"""다음은 사내 지식문서에서 검색된 관련 내용입니다:

{context}

질문: {query}

위 문서 내용을 바탕으로 질문에 답변해주세요."""

        try:
            # GPT API 호출
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3  # 일관성 있는 답변을 위해 낮은 temperature
            )
            
            answer = response.choices[0].message.content
            
            # 출처 정보 구성
            sources = []
            for doc in context_documents:
                source_info = {
                    'file_name': doc['metadata'].get('file_name', 'Unknown'),
                    'type': doc['metadata'].get('type', 'unknown'),
                    'score': round(doc['score'], 3)
                }
                if source_info not in sources:
                    sources.append(source_info)
            
            return {
                'answer': answer,
                'sources': sources
            }
        
        except Exception as e:
            print(f"답변 생성 오류: {str(e)}")
            return {
                'answer': "죄송합니다. 답변 생성 중 오류가 발생했습니다.",
                'sources': []
            }
    
    def _build_context(self, documents: List[Dict], max_length: int = 4000) -> str:
        """
        검색된 문서들을 컨텍스트로 구성
        
        Args:
            documents: 검색된 문서 리스트
            max_length: 최대 컨텍스트 길이 (문자 수)
        
        Returns:
            컨텍스트 텍스트
        """
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(documents):
            # 문서 정보
            file_name = doc['metadata'].get('file_name', 'Unknown')
            doc_type = doc['metadata'].get('type', 'text')
            
            # 컨텍스트 항목 구성
            if doc_type == 'ocr':
                context_item = f"[문서 {i+1}: {file_name} (이미지 내용)]\n{doc['text']}\n"
            else:
                context_item = f"[문서 {i+1}: {file_name}]\n{doc['text']}\n"
            
            # 길이 체크
            if current_length + len(context_item) > max_length:
                break
            
            context_parts.append(context_item)
            current_length += len(context_item)
        
        return "\n".join(context_parts)
    
    def query(self, query: str, top_k: int = 5, max_tokens: int = 1000) -> Dict:
        """
        전체 RAG 파이프라인 실행
        
        Args:
            query: 사용자 질의
            top_k: 검색할 문서 수
            max_tokens: 답변 최대 토큰 수
        
        Returns:
            {
                'query': str,
                'answer': str,
                'sources': List[dict],
                'retrieved_docs': List[dict]
            }
        """
        # 1. 관련 문서 검색
        retrieved_docs = self.retrieve_relevant_documents(query, top_k=top_k)
        
        if not retrieved_docs:
            return {
                'query': query,
                'answer': "관련된 문서를 찾을 수 없습니다. 다른 질문을 시도해보세요.",
                'sources': [],
                'retrieved_docs': []
            }
        
        # 2. 답변 생성
        result = self.generate_answer(query, retrieved_docs, max_tokens=max_tokens)
        
        return {
            'query': query,
            'answer': result['answer'],
            'sources': result['sources'],
            'retrieved_docs': retrieved_docs[:3]  # 상위 3개만 포함
        }


if __name__ == "__main__":
    # 테스트 코드
    from vector_store import VectorStore
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    db_path = os.getenv('DB_BASE_PATH', './vectordb')
    vector_store = VectorStore(db_path)
    
    rag_engine = RAGEngine(vector_store)
    
    # 테스트 질의
    query = "인공지능이란 무엇인가?"
    result = rag_engine.query(query)
    
    print(f"질문: {result['query']}")
    print(f"답변: {result['answer']}")
    print(f"출처: {result['sources']}")
