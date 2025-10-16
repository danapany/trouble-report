"""
벡터 저장소 관리 모듈
- ChromaDB를 사용한 벡터 저장 및 검색
- OpenAI Embedding API 활용
"""

import os
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm


class VectorStore:
    def __init__(self, db_path: str, collection_name: str = "rag_documents", 
                 openai_api_key: Optional[str] = None):
        """
        Args:
            db_path: ChromaDB 저장 경로
            collection_name: 컬렉션 이름
            openai_api_key: OpenAI API 키
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        # OpenAI 클라이언트 초기화
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        # ChromaDB 클라이언트 초기화
        self._initialize_chroma()
    
    def _initialize_chroma(self):
        """ChromaDB 클라이언트 및 컬렉션 초기화"""
        try:
            # ChromaDB 저장 경로 생성
            Path(self.db_path).mkdir(parents=True, exist_ok=True)
            
            # ChromaDB 클라이언트 생성
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # 컬렉션 가져오기 또는 생성
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                print(f"기존 컬렉션 '{self.collection_name}' 로드 완료")
            except:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}  # 코사인 유사도 사용
                )
                print(f"새 컬렉션 '{self.collection_name}' 생성 완료")
        
        except Exception as e:
            print(f"ChromaDB 초기화 오류: {str(e)}")
            raise
    
    def get_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """
        OpenAI API를 사용하여 텍스트 임베딩 생성
        
        Args:
            text: 임베딩할 텍스트
            model: OpenAI 임베딩 모델
        
        Returns:
            임베딩 벡터
        """
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"임베딩 생성 오류: {str(e)}")
            return []
    
    def get_embeddings_batch(self, texts: List[str], model: str = "text-embedding-3-small",
                            batch_size: int = 100) -> List[List[float]]:
        """
        여러 텍스트의 임베딩을 배치로 생성
        
        Args:
            texts: 임베딩할 텍스트 리스트
            model: OpenAI 임베딩 모델
            batch_size: 배치 크기
        
        Returns:
            임베딩 벡터 리스트
        """
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="임베딩 생성 중"):
            batch = texts[i:i + batch_size]
            try:
                response = self.openai_client.embeddings.create(
                    input=batch,
                    model=model
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"배치 임베딩 생성 오류: {str(e)}")
                # 오류 발생 시 빈 임베딩 추가
                embeddings.extend([[] for _ in batch])
        
        return embeddings
    
    def add_documents(self, documents: List[Dict], batch_size: int = 100):
        """
        문서를 벡터 저장소에 추가
        
        Args:
            documents: [{
                'id': str,
                'text': str,
                'metadata': dict
            }]
            batch_size: 배치 크기
        """
        if not documents:
            print("추가할 문서가 없습니다.")
            return
        
        print(f"\n총 {len(documents)}개 문서를 벡터 저장소에 추가합니다...")
        
        # 텍스트 추출
        texts = [doc['text'] for doc in documents]
        ids = [doc['id'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        # 임베딩 생성
        embeddings = self.get_embeddings_batch(texts, batch_size=batch_size)
        
        # ChromaDB에 추가
        try:
            for i in tqdm(range(0, len(documents), batch_size), desc="DB에 저장 중"):
                batch_ids = ids[i:i + batch_size]
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_texts,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas
                )
            
            print(f"✓ {len(documents)}개 문서 추가 완료")
        except Exception as e:
            print(f"문서 추가 오류: {str(e)}")
    
    def search(self, query: str, n_results: int = 5) -> Dict:
        """
        쿼리와 유사한 문서 검색
        
        Args:
            query: 검색 쿼리
            n_results: 반환할 결과 수
        
        Returns:
            {
                'documents': List[str],
                'metadatas': List[dict],
                'distances': List[float]
            }
        """
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.get_embedding(query)
            
            if not query_embedding:
                return {'documents': [], 'metadatas': [], 'distances': []}
            
            # 유사 문서 검색
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            return {
                'documents': results['documents'][0] if results['documents'] else [],
                'metadatas': results['metadatas'][0] if results['metadatas'] else [],
                'distances': results['distances'][0] if results['distances'] else []
            }
        except Exception as e:
            print(f"검색 오류: {str(e)}")
            return {'documents': [], 'metadatas': [], 'distances': []}
    
    def get_collection_count(self) -> int:
        """컬렉션의 문서 수 반환"""
        try:
            return self.collection.count()
        except:
            return 0
    
    def reset_collection(self):
        """컬렉션 초기화 (모든 데이터 삭제)"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"컬렉션 '{self.collection_name}' 초기화 완료")
        except Exception as e:
            print(f"컬렉션 초기화 오류: {str(e)}")


if __name__ == "__main__":
    # 테스트 코드
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    db_path = os.getenv('DB_BASE_PATH', './vectordb')
    vector_store = VectorStore(db_path)
    
    # 테스트 문서 추가
    test_docs = [
        {
            'id': 'test_1',
            'text': '인공지능은 컴퓨터 과학의 한 분야입니다.',
            'metadata': {'source': 'test', 'type': 'text'}
        }
    ]
    
    vector_store.add_documents(test_docs)
    print(f"컬렉션 문서 수: {vector_store.get_collection_count()}")
