"""
챗봇 로직 모듈
- 문서 인덱싱 파이프라인
- 챗봇 인터페이스
"""

import os
from typing import List, Dict, Optional
from pathlib import Path
from tqdm import tqdm
import uuid

from document_processor import DocumentProcessor
from ocr_processor import OCRProcessor
from vector_store import VectorStore
from rag_engine import RAGEngine


class RAGChatbot:
    def __init__(self, data_dir: str, db_path: str, 
                 openai_api_key: Optional[str] = None):
        """
        Args:
            data_dir: 문서 데이터 디렉토리
            db_path: 벡터 DB 경로
            openai_api_key: OpenAI API 키
        """
        self.data_dir = data_dir
        self.db_path = db_path
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        # 디렉토리 경로 설정
        self.documents_dir = Path(data_dir) / "documents"
        self.images_dir = Path(data_dir) / "images"
        
        # 디렉토리 생성
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # 컴포넌트 초기화
        self.doc_processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
        self.ocr_processor = None  # 필요시 초기화
        self.vector_store = VectorStore(db_path, openai_api_key=self.openai_api_key)
        self.rag_engine = RAGEngine(self.vector_store, openai_api_key=self.openai_api_key)
    
    def initialize_ocr(self, gpu: bool = False):
        """OCR 프로세서 초기화 (필요시 호출)"""
        if self.ocr_processor is None:
            self.ocr_processor = OCRProcessor(languages=['ko', 'en'], gpu=gpu)
    
    def index_documents(self, enable_ocr: bool = True, gpu: bool = False) -> Dict:
        """
        문서 인덱싱 파이프라인 실행
        
        Args:
            enable_ocr: OCR 활성화 여부
            gpu: GPU 사용 여부
        
        Returns:
            {
                'total_docs': int,
                'total_chunks': int,
                'total_images': int,
                'ocr_texts': int,
                'status': str
            }
        """
        print("\n" + "="*60)
        print("문서 인덱싱 시작")
        print("="*60)
        
        stats = {
            'total_docs': 0,
            'total_chunks': 0,
            'total_images': 0,
            'ocr_texts': 0,
            'status': 'success'
        }
        
        try:
            # 1. Word 문서 처리
            print("\n[1/4] Word 문서 처리 중...")
            processed_docs = self.doc_processor.process_documents_batch(
                str(self.documents_dir),
                str(self.images_dir)
            )
            
            if not processed_docs:
                print("⚠ 처리할 문서가 없습니다. data/documents/ 폴더에 .docx 파일을 추가하세요.")
                stats['status'] = 'no_documents'
                return stats
            
            stats['total_docs'] = len(processed_docs)
            print(f"✓ {stats['total_docs']}개 문서 처리 완료")
            
            # 2. 텍스트 청크를 벡터 DB에 추가
            print("\n[2/4] 텍스트 청크 벡터화 중...")
            text_documents = []
            
            for doc in processed_docs:
                for i, chunk in enumerate(doc['chunks']):
                    doc_id = f"{Path(doc['file_name']).stem}_chunk_{i}"
                    text_documents.append({
                        'id': doc_id,
                        'text': chunk,
                        'metadata': {
                            'file_name': doc['file_name'],
                            'file_path': doc['file_path'],
                            'chunk_index': i,
                            'type': 'text'
                        }
                    })
            
            stats['total_chunks'] = len(text_documents)
            
            if text_documents:
                self.vector_store.add_documents(text_documents)
                print(f"✓ {len(text_documents)}개 텍스트 청크 추가 완료")
            
            # 3. 이미지 추출 및 카운트
            print("\n[3/4] 이미지 처리 중...")
            all_images = []
            for doc in processed_docs:
                all_images.extend(doc['images'])
            
            stats['total_images'] = len(all_images)
            print(f"✓ {stats['total_images']}개 이미지 추출 완료")
            
            # 4. OCR 처리 (옵션)
            if enable_ocr and all_images:
                print("\n[4/4] OCR 처리 중...")
                self.initialize_ocr(gpu=gpu)
                
                ocr_results = self.ocr_processor.process_images_batch(all_images)
                
                # OCR 텍스트를 벡터 DB에 추가
                ocr_documents = []
                for image_path, ocr_text in ocr_results.items():
                    if ocr_text:  # 텍스트가 있는 경우만
                        doc_id = f"ocr_{uuid.uuid4().hex[:8]}_{Path(image_path).stem}"
                        
                        # 원본 문서 정보 찾기
                        source_doc = Path(image_path).parent.name
                        
                        ocr_documents.append({
                            'id': doc_id,
                            'text': ocr_text,
                            'metadata': {
                                'file_name': f"{source_doc} (이미지)",
                                'image_path': image_path,
                                'type': 'ocr'
                            }
                        })
                
                stats['ocr_texts'] = len(ocr_documents)
                
                if ocr_documents:
                    self.vector_store.add_documents(ocr_documents)
                    print(f"✓ {len(ocr_documents)}개 OCR 텍스트 추가 완료")
            else:
                print("\n[4/4] OCR 처리 건너뜀")
            
            # 완료 메시지
            print("\n" + "="*60)
            print("인덱싱 완료!")
            print(f"  - 처리된 문서: {stats['total_docs']}개")
            print(f"  - 텍스트 청크: {stats['total_chunks']}개")
            print(f"  - 추출된 이미지: {stats['total_images']}개")
            print(f"  - OCR 텍스트: {stats['ocr_texts']}개")
            print(f"  - 총 벡터 DB 항목: {self.vector_store.get_collection_count()}개")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"\n❌ 인덱싱 오류: {str(e)}")
            stats['status'] = 'error'
            import traceback
            traceback.print_exc()
        
        return stats
    
    def chat(self, query: str, top_k: int = 5) -> Dict:
        """
        사용자 질의에 대한 답변 생성
        
        Args:
            query: 사용자 질의
            top_k: 검색할 문서 수
        
        Returns:
            RAG 엔진 결과
        """
        # 벡터 DB가 비어있는지 확인
        if self.vector_store.get_collection_count() == 0:
            return {
                'query': query,
                'answer': "⚠ 인덱싱된 문서가 없습니다. 먼저 '문서 인덱싱 시작' 버튼을 클릭하여 문서를 인덱싱해주세요.",
                'sources': [],
                'retrieved_docs': []
            }
        
        # RAG 실행
        return self.rag_engine.query(query, top_k=top_k)
    
    def get_stats(self) -> Dict:
        """현재 시스템 통계 반환"""
        doc_count = len(list(self.documents_dir.glob('*.docx')))
        vector_count = self.vector_store.get_collection_count()
        
        return {
            'document_count': doc_count,
            'vector_count': vector_count,
            'documents_dir': str(self.documents_dir),
            'db_path': self.db_path
        }
    
    def reset_database(self):
        """벡터 DB 초기화"""
        self.vector_store.reset_collection()


if __name__ == "__main__":
    # 테스트 코드
    from dotenv import load_dotenv
    
    load_dotenv()
    
    data_dir = "../data"
    db_path = os.getenv('DB_BASE_PATH', './vectordb')
    
    chatbot = RAGChatbot(data_dir, db_path)
    
    # 통계 확인
    stats = chatbot.get_stats()
    print(f"문서 수: {stats['document_count']}")
    print(f"벡터 수: {stats['vector_count']}")
