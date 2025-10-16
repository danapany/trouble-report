"""
OCR 처리 모듈
- 이미지에서 텍스트 추출 (한글, 영문)
- EasyOCR 사용
"""

import easyocr
from typing import List, Dict
from pathlib import Path
import numpy as np
from PIL import Image


class OCRProcessor:
    def __init__(self, languages: List[str] = ['ko', 'en'], gpu: bool = False):
        """
        Args:
            languages: OCR 인식 언어 리스트
            gpu: GPU 사용 여부
        """
        self.languages = languages
        self.gpu = gpu
        self.reader = None
        self._initialize_reader()
    
    def _initialize_reader(self):
        """EasyOCR Reader 초기화"""
        try:
            print(f"EasyOCR 초기화 중... (언어: {', '.join(self.languages)}, GPU: {self.gpu})")
            self.reader = easyocr.Reader(self.languages, gpu=self.gpu)
            print("EasyOCR 초기화 완료")
        except Exception as e:
            print(f"EasyOCR 초기화 오류: {str(e)}")
            raise
    
    def extract_text_from_image(self, image_path: str, detail: int = 0) -> str:
        """
        이미지에서 텍스트 추출
        
        Args:
            image_path: 이미지 파일 경로
            detail: OCR 상세 레벨 (0: text only, 1: text + confidence, 2: full detail)
        
        Returns:
            추출된 텍스트
        """
        try:
            # 이미지 로드
            image = Image.open(image_path)
            image_np = np.array(image)
            
            # OCR 수행
            results = self.reader.readtext(image_np, detail=detail)
            
            # 텍스트만 추출
            if detail == 0:
                text = ' '.join(results)
            else:
                # detail=1 or 2: [(bbox, text, confidence), ...]
                text = ' '.join([item[1] for item in results])
            
            return text.strip()
        except Exception as e:
            print(f"OCR 처리 오류 ({image_path}): {str(e)}")
            return ""
    
    def process_images_batch(self, image_paths: List[str]) -> Dict[str, str]:
        """
        여러 이미지를 배치로 처리
        
        Args:
            image_paths: 이미지 파일 경로 리스트
        
        Returns:
            {image_path: extracted_text}
        """
        results = {}
        
        for image_path in image_paths:
            try:
                text = self.extract_text_from_image(image_path)
                results[image_path] = text
                
                if text:
                    print(f"OCR 완료: {Path(image_path).name} - {len(text)} 문자 추출")
                else:
                    print(f"OCR 결과 없음: {Path(image_path).name}")
            except Exception as e:
                print(f"이미지 처리 오류 ({image_path}): {str(e)}")
                results[image_path] = ""
                continue
        
        return results
    
    def extract_text_with_confidence(self, image_path: str, min_confidence: float = 0.5) -> List[Dict]:
        """
        이미지에서 텍스트를 신뢰도와 함께 추출
        
        Args:
            image_path: 이미지 파일 경로
            min_confidence: 최소 신뢰도 임계값
        
        Returns:
            [{'text': str, 'confidence': float}, ...]
        """
        try:
            image = Image.open(image_path)
            image_np = np.array(image)
            
            # OCR 수행 (detail=1: bbox + text + confidence)
            results = self.reader.readtext(image_np, detail=1)
            
            # 신뢰도 필터링
            filtered_results = []
            for bbox, text, confidence in results:
                if confidence >= min_confidence:
                    filtered_results.append({
                        'text': text,
                        'confidence': confidence
                    })
            
            return filtered_results
        except Exception as e:
            print(f"OCR 처리 오류 ({image_path}): {str(e)}")
            return []


if __name__ == "__main__":
    # 테스트 코드
    ocr = OCRProcessor(gpu=False)
    
    # 단일 이미지 테스트
    test_image = "../data/images/test_image.png"
    import os
    if os.path.exists(test_image):
        text = ocr.extract_text_from_image(test_image)
        print(f"추출된 텍스트: {text}")
