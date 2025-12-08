"""
ChromaDB Vector Database RAG 적합성 분석 스크립트
- Collection 정보 조회
- 저장된 문서 통계
- RAG 품질 검증 (문서 길이, 중복, 의미 분포 등)
- 샘플 데이터 확인
"""
import sys
import subprocess
import logging
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

logger.info("="*80)
logger.info("📊 Vector DB 분석 시작")
logger.info("="*80)

# 1. 패키지 설치
logger.info("📦 패키지 설치 중...")
packages = ["chromadb", "sentence-transformers"]
for pkg in packages:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    except Exception as e:
        logger.error(f"패키지 설치 실패: {pkg} - {e}")
logger.info("✅ 패키지 설치 완료\n")

import chromadb
import json


def _get_package_version(package_module, package_name):
    """Return an installed package version for logging/debugging."""
    module_version = getattr(package_module, "__version__", None)
    if module_version:
        return module_version

    try:
        from importlib import metadata as importlib_metadata  # py3.8+
    except ImportError:
        importlib_metadata = None

    if importlib_metadata is None:  # pragma: no cover - best effort fallback
        try:
            import importlib_metadata as importlib_metadata  # type: ignore
        except ImportError:
            return "unknown"

    try:
        return importlib_metadata.version(package_name)
    except Exception:
        return "unknown"


chromadb_version = _get_package_version(chromadb, "chromadb")
logger.info(f"🧬 ChromaDB 버전: {chromadb_version}")

# 2. Vector DB 설정 (실제 경로로 수정)ㅁ
VECTORDB_PATH = "/data/member/jks/redmine_RAG/vectordb/chroma_db_v0.1.2"
COLLECTION_NAME = "redmine_issues_raw_v2"

logger.info(f"📂 Vector DB 경로: {VECTORDB_PATH}")
logger.info(f"📚 Collection: {COLLECTION_NAME}\n")

# 2-1. 디렉토리 구조 분석
logger.info("="*80)
logger.info("📁 Vector DB 디렉토리 구조")
logger.info("="*80)

import os

if os.path.exists(VECTORDB_PATH):
    logger.info(f"경로: {VECTORDB_PATH}\n")

    # 전체 크기 계산
    total_size = 0
    file_count = 0
    dir_count = 0

    for root, dirs, files in os.walk(VECTORDB_PATH):
        dir_count += len(dirs)
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                total_size += size
                file_count += 1

    logger.info(f"총 크기: {total_size / (1024*1024):.2f} MB")
    logger.info(f"파일 개수: {file_count}개")
    logger.info(f"디렉토리 개수: {dir_count}개\n")

    # 주요 파일/디렉토리 상세
    logger.info("주요 구성 요소:")
    for item in sorted(os.listdir(VECTORDB_PATH)):
        item_path = os.path.join(VECTORDB_PATH, item)
        if os.path.isfile(item_path):
            size = os.path.getsize(item_path)
            logger.info(f"  📄 {item} ({size / (1024*1024):.2f} MB)")
        elif os.path.isdir(item_path):
            # 서브디렉토리 파일 개수
            sub_files = []
            for root, dirs, files in os.walk(item_path):
                sub_files.extend(files)
            logger.info(f"  📂 {item}/ ({len(sub_files)}개 파일)")

            # 서브디렉토리 내용 표시
            for sub_item in sorted(os.listdir(item_path))[:5]:  # 최대 5개만
                sub_path = os.path.join(item_path, sub_item)
                if os.path.isfile(sub_path):
                    size = os.path.getsize(sub_path)
                    logger.info(f"      └─ {sub_item} ({size / (1024*1024):.2f} MB)")

    logger.info("")
else:
    logger.error(f"경로가 존재하지 않습니다: {VECTORDB_PATH}\n")

try:
    # 3. ChromaDB 연결
    logger.info("="*80)
    logger.info("🔗 ChromaDB 연결")
    logger.info("="*80)
    client = chromadb.PersistentClient(path=VECTORDB_PATH)
    logger.info("✅ 연결 성공\n")

    # 4. 전체 Collection 목록
    logger.info("="*80)
    logger.info("📚 전체 Collections")
    logger.info("="*80)
    collections = client.list_collections()
    if collections:
        for idx, coll in enumerate(collections, 1):
            logger.info(f"{idx}. {coll.name}")
    else:
        logger.info("Collection이 없습니다.")
    logger.info("")

    # 5. 특정 Collection 상세 정보
    logger.info("="*80)
    logger.info(f"📊 Collection '{COLLECTION_NAME}' 상세 정보")
    logger.info("="*80)

    try:
        collection = client.get_collection(name=COLLECTION_NAME)

        # Collection 메타데이터
        collection_metadata = collection.metadata
        logger.info(f"Collection ID: {collection.id}")
        logger.info(f"Collection 이름: {collection.name}")
        if collection_metadata:
            logger.info(f"메타데이터: {collection_metadata}")

        # 전체 문서 수
        count = collection.count()
        logger.info(f"총 문서 수: {count:,}개\n")

        if count > 0:
            # 6. 샘플 데이터 조회 (처음 5개)
            logger.info("="*80)
            logger.info("📄 샘플 데이터 (처음 5개)")
            logger.info("="*80)

            results = collection.get(
                limit=5,
                include=["documents", "metadatas", "embeddings"]
            )

            # Embedding 차원 확인 (첫 번째 것만)
            emb_dim = 0
            if results.get('embeddings') is not None and len(results['embeddings']) > 0:
                import numpy as np
                emb_array = np.array(results['embeddings'][0])
                emb_dim = len(emb_array)
                logger.info(f"✅ Embedding 차원: {emb_dim}\n")

            # 데이터 스키마 분석
            logger.info("📋 데이터 스키마 구조:")
            logger.info(f"  - IDs: {type(results['ids']).__name__} (총 {len(results['ids'])}개)")
            logger.info(f"  - Documents: {type(results['documents']).__name__} (총 {len(results['documents'])}개)")
            logger.info(f"  - Metadatas: {type(results['metadatas']).__name__} (총 {len(results['metadatas'])}개)")
            logger.info(f"  - Embeddings: {type(results['embeddings']).__name__} (총 {len(results['embeddings'])}개)")

            # 메타데이터 필드 확인
            if results['metadatas'] and len(results['metadatas']) > 0:
                sample_meta = results['metadatas'][0]
                logger.info(f"\n  메타데이터 필드:")
                for key in sorted(sample_meta.keys()):
                    logger.info(f"    - {key}: {type(sample_meta[key]).__name__}")
            logger.info("")

            for idx, (doc_id, doc, meta) in enumerate(zip(
                results['ids'],
                results['documents'],
                results['metadatas']
            ), 1):
                logger.info(f"\n[{idx}] ID: {doc_id}")
                logger.info(f"제목: {meta.get('subject', 'N/A')}")
                logger.info(f"생성일: {meta.get('created_on', 'N/A')}")
                logger.info(f"수정일: {meta.get('updated_on', 'N/A')}")
                logger.info(f"문서 길이: {len(doc)} 문자")
                logger.info(f"내용 미리보기:\n{doc[:200]}...")
                logger.info("-" * 80)

            # 7. 메타데이터 통계
            logger.info("\n" + "="*80)
            logger.info("📈 메타데이터 통계")
            logger.info("="*80)

            # 전체 데이터 조회 (메타데이터만)
            all_results = collection.get(
                include=["metadatas"]
            )

            # 생성일 기준 통계
            created_dates = [meta.get('created_on', '')[:10] for meta in all_results['metadatas'] if meta.get('created_on')]
            updated_dates = [meta.get('updated_on', '')[:10] for meta in all_results['metadatas'] if meta.get('updated_on')]

            if created_dates:
                logger.info(f"가장 오래된 이슈: {min(created_dates)}")
                logger.info(f"가장 최근 이슈: {max(created_dates)}")

            if updated_dates:
                logger.info(f"가장 최근 업데이트: {max(updated_dates)}")

            # 제목 길이 통계
            subjects = [meta.get('subject', '') for meta in all_results['metadatas']]
            subject_lengths = [len(s) for s in subjects if s]
            if subject_lengths:
                logger.info(f"\n제목 길이 통계:")
                logger.info(f"  평균: {sum(subject_lengths) / len(subject_lengths):.1f} 문자")
                logger.info(f"  최소: {min(subject_lengths)} 문자")
                logger.info(f"  최대: {max(subject_lengths)} 문자")

            # 8. RAG 품질 검증
            logger.info("\n" + "="*80)
            logger.info("🔍 RAG 품질 검증")
            logger.info("="*80)

            # 8-1. 문서 길이 분포 분석
            all_docs = collection.get(include=["documents"])
            doc_lengths = [len(doc) for doc in all_docs['documents']]

            # 빈 문서 개수
            empty_docs = sum(1 for length in doc_lengths if length == 0)
            # 너무 짧은 문서 (100자 미만)
            too_short = sum(1 for length in doc_lengths if 0 < length < 100)
            # 적정 길이 (100-5000자)
            optimal = sum(1 for length in doc_lengths if 100 <= length <= 5000)
            # 긴 문서 (5000자 이상)
            long_docs = sum(1 for length in doc_lengths if length > 5000)

            logger.info(f"\n📏 문서 길이 분포:")
            logger.info(f"  빈 문서 (0자): {empty_docs}개 ({empty_docs/len(doc_lengths)*100:.1f}%)")
            logger.info(f"  너무 짧음 (1-99자): {too_short}개 ({too_short/len(doc_lengths)*100:.1f}%)")
            logger.info(f"  적정 길이 (100-5000자): {optimal}개 ({optimal/len(doc_lengths)*100:.1f}%) ✅")
            logger.info(f"  긴 문서 (5000자+): {long_docs}개 ({long_docs/len(doc_lengths)*100:.1f}%)")

            if empty_docs > 0:
                logger.warning(f"⚠️  빈 문서가 {empty_docs}개 발견되었습니다. RAG 품질에 영향을 줄 수 있습니다.")

            if too_short > len(doc_lengths) * 0.1:
                logger.warning(f"⚠️  너무 짧은 문서가 {too_short}개({too_short/len(doc_lengths)*100:.1f}%)입니다. 의미 있는 정보가 부족할 수 있습니다.")

            # 8-2. 중복 제목 확인
            logger.info(f"\n📋 제목 중복 분석:")
            subjects = [meta.get('subject', '') for meta in all_results['metadatas']]
            unique_subjects = len(set(subjects))
            duplicate_ratio = (1 - unique_subjects / len(subjects)) * 100
            logger.info(f"  전체 제목: {len(subjects)}개")
            logger.info(f"  고유 제목: {unique_subjects}개")
            logger.info(f"  중복률: {duplicate_ratio:.1f}%")

            if duplicate_ratio > 5:
                logger.warning(f"⚠️  중복 제목이 {duplicate_ratio:.1f}%입니다. 중복 문서 확인이 필요합니다.")

            # 8-3. 검색 테스트
            logger.info("\n" + "="*80)
            logger.info("🔍 유사도 검색 테스트")
            logger.info("="*80)

            test_queries = [
                "버그 수정",
                "모델 학습",
                "데이터셋 전처리"
            ]

            try:
                from sentence_transformers import SentenceTransformer

                logger.info("임베딩 모델 로드 중...")
                model = SentenceTransformer("intfloat/multilingual-e5-large")

                for query_idx, test_query in enumerate(test_queries, 1):
                    logger.info(f"\n[테스트 {query_idx}] 검색어: '{test_query}'")

                    query_embedding = model.encode([test_query])[0]

                    search_results = collection.query(
                        query_embeddings=[query_embedding.tolist()],
                        n_results=3,
                        include=["documents", "metadatas", "distances"]
                    )

                    for idx, (doc_id, doc, meta, dist) in enumerate(zip(
                        search_results['ids'][0],
                        search_results['documents'][0],
                        search_results['metadatas'][0],
                        search_results['distances'][0]
                    ), 1):
                        logger.info(f"  [{idx}] 거리: {dist:.4f} | ID: {doc_id}")
                        logger.info(f"      제목: {meta.get('subject', 'N/A')}")
                        logger.info(f"      내용: {doc[:100]}...")

                    # 검색 품질 평가
                    avg_distance = sum(search_results['distances'][0]) / len(search_results['distances'][0])
                    if avg_distance > 1.0:
                        logger.warning(f"      ⚠️  평균 거리 {avg_distance:.4f} - 검색 품질이 낮을 수 있습니다.")
                    logger.info("")

            except Exception as e:
                logger.warning(f"검색 테스트 실패: {e}")
                import traceback
                traceback.print_exc()

            # 8-4. RAG 종합 평가
            logger.info("\n" + "="*80)
            logger.info("📊 RAG 적합성 종합 평가")
            logger.info("="*80)

            score = 0
            max_score = 5

            # 1. 문서 개수
            if count >= 500:
                logger.info("✅ 문서 개수: 충분함 (500개 이상)")
                score += 1
            else:
                logger.warning(f"⚠️  문서 개수: 부족함 ({count}개 < 500개)")

            # 2. 빈 문서 비율
            empty_ratio = empty_docs / len(doc_lengths) * 100
            if empty_ratio < 1:
                logger.info(f"✅ 빈 문서 비율: 양호함 ({empty_ratio:.1f}% < 1%)")
                score += 1
            else:
                logger.warning(f"⚠️  빈 문서 비율: 높음 ({empty_ratio:.1f}% >= 1%)")

            # 3. 적정 길이 문서 비율
            optimal_ratio = optimal / len(doc_lengths) * 100
            if optimal_ratio >= 60:
                logger.info(f"✅ 적정 길이 문서: 충분함 ({optimal_ratio:.1f}% >= 60%)")
                score += 1
            else:
                logger.warning(f"⚠️  적정 길이 문서: 부족함 ({optimal_ratio:.1f}% < 60%)")

            # 4. 중복 비율
            if duplicate_ratio < 5:
                logger.info(f"✅ 중복 비율: 낮음 ({duplicate_ratio:.1f}% < 5%)")
                score += 1
            else:
                logger.warning(f"⚠️  중복 비율: 높음 ({duplicate_ratio:.1f}% >= 5%)")

            # 5. Embedding 차원
            if emb_dim == 1024:
                logger.info(f"✅ Embedding 차원: 정상 (1024)")
                score += 1
            else:
                logger.warning(f"⚠️  Embedding 차원: 비정상 ({emb_dim})")

            logger.info(f"\n🎯 RAG 적합성 점수: {score}/{max_score}")

            if score >= 4:
                logger.info("✅ RAG에 적합한 데이터입니다!")
            elif score >= 3:
                logger.warning("⚠️  일부 개선이 필요합니다.")
            else:
                logger.error("❌ RAG 품질 개선이 시급합니다.")

        else:
            logger.info("Collection이 비어있습니다.")

    except Exception as e:
        logger.error(f"Collection '{COLLECTION_NAME}' 조회 실패: {e}")
        logger.info("\n사용 가능한 Collections:")
        for coll in collections:
            logger.info(f"  - {coll.name}")

    # 9. JSON 파일 분석
    logger.info("\n" + "="*80)
    logger.info("📁 JSON 파일 분석")
    logger.info("="*80)

    json_path = "/data/member/jks/redmine_RAG/vectordb/redmine_data/redmine_raw_issues.json"
    try:
        import os
        if os.path.exists(json_path):
            file_size = os.path.getsize(json_path)
            logger.info(f"파일 경로: {json_path}")
            logger.info(f"파일 크기: {file_size / 1024 / 1024:.2f} MB")

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            logger.info(f"수집 시각: {data.get('collected_at', 'N/A')}")
            logger.info(f"총 이슈 수: {data.get('total_issues', 0):,}개")

            if data.get('issues'):
                issues = data['issues']
                text_lengths = [len(issue.get('raw_text', '')) for issue in issues]

                logger.info(f"\n이슈 텍스트 길이 통계:")
                logger.info(f"  평균: {sum(text_lengths) / len(text_lengths):.0f} 문자")
                logger.info(f"  최소: {min(text_lengths)} 문자")
                logger.info(f"  최대: {max(text_lengths):,} 문자")

                # 가장 긴 이슈
                max_idx = text_lengths.index(max(text_lengths))
                longest_issue = issues[max_idx]
                logger.info(f"\n가장 긴 이슈:")
                logger.info(f"  ID: {longest_issue['id']}")
                logger.info(f"  제목: {longest_issue['subject']}")
                logger.info(f"  길이: {text_lengths[max_idx]:,} 문자")
        else:
            logger.info(f"JSON 파일이 없습니다: {json_path}")

    except Exception as e:
        logger.error(f"JSON 파일 분석 실패: {e}")

except Exception as e:
    logger.error(f"Vector DB 연결 실패: {e}")
    import traceback
    traceback.print_exc()

logger.info("\n" + "="*80)
logger.info("✅ 분석 완료")
logger.info("="*80)
