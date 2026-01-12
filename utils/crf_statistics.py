"""CRF 데이터 통계 계산 모듈"""
import logging
import re
from collections import Counter
from datetime import datetime

logger = logging.getLogger(__name__)


class CRFStatisticsMixin:
    """CRF 데이터 통계 계산을 위한 Mixin 클래스"""

    def calculate_crf_statistics(self, documents: list, metadatas: list, hospital_code: str = None) -> dict:
        """
        CRF 문서 리스트에서 통계 계산

        Args:
            documents: ChromaDB에서 가져온 문서 리스트
            metadatas: 문서 메타데이터 리스트
            hospital_code: 병원 코드 (예: "02")

        Returns:
            통계 데이터 딕셔너리
        """
        logger.info(f"📊 통계 계산 시작: {len(documents)}개 문서")

        stats = {
            'total_patients': 0,
            'total_documents': len(documents),
            'hospital_code': hospital_code,
            'hospital_name': self._get_hospital_name(hospital_code) if hospital_code else "전체 병원",
            'age_stats': {},
            'tumor_size_stats': {},
            'er_stats': {},
            'pr_stats': {},
            'her2_stats': {},
            'biomarker_combinations': {},
            'stage_distribution': {},
            'ng_distribution': {},  # Nuclear Grade
            'hg_distribution': {},  # Histologic Grade
            'ki67_stats': {},  # Ki-67 LI
            'ki67_thresholds': [],  # Ki-67 임계값별 통계
            'lymph_node_stats': {},  # 림프절 전이
            'tnm_stats': {},  # T/N/M category
            'histologic_type_distribution': {},  # 조직학적 타입
            'surgery_type_distribution': {},  # 수술 방법
            'recurrence_stats': {},  # 재발 여부
            'survival_stats': {},  # 생존 여부
            'hospital_counts': {},  # 병원별 건수
            'surgery_year_distribution': {},  # 수술 연도별 건수
            'stage_ng_distribution': {},  # Stage x NG
            'stage_hg_distribution': {},  # Stage x HG
            # 새로 추가된 통계
            'tumor_location_distribution': {},  # 암의 위치 (Rt./Lt./Both)
            'tumor_number_distribution': {},  # 암의 개수 (single/multiple)
            'her2_ihc_distribution': {},  # HER2 IHC 등급
            'dcis_lcis_distribution': {},  # DCIS/LCIS 여부
            'mitotic_rate_distribution': {},  # HG score 3 (Mitotic Rate)
            'er_allred_stats': {},  # ER Allred score
            'pr_allred_stats': {},  # PR Allred score
            'adjuvant_endocrine_stats': {},  # 보조 호르몬 치료
            'adjuvant_rtx_stats': {},  # 보조 방사선 치료
            'neoadjuvant_ctx_stats': {},  # 수술 전 항암 치료
            'neoadjuvant_response_distribution': {},  # 수술 전 항암 반응
            'followup_period_stats': {},  # 추적 관찰 기간
            'unique_records': set()
        }

        # 데이터 파싱
        ages = []
        tumor_sizes = []
        er_positive = 0
        er_total = 0
        pr_positive = 0
        pr_total = 0
        her2_positive = 0
        her2_total = 0
        stages = []
        ng_grades = []  # Nuclear Grade
        hg_grades = []  # Histologic Grade
        ki67_values = []  # Ki-67
        lymph_node_positive = 0  # 림프절 전이 양성
        lymph_node_total = 0
        lymph_node_counts = []  # 전이 림프절 개수
        t_categories = []
        n_categories = []
        m_categories = []
        histologic_types = []
        surgery_types = []
        axillary_recurrence = 0
        surgery_site_recurrence = 0
        distant_metastasis = 0
        recurrence_total = 0
        alive = 0
        dead_from_disease = 0
        dead_from_other = 0
        survival_total = 0
        # 새로 추가된 변수들
        tumor_locations = []  # 암의 위치
        tumor_numbers = []  # 암의 개수
        her2_ihc_grades = []  # HER2 IHC 등급
        dcis_lcis_statuses = []  # DCIS/LCIS 여부
        mitotic_rates = []  # Mitotic Rate
        er_allred_scores = []  # ER Allred score
        pr_allred_scores = []  # PR Allred score
        adjuvant_endocrine = []  # 보조 호르몬 치료
        adjuvant_rtx = []  # 보조 방사선 치료
        neoadjuvant_ctx = []  # 수술 전 항암 치료
        neoadjuvant_response = []  # 수술 전 항암 반응
        surgery_dates = []  # 수술 날짜
        followup_dates = []  # 추적 관찰 날짜
        # 카운터
        hospital_counter = Counter()
        surgery_year_counter = Counter()
        stage_ng_counter = Counter()
        stage_hg_counter = Counter()
        biomarker_combo_counter = Counter()
        KI67_THRESHOLDS = [10, 20, 30]  # 필요시 확장

        for doc, meta in zip(documents, metadatas):
            # Record ID 수집 (고유 환자 수)
            record_id = meta.get('record_id')
            if record_id:
                stats['unique_records'].add(record_id)

            hospital_code = meta.get('hospital')
            if hospital_code:
                hospital_counter[hospital_code] += 1

            # 수술 연도 추출 (메타데이터 필드 우선)
            surgery_date = meta.get("수술연월일") or meta.get("surgery_date")
            if surgery_date:
                try:
                    year = int(str(surgery_date)[:4])
                    surgery_year_counter[year] += 1
                except Exception:
                    pass

            stage_val = None
            ng_val = None
            hg_val = None
            er_val = None
            pr_val = None
            her2_val = None

            # 진단 시 나이 추출 (형식: "나이 (진단시): 63")
            age_match = re.search(r'나이\s*\(진단시\)\s*:\s*(\d+)', doc)
            if age_match:
                try:
                    ages.append(int(age_match.group(1)))
                except ValueError:
                    pass

            # 암 크기 (장경) 추출 (형식: "암 size (mm)_장경: 19")
            size_match = re.search(r'암\s*size\s*\(mm\)_장경\s*:\s*(\d+(?:\.\d+)?)', doc)
            if size_match:
                try:
                    tumor_sizes.append(float(size_match.group(1)))
                except ValueError:
                    pass

            # ER 상태 (형식: "ER (-/+): 1" → 1은 양성, 0은 음성)
            er_match = re.search(r'ER\s*\(-/\+\)\s*:\s*([01])', doc)
            if er_match:
                er_total += 1
                er_val = er_match.group(1)
                if er_match.group(1) == '1':
                    er_positive += 1

            # PR 상태 (형식: "PR (-/+): 1" → 1은 양성, 0은 음성)
            pr_match = re.search(r'PR\s*\(-/\+\)\s*:\s*([01])', doc)
            if pr_match:
                pr_total += 1
                pr_val = pr_match.group(1)
                if pr_match.group(1) == '1':
                    pr_positive += 1

            # HER2 상태 (형식: "HER2 (-/+): 0" → 1은 양성, 0은 음성)
            her2_match = re.search(r'HER2\s*\(-/\+\)\s*:\s*([01])', doc)
            if her2_match:
                her2_total += 1
                her2_val = her2_match.group(1)
                if her2_match.group(1) == '1':
                    her2_positive += 1
            # 바이오마커 조합 카운트 (ER/PR/HER2 모두 값이 있는 경우)
            if er_val is not None and pr_val is not None and her2_val is not None:
                er_pos = er_val == '1'
                pr_pos = pr_val == '1'
                her2_pos = her2_val == '1'
                biomarker_combo_counter['her2_positive'] += int(her2_pos)
                biomarker_combo_counter['er_positive'] += int(er_pos)
                biomarker_combo_counter['pr_positive'] += int(pr_pos)
                biomarker_combo_counter['er_pr_positive'] += int(er_pos and pr_pos)
                biomarker_combo_counter['hr_positive_her2_negative'] += int((er_pos or pr_pos) and not her2_pos)
                biomarker_combo_counter['triple_negative'] += int((not er_pos) and (not pr_pos) and (not her2_pos))

            # AJCC Stage (형식: "AJCC stage (8판): 1" → 숫자를 Stage I, Stage II 등으로 변환)
            stage_match = re.search(r'AJCC\s*stage\s*\(8판\)\s*:\s*(\d+)', doc, re.IGNORECASE)
            if stage_match:
                stage_num = stage_match.group(1)
                # 1 → Stage I, 2 → Stage II, 등
                stage_map = {'1': 'Stage I', '2': 'Stage II', '3': 'Stage III', '4': 'Stage IV'}
                stage_name = stage_map.get(stage_num, f'Stage {stage_num}')
                stages.append(stage_name)
                stage_val = stage_name

            # Nuclear Grade (형식: "NG (1/2/3): 2")
            ng_match = re.search(r'NG\s*\(1/2/3\)\s*:\s*([123])', doc)
            if ng_match:
                ng_grade = f'Grade {ng_match.group(1)}'
                ng_grades.append(ng_grade)
                ng_val = ng_grade

            # Histologic Grade (형식: "HG (1/2/3/4): 1")
            hg_match = re.search(r'HG\s*\(1/2/3/4\)\s*:\s*([1234])', doc)
            if hg_match:
                hg_grade = f'Grade {hg_match.group(1)}'
                hg_grades.append(hg_grade)
                hg_val = hg_grade
            # Stage x Grade 교차 카운트
            if stage_val and ng_val:
                stage_ng_counter[(stage_val, ng_val)] += 1
            if stage_val and hg_val:
                stage_hg_counter[(stage_val, hg_val)] += 1

            # Ki-67 (형식: "KI-67 LI (%): 12")
            ki67_match = re.search(r'KI-67\s*LI\s*\(%\)\s*:\s*(\d+(?:\.\d+)?)', doc, re.IGNORECASE)
            if ki67_match:
                try:
                    ki67_values.append(float(ki67_match.group(1)))
                except ValueError:
                    pass

            # 림프절 전이 여부 (형식: "림프절 전이여부_수술당시 (0: No, 1: Yes_SN, 2: Yes_nonSN, 3: Yes_SN+nonSN): 0")
            ln_match = re.search(r'림프절\s*전이여부_수술당시.*?:\s*([0123])', doc)
            if ln_match:
                lymph_node_total += 1
                if ln_match.group(1) != '0':  # 0이 아니면 전이 있음
                    lymph_node_positive += 1

            # 전이 림프절 개수 (형식: "전이 림프절 개수_수술당시: 0")
            ln_count_match = re.search(r'전이\s*림프절\s*개수_수술당시\s*:\s*(\d+)', doc)
            if ln_count_match:
                try:
                    count = int(ln_count_match.group(1))
                    if count > 0:
                        lymph_node_counts.append(count)
                except ValueError:
                    pass

            # T category (형식: "T category: 1")
            t_match = re.search(r'T\s*category\s*:\s*(\d+)', doc)
            if t_match:
                t_categories.append(f'T{t_match.group(1)}')

            # N category (형식: "N category: 0")
            n_match = re.search(r'N\s*category\s*:\s*(\d+)', doc)
            if n_match:
                n_categories.append(f'N{n_match.group(1)}')

            # M category (형식: "M category (수술당시 원격전이여부_0: pM0, 1: pM1): 0")
            m_match = re.search(r'M\s*category.*?:\s*([01])', doc)
            if m_match:
                m_categories.append('M1' if m_match.group(1) == '1' else 'M0')

            # 조직학적 타입 (형식: "진단명 (histologic type: ductal/ lobular/ mucinous/ other): : 1")
            histologic_match = re.search(r'진단명\s*\(histologic\s*type.*?\)\s*:\s*:\s*([1234])', doc)
            if histologic_match:
                type_map = {'1': 'Ductal', '2': 'Lobular', '3': 'Mucinous', '4': 'Other'}
                histologic_types.append(type_map.get(histologic_match.group(1), f'Type {histologic_match.group(1)}'))

            # 수술 방법 (형식: "수술명 (partial/total): 2")
            surgery_match = re.search(r'수술명\s*\(partial/total\)\s*:\s*([12])', doc)
            if surgery_match:
                surgery_types.append('Total mastectomy' if surgery_match.group(1) == '2' else 'Partial mastectomy')

            # 재발 여부 - Axillary LN 재발 (형식: "Axillary LN 재발 여부 (0/1): 0")
            axillary_rec_match = re.search(r'Axillary\s*LN\s*재발\s*여부\s*\(0/1\)\s*:\s*([01])', doc)
            if axillary_rec_match:
                recurrence_total += 1
                if axillary_rec_match.group(1) == '1':
                    axillary_recurrence += 1

            # 재발 여부 - 수술부위 재발 (형식: "수술부위 재발여부 (0/1): 0")
            surgery_rec_match = re.search(r'수술부위\s*재발여부\s*\(0/1\)\s*:\s*([01])', doc)
            if surgery_rec_match:
                if surgery_rec_match.group(1) == '1':
                    surgery_site_recurrence += 1

            # 재발 여부 - 원격 전이 (형식: "다른 장기로 전이 여부 (0/1): 0")
            distant_match = re.search(r'다른\s*장기로\s*전이\s*여부\s*\(0/1\)\s*:\s*([01])', doc)
            if distant_match:
                if distant_match.group(1) == '1':
                    distant_metastasis += 1

            # 생존 여부 (형식: "이 질병으로 사망여부 (0:생존/ 1:사망/ 2:다른이유로사망): 0")
            survival_match = re.search(r'이\s*질병으로\s*사망여부.*?:\s*([012])', doc)
            if survival_match:
                survival_total += 1
                status = survival_match.group(1)
                if status == '0':
                    alive += 1
                elif status == '1':
                    dead_from_disease += 1
                elif status == '2':
                    dead_from_other += 1

            # ===== 새로 추가된 19개 필드 추출 =====

            # 암의 위치 (형식: "암의 위치 (Rt./Lt./Both): 2")
            location_match = re.search(r'암의\s*위치\s*\(Rt\./Lt\./Both\)\s*:\s*([123])', doc)
            if location_match:
                location_map = {'1': 'Right', '2': 'Left', '3': 'Both'}
                tumor_locations.append(location_map.get(location_match.group(1), location_match.group(1)))

            # 암의 개수 (형식: "암의 개수 (single/multiple): 1")
            number_match = re.search(r'암의\s*개수\s*\(single/multiple\)\s*:\s*([12])', doc)
            if number_match:
                tumor_numbers.append('Single' if number_match.group(1) == '1' else 'Multiple')

            # HER2 IHC (형식: "HER2_IHC (0/+1/ +2/ +3): 1")
            her2_ihc_match = re.search(r'HER2_IHC\s*\(0/\+1/\s*\+2/\s*\+3\)\s*:\s*([0123])', doc)
            if her2_ihc_match:
                ihc_map = {'0': 'IHC 0', '1': 'IHC 1+', '2': 'IHC 2+', '3': 'IHC 3+'}
                her2_ihc_grades.append(ihc_map.get(her2_ihc_match.group(1), her2_ihc_match.group(1)))

            # DCIS/LCIS 여부 (형식: "DCIS or LCIS 여부 (0: no DCIS/LCIS, 1: DCIS/LCIS present, EIC(-), 2: DCIS/LCIS present, EIC(+)), : 2")
            dcis_lcis_match = re.search(r'DCIS\s*or\s*LCIS\s*여부.*?:\s*([012])', doc)
            if dcis_lcis_match:
                dcis_map = {'0': 'No DCIS/LCIS', '1': 'DCIS/LCIS present, EIC(-)', '2': 'DCIS/LCIS present, EIC(+)'}
                dcis_lcis_statuses.append(dcis_map.get(dcis_lcis_match.group(1), dcis_lcis_match.group(1)))

            # Mitotic Rate (형식: "HG_score 3 (Mitotic Rate) (1/2/3/4): 4")
            mitotic_match = re.search(r'HG_score\s*3\s*\(Mitotic\s*Rate\)\s*\(1/2/3/4\)\s*:\s*([1234])', doc)
            if mitotic_match:
                mitotic_rates.append(f'Score {mitotic_match.group(1)}')

            # ER Allred score (형식: "ER (Allred score)" 다음 줄에 "스코어 계산 필요: 100")
            # 두 개의 "스코어 계산 필요"가 있으므로 ER과 PR을 구분해야 함
            # ER 섹션에서 첫 번째 스코어
            er_section = re.search(r'ER\s*\(-/\+\).*?(?=PR\s*\(-/\+\)|$)', doc, re.DOTALL)
            if er_section:
                er_allred_match = re.search(r'스코어\s*계산\s*필요\s*:\s*(\d+)', er_section.group(0))
                if er_allred_match:
                    try:
                        er_allred_scores.append(int(er_allred_match.group(1)))
                    except ValueError:
                        pass

            # PR Allred score (형식: PR 섹션의 "스코어 계산 필요: 50")
            pr_section = re.search(r'PR\s*\(-/\+\).*?(?=KI-67|HER2|$)', doc, re.DOTALL)
            if pr_section:
                pr_allred_match = re.search(r'스코어\s*계산\s*필요\s*:\s*(\d+)', pr_section.group(0))
                if pr_allred_match:
                    try:
                        pr_allred_scores.append(int(pr_allred_match.group(1)))
                    except ValueError:
                        pass

            # 보조 호르몬 치료 (형식: "adjuvant Endocrine/Hormonal Tx: 2")
            adj_endo_match = re.search(r'adjuvant\s*Endocrine/Hormonal\s*Tx\s*:\s*([012])', doc, re.IGNORECASE)
            if adj_endo_match:
                adjuvant_endocrine.append(adj_endo_match.group(1))

            # 보조 방사선 치료 (형식: "adjuvant RTx : 0")
            adj_rtx_match = re.search(r'adjuvant\s*RTx\s*:\s*([012])', doc, re.IGNORECASE)
            if adj_rtx_match:
                adjuvant_rtx.append(adj_rtx_match.group(1))

            # 수술 전 항암 치료 (형식: "neoadjuvantCTx (0:무, 1:유): 0")
            neoadj_ctx_match = re.search(r'neoadjuvantCTx\s*\(0:무,\s*1:유\)\s*:\s*([01])', doc, re.IGNORECASE)
            if neoadj_ctx_match:
                neoadjuvant_ctx.append('유' if neoadj_ctx_match.group(1) == '1' else '무')

            # 수술 전 항암 반응 (형식: "neoadjuvantCTx response_MP (1, 2, 3, 4, 5, 6)- 기준 다름-보정필요: 0")
            neoadj_response_match = re.search(r'neoadjuvantCTx\s*response_MP.*?:\s*([0-6])', doc, re.IGNORECASE)
            if neoadj_response_match:
                response = neoadj_response_match.group(1)
                if response != '0':  # 0이 아닌 경우만 기록
                    neoadjuvant_response.append(f'Response {response}')

            # 수술 날짜 (형식: "수술연월일: 2015-10-23 00:00:00")
            surgery_date_match = re.search(r'수술연월일\s*:\s*(\d{4}-\d{2}-\d{2})', doc)
            if surgery_date_match:
                try:
                    surgery_dates.append(datetime.strptime(surgery_date_match.group(1), "%Y-%m-%d").date())
                except Exception:
                    pass

            # 추적 관찰 날짜 (형식: "Last F/U 날짜 (연-월-일): 2020-09-25 00:00:00")
            followup_date_match = re.search(r'Last\s*F/U\s*날짜.*?:\s*(\d{4}-\d{2}-\d{2})', doc, re.IGNORECASE)
            if followup_date_match:
                try:
                    followup_dates.append(datetime.strptime(followup_date_match.group(1), "%Y-%m-%d").date())
                except Exception:
                    pass

        # 통계 계산
        stats['total_patients'] = len(stats['unique_records'])

        # 나이 통계
        if ages:
            stats['age_stats'] = {
                'mean': round(sum(ages) / len(ages), 1),
                'min': min(ages),
                'max': max(ages),
                'count': len(ages)
            }

        # 종양 크기 통계
        if tumor_sizes:
            stats['tumor_size_stats'] = {
                'mean': round(sum(tumor_sizes) / len(tumor_sizes), 1),
                'min': min(tumor_sizes),
                'max': max(tumor_sizes),
                'count': len(tumor_sizes)
            }

        # ER 통계
        if er_total > 0:
            stats['er_stats'] = {
                'positive': er_positive,
                'total': er_total,
                'percentage': round(er_positive / er_total * 100, 1)
            }

        # PR 통계
        if pr_total > 0:
            stats['pr_stats'] = {
                'positive': pr_positive,
                'total': pr_total,
                'percentage': round(pr_positive / pr_total * 100, 1)
            }

        # HER2 통계
        if her2_total > 0:
            stats['her2_stats'] = {
                'positive': her2_positive,
                'total': her2_total,
                'percentage': round(her2_positive / her2_total * 100, 1)
            }

        # Stage 분포
        if stages:
            stage_counter = Counter(stages)
            stats['stage_distribution'] = dict(stage_counter)

        # Nuclear Grade 분포
        if ng_grades:
            ng_counter = Counter(ng_grades)
            stats['ng_distribution'] = dict(ng_counter)

        # Histologic Grade 분포
        if hg_grades:
            hg_counter = Counter(hg_grades)
            stats['hg_distribution'] = dict(hg_counter)

        # Ki-67 통계
        if ki67_values:
            stats['ki67_stats'] = {
                'mean': round(sum(ki67_values) / len(ki67_values), 1),
                'min': min(ki67_values),
                'max': max(ki67_values),
                'count': len(ki67_values)
            }
            thresholds = []
            for th in KI67_THRESHOLDS:
                above_count = sum(1 for v in ki67_values if v >= th)
                thresholds.append({
                    'threshold': th,
                    'count': above_count,
                    'percentage': round(above_count / len(ki67_values) * 100, 1)
                })
            stats['ki67_thresholds'] = thresholds

        # 림프절 전이 통계
        if lymph_node_total > 0:
            stats['lymph_node_stats'] = {
                'positive': lymph_node_positive,
                'total': lymph_node_total,
                'percentage': round(lymph_node_positive / lymph_node_total * 100, 1)
            }
            if lymph_node_counts:
                stats['lymph_node_stats']['mean_count'] = round(sum(lymph_node_counts) / len(lymph_node_counts), 1)
                stats['lymph_node_stats']['max_count'] = max(lymph_node_counts)

        # TNM 분류 통계
        if t_categories or n_categories or m_categories:
            stats['tnm_stats'] = {}
            if t_categories:
                stats['tnm_stats']['T'] = dict(Counter(t_categories))
            if n_categories:
                stats['tnm_stats']['N'] = dict(Counter(n_categories))
            if m_categories:
                stats['tnm_stats']['M'] = dict(Counter(m_categories))

        # 조직학적 타입 분포
        if histologic_types:
            histologic_counter = Counter(histologic_types)
            stats['histologic_type_distribution'] = dict(histologic_counter)

        # 수술 방법 분포
        if surgery_types:
            surgery_counter = Counter(surgery_types)
            stats['surgery_type_distribution'] = dict(surgery_counter)

        # 재발 통계
        if recurrence_total > 0:
            total_recurrence = axillary_recurrence + surgery_site_recurrence + distant_metastasis
            stats['recurrence_stats'] = {
                'axillary_ln': axillary_recurrence,
                'surgery_site': surgery_site_recurrence,
                'distant_metastasis': distant_metastasis,
                'total_with_recurrence': total_recurrence,
                'total_patients': recurrence_total,
                'recurrence_rate': round(total_recurrence / recurrence_total * 100, 1) if total_recurrence > 0 else 0
            }

        # 생존 통계
        if survival_total > 0:
            stats['survival_stats'] = {
                'alive': alive,
                'dead_from_disease': dead_from_disease,
                'dead_from_other': dead_from_other,
                'total': survival_total,
                'survival_rate': round(alive / survival_total * 100, 1)
            }

        # 병원별 건수
        if hospital_counter:
            stats['hospital_counts'] = dict(hospital_counter)

        # 수술 연도 분포
        if surgery_year_counter:
            stats['surgery_year_distribution'] = dict(sorted(surgery_year_counter.items()))

        # 바이오마커 조합 통계
        if biomarker_combo_counter:
            total = len(stats['unique_records']) or len(documents)
            combos = {}
            for key, val in biomarker_combo_counter.items():
                combos[key] = {
                    'count': val,
                    'percentage': round(val / total * 100, 1) if total else 0
                }
            stats['biomarker_combinations'] = combos

        # Stage x NG/HG 교차
        if stage_ng_counter:
            stats['stage_ng_distribution'] = {f"{k[0]} | {k[1]}": v for k, v in stage_ng_counter.items()}
        if stage_hg_counter:
            stats['stage_hg_distribution'] = {f"{k[0]} | {k[1]}": v for k, v in stage_hg_counter.items()}

        # ===== 새로 추가된 필드 통계 =====

        # 암의 위치 분포
        if tumor_locations:
            stats['tumor_location_distribution'] = dict(Counter(tumor_locations))

        # 암의 개수 분포
        if tumor_numbers:
            stats['tumor_number_distribution'] = dict(Counter(tumor_numbers))

        # HER2 IHC 분포
        if her2_ihc_grades:
            stats['her2_ihc_distribution'] = dict(Counter(her2_ihc_grades))

        # DCIS/LCIS 분포
        if dcis_lcis_statuses:
            stats['dcis_lcis_distribution'] = dict(Counter(dcis_lcis_statuses))

        # Mitotic Rate 분포
        if mitotic_rates:
            stats['mitotic_rate_distribution'] = dict(Counter(mitotic_rates))

        # ER Allred score 통계
        if er_allred_scores:
            stats['er_allred_stats'] = {
                'mean': round(sum(er_allred_scores) / len(er_allred_scores), 1),
                'min': min(er_allred_scores),
                'max': max(er_allred_scores),
                'count': len(er_allred_scores)
            }

        # PR Allred score 통계
        if pr_allred_scores:
            stats['pr_allred_stats'] = {
                'mean': round(sum(pr_allred_scores) / len(pr_allred_scores), 1),
                'min': min(pr_allred_scores),
                'max': max(pr_allred_scores),
                'count': len(pr_allred_scores)
            }

        # 보조 호르몬 치료 분포
        if adjuvant_endocrine:
            endo_counter = Counter(adjuvant_endocrine)
            stats['adjuvant_endocrine_stats'] = {
                'no_treatment': endo_counter.get('0', 0),
                'treatment_yes': endo_counter.get('1', 0) + endo_counter.get('2', 0),
                'total': len(adjuvant_endocrine)
            }

        # 보조 방사선 치료 분포
        if adjuvant_rtx:
            rtx_counter = Counter(adjuvant_rtx)
            stats['adjuvant_rtx_stats'] = {
                'no_treatment': rtx_counter.get('0', 0),
                'treatment_yes': rtx_counter.get('1', 0) + rtx_counter.get('2', 0),
                'total': len(adjuvant_rtx)
            }

        # 수술 전 항암 치료 분포
        if neoadjuvant_ctx:
            stats['neoadjuvant_ctx_stats'] = dict(Counter(neoadjuvant_ctx))

        # 수술 전 항암 반응 분포
        if neoadjuvant_response:
            stats['neoadjuvant_response_distribution'] = dict(Counter(neoadjuvant_response))

        # 추적 관찰 기간 통계
        if surgery_dates and followup_dates:
            followup_periods = []
            for surg_date, fu_date in zip(surgery_dates, followup_dates):
                if surg_date and fu_date and fu_date > surg_date:
                    period_days = (fu_date - surg_date).days
                    followup_periods.append(period_days)

            if followup_periods:
                stats['followup_period_stats'] = {
                    'mean_days': round(sum(followup_periods) / len(followup_periods), 1),
                    'mean_months': round(sum(followup_periods) / len(followup_periods) / 30.44, 1),
                    'mean_years': round(sum(followup_periods) / len(followup_periods) / 365.25, 1),
                    'min_days': min(followup_periods),
                    'max_days': max(followup_periods),
                    'count': len(followup_periods)
                }

        logger.info(f"  ✅ 통계 계산 완료: {stats['total_patients']}명")
        return stats

    def _get_hospital_name(self, hospital_code: str) -> str:
        """병원 코드를 병원명으로 변환"""
        hospital_map = {
            "01": "세브란스",
            "02": "계명대",
            "03": "분당차",
            "04": "강남세브란스",
            "05": "강남차",
            "06": "단국대",
            "07": "이화여대"
        }
        return hospital_map.get(hospital_code, f"병원 {hospital_code}")

    def get_dataset_metadata(self, all_docs: list) -> dict:
        """
        CRF 데이터셋의 메타 정보 반환

        Args:
            all_docs: 전체 CRF 문서 리스트

        Returns:
            메타정보 딕셔너리 (병원 목록, 데이터 수집 기간, 필드 정보 등)
        """
        metadata = {
            "total_records": 0,
            "hospitals": {},
            "data_collection_period": {},
            "available_fields": [],
            "record_id_range": {}
        }

        if not all_docs:
            return metadata

        hospital_counts = {}
        all_dates = []
        field_set = set()
        record_ids = []

        for doc in all_docs:
            fields = doc.get("fields") if isinstance(doc, dict) else {}
            if not isinstance(fields, dict):
                fields = {}

            # 병원별 집계
            hospital = doc.get("hospital", "Unknown")
            hospital_counts[hospital] = hospital_counts.get(hospital, 0) + 1

            # Record ID 수집
            record_id = doc.get("record_id")
            if record_id:
                record_ids.append(record_id)

            # 수술 날짜 수집
            surgery_date = doc.get("수술연월일") or doc.get("surgery_date") or fields.get("수술연월일") or fields.get("surgery_date")
            if surgery_date:
                try:
                    if isinstance(surgery_date, str):
                        # YYYY-MM-DD 형식 파싱
                        date_obj = datetime.strptime(surgery_date, "%Y-%m-%d")
                        all_dates.append(date_obj)
                except:
                    pass

            # 필드명 수집
            for key in doc.keys():
                if key not in ["hospital", "record_id", "sheet", "path_no"]:
                    field_set.add(key)
            # fields 딕셔너리 내 컬럼도 포함
            for key in fields.keys():
                field_set.add(key)

        # 병원 정보 정리 (코드 → 이름 변환)
        for hospital_code, count in hospital_counts.items():
            hospital_name = self._get_hospital_name(hospital_code)
            metadata["hospitals"][hospital_name] = {
                "code": hospital_code,
                "count": count
            }

        # 데이터 수집 기간
        if all_dates:
            metadata["data_collection_period"] = {
                "earliest": min(all_dates).strftime("%Y-%m-%d"),
                "latest": max(all_dates).strftime("%Y-%m-%d"),
                "total_days": (max(all_dates) - min(all_dates)).days
            }

        # Record ID 범위
        if record_ids:
            metadata["record_id_range"] = {
                "first": min(record_ids),
                "last": max(record_ids),
                "total": len(set(record_ids))
            }

        # 사용 가능한 필드 목록 (정렬)
        metadata["available_fields"] = sorted(list(field_set))
        metadata["total_records"] = len(all_docs)

        return metadata

    def format_metadata_for_llm(self, metadata: dict) -> str:
        """
        메타데이터를 LLM이 읽기 쉬운 텍스트로 포맷팅

        Args:
            metadata: get_dataset_metadata()의 반환값

        Returns:
            포맷팅된 텍스트
        """
        lines = []
        lines.append("=== CRF Breast 데이터셋 메타 정보 ===")
        lines.append(f"\n총 레코드 수: {metadata.get('total_records', 0)}개")

        # 병원 정보
        hospitals = metadata.get('hospitals', {})
        if hospitals:
            lines.append(f"\n수집 병원 목록 ({len(hospitals)}개):")
            for hospital_name, info in sorted(hospitals.items()):
                lines.append(f"  - {hospital_name} (코드: {info['code']}): {info['count']}개 레코드")

        # 데이터 수집 기간
        period = metadata.get('data_collection_period', {})
        if period:
            lines.append(f"\n데이터 수집 기간:")
            lines.append(f"  - 가장 오래된 수술일: {period.get('earliest', 'N/A')}")
            lines.append(f"  - 가장 최신 수술일: {period.get('latest', 'N/A')}")
            lines.append(f"  - 총 수집 기간: 약 {period.get('total_days', 0)}일")

        # Record ID 범위
        record_range = metadata.get('record_id_range', {})
        if record_range:
            lines.append(f"\nRecord ID 범위:")
            lines.append(f"  - 첫 번째: {record_range.get('first', 'N/A')}")
            lines.append(f"  - 마지막: {record_range.get('last', 'N/A')}")
            lines.append(f"  - 고유 환자 수: {record_range.get('total', 0)}명")

        # 사용 가능한 필드 목록
        fields = metadata.get('available_fields', [])
        if fields:
            lines.append(f"\n사용 가능한 데이터 필드 ({len(fields)}개):")
            # 필드가 너무 많으면 상위 30개만 표시
            display_fields = fields[:30]
            for field in display_fields:
                lines.append(f"  - {field}")
            if len(fields) > 30:
                lines.append(f"  ...총 {len(fields)}개 필드 중 30개만 표시")

        return "\n".join(lines)

    def format_statistics_for_llm(self, stats: dict) -> str:
        """
        통계 데이터를 LLM이 읽기 쉬운 텍스트로 포맷팅

        Args:
            stats: calculate_crf_statistics()의 반환값

        Returns:
            포맷팅된 텍스트
        """
        lines = []
        lines.append(f"=== {stats['hospital_name']} CRF 데이터 통계 ===")
        lines.append(f"\n총 환자 수: {stats['total_patients']}명")
        lines.append(f"총 문서 수: {stats['total_documents']}개")

        if stats['age_stats']:
            age = stats['age_stats']
            lines.append(f"\n진단 시 나이:")
            lines.append(f"  - 평균: {age['mean']}세")
            lines.append(f"  - 범위: {age['min']}세 ~ {age['max']}세")
            lines.append(f"  - 데이터 수: {age['count']}명")

        if stats['tumor_size_stats']:
            size = stats['tumor_size_stats']
            lines.append(f"\n암 크기 (장경):")
            lines.append(f"  - 평균: {size['mean']} mm")
            lines.append(f"  - 범위: {size['min']} mm ~ {size['max']} mm")
            lines.append(f"  - 데이터 수: {size['count']}명")

        if stats['er_stats']:
            er = stats['er_stats']
            lines.append(f"\nER (Estrogen Receptor):")
            lines.append(f"  - Positive: {er['positive']}명 ({er['percentage']}%)")
            lines.append(f"  - Negative: {er['total'] - er['positive']}명")
            lines.append(f"  - 총 데이터: {er['total']}명")

        if stats['pr_stats']:
            pr = stats['pr_stats']
            lines.append(f"\nPR (Progesterone Receptor):")
            lines.append(f"  - Positive: {pr['positive']}명 ({pr['percentage']}%)")
            lines.append(f"  - Negative: {pr['total'] - pr['positive']}명")
            lines.append(f"  - 총 데이터: {pr['total']}명")

        if stats['her2_stats']:
            her2 = stats['her2_stats']
            lines.append(f"\nHER2:")
            lines.append(f"  - Positive: {her2['positive']}명 ({her2['percentage']}%)")
            lines.append(f"  - Negative: {her2['total'] - her2['positive']}명")
            lines.append(f"  - 총 데이터: {her2['total']}명")

        if stats['stage_distribution']:
            lines.append(f"\nAJCC Stage 분포:")
            for stage, count in sorted(stats['stage_distribution'].items()):
                lines.append(f"  - {stage}: {count}명")

        if stats['ng_distribution']:
            lines.append(f"\nNuclear Grade (NG) 분포:")
            for grade, count in sorted(stats['ng_distribution'].items()):
                lines.append(f"  - {grade}: {count}명")

        if stats['hg_distribution']:
            lines.append(f"\nHistologic Grade (HG) 분포:")
            for grade, count in sorted(stats['hg_distribution'].items()):
                lines.append(f"  - {grade}: {count}명")

        if stats['ki67_stats']:
            ki67 = stats['ki67_stats']
            lines.append(f"\nKi-67 증식 지표:")
            lines.append(f"  - 평균: {ki67['mean']}%")
            lines.append(f"  - 범위: {ki67['min']}% ~ {ki67['max']}%")
            lines.append(f"  - 데이터 수: {ki67['count']}명")
        if stats['ki67_thresholds']:
            lines.append(f"\nKi-67 임계값별 통계:")
            for item in stats['ki67_thresholds']:
                lines.append(f"  - {item['threshold']}% 이상: {item['count']}명 ({item['percentage']}%)")

        if stats['lymph_node_stats']:
            ln = stats['lymph_node_stats']
            lines.append(f"\n림프절 전이:")
            lines.append(f"  - 전이 있음: {ln['positive']}명 ({ln['percentage']}%)")
            lines.append(f"  - 전이 없음: {ln['total'] - ln['positive']}명")
            lines.append(f"  - 총 데이터: {ln['total']}명")
            if 'mean_count' in ln:
                lines.append(f"  - 평균 전이 개수: {ln['mean_count']}개 (최대: {ln['max_count']}개)")

        if stats['tnm_stats']:
            tnm = stats['tnm_stats']
            if 'T' in tnm:
                lines.append(f"\nT category (종양 크기):")
                for cat, count in sorted(tnm['T'].items()):
                    lines.append(f"  - {cat}: {count}명")
            if 'N' in tnm:
                lines.append(f"\nN category (림프절):")
                for cat, count in sorted(tnm['N'].items()):
                    lines.append(f"  - {cat}: {count}명")
            if 'M' in tnm:
                lines.append(f"\nM category (원격 전이):")
                for cat, count in sorted(tnm['M'].items()):
                    lines.append(f"  - {cat}: {count}명")

        if stats['histologic_type_distribution']:
            lines.append(f"\n조직학적 타입:")
            for htype, count in sorted(stats['histologic_type_distribution'].items()):
                lines.append(f"  - {htype}: {count}명")

        if stats['surgery_type_distribution']:
            lines.append(f"\n수술 방법:")
            for stype, count in sorted(stats['surgery_type_distribution'].items()):
                lines.append(f"  - {stype}: {count}명")

        if stats.get('biomarker_combinations'):
            lines.append(f"\n바이오마커 조합 통계:")
            combos = stats['biomarker_combinations']
            for label, data in combos.items():
                lines.append(f"  - {label}: {data['count']}명 ({data['percentage']}%)")

        if stats.get('hospital_counts'):
            lines.append(f"\n병원별 건수:")
            for hosp, count in sorted(stats['hospital_counts'].items()):
                lines.append(f"  - 병원 {hosp}: {count}명")

        if stats.get('surgery_year_distribution'):
            lines.append(f"\n수술 연도 분포:")
            for year, count in sorted(stats['surgery_year_distribution'].items()):
                lines.append(f"  - {year}: {count}명")

        if stats.get('stage_ng_distribution'):
            lines.append(f"\nStage x NG 분포:")
            for label, count in sorted(stats['stage_ng_distribution'].items()):
                lines.append(f"  - {label}: {count}명")

        if stats.get('stage_hg_distribution'):
            lines.append(f"\nStage x HG 분포:")
            for label, count in sorted(stats['stage_hg_distribution'].items()):
                lines.append(f"  - {label}: {count}명")

        if stats['recurrence_stats']:
            rec = stats['recurrence_stats']
            lines.append(f"\n재발 현황:")
            lines.append(f"  - Axillary LN 재발: {rec['axillary_ln']}명")
            lines.append(f"  - 수술 부위 재발: {rec['surgery_site']}명")
            lines.append(f"  - 원격 전이: {rec['distant_metastasis']}명")
            lines.append(f"  - 총 재발 환자: {rec['total_with_recurrence']}명 ({rec['recurrence_rate']}%)")

        if stats['survival_stats']:
            surv = stats['survival_stats']
            lines.append(f"\n생존 현황:")
            lines.append(f"  - 생존: {surv['alive']}명 ({surv['survival_rate']}%)")
            lines.append(f"  - 질병으로 사망: {surv['dead_from_disease']}명")
            lines.append(f"  - 기타 사망: {surv['dead_from_other']}명")
            lines.append(f"  - 총 데이터: {surv['total']}명")

        # ===== 새로 추가된 필드 통계 포맷팅 =====

        # 암의 위치 분포
        if stats.get('tumor_location_distribution'):
            lines.append(f"\n암의 위치:")
            for location, count in sorted(stats['tumor_location_distribution'].items()):
                lines.append(f"  - {location}: {count}명")

        # 암의 개수 분포
        if stats.get('tumor_number_distribution'):
            lines.append(f"\n암의 개수:")
            for number, count in sorted(stats['tumor_number_distribution'].items()):
                lines.append(f"  - {number}: {count}명")

        # HER2 IHC 분포
        if stats.get('her2_ihc_distribution'):
            lines.append(f"\nHER2 IHC 등급:")
            for grade, count in sorted(stats['her2_ihc_distribution'].items()):
                lines.append(f"  - {grade}: {count}명")

        # DCIS/LCIS 분포
        if stats.get('dcis_lcis_distribution'):
            lines.append(f"\nDCIS/LCIS 여부:")
            for status, count in sorted(stats['dcis_lcis_distribution'].items()):
                lines.append(f"  - {status}: {count}명")

        # Mitotic Rate 분포
        if stats.get('mitotic_rate_distribution'):
            lines.append(f"\nMitotic Rate (HG score 3):")
            for rate, count in sorted(stats['mitotic_rate_distribution'].items()):
                lines.append(f"  - {rate}: {count}명")

        # ER Allred score
        if stats.get('er_allred_stats'):
            er_allred = stats['er_allred_stats']
            lines.append(f"\nER Allred Score:")
            lines.append(f"  - 평균: {er_allred['mean']}")
            lines.append(f"  - 범위: {er_allred['min']} ~ {er_allred['max']}")
            lines.append(f"  - 데이터 수: {er_allred['count']}명")

        # PR Allred score
        if stats.get('pr_allred_stats'):
            pr_allred = stats['pr_allred_stats']
            lines.append(f"\nPR Allred Score:")
            lines.append(f"  - 평균: {pr_allred['mean']}")
            lines.append(f"  - 범위: {pr_allred['min']} ~ {pr_allred['max']}")
            lines.append(f"  - 데이터 수: {pr_allred['count']}명")

        # 보조 호르몬 치료
        if stats.get('adjuvant_endocrine_stats'):
            endo = stats['adjuvant_endocrine_stats']
            lines.append(f"\n보조 호르몬 치료:")
            lines.append(f"  - 치료 받음: {endo['treatment_yes']}명")
            lines.append(f"  - 치료 안 받음: {endo['no_treatment']}명")
            lines.append(f"  - 총 데이터: {endo['total']}명")

        # 보조 방사선 치료
        if stats.get('adjuvant_rtx_stats'):
            rtx = stats['adjuvant_rtx_stats']
            lines.append(f"\n보조 방사선 치료:")
            lines.append(f"  - 치료 받음: {rtx['treatment_yes']}명")
            lines.append(f"  - 치료 안 받음: {rtx['no_treatment']}명")
            lines.append(f"  - 총 데이터: {rtx['total']}명")

        # 수술 전 항암 치료
        if stats.get('neoadjuvant_ctx_stats'):
            lines.append(f"\n수술 전 항암 치료:")
            for status, count in sorted(stats['neoadjuvant_ctx_stats'].items()):
                lines.append(f"  - {status}: {count}명")

        # 수술 전 항암 반응
        if stats.get('neoadjuvant_response_distribution'):
            lines.append(f"\n수술 전 항암 치료 반응:")
            for response, count in sorted(stats['neoadjuvant_response_distribution'].items()):
                lines.append(f"  - {response}: {count}명")

        # 추적 관찰 기간
        if stats.get('followup_period_stats'):
            fu = stats['followup_period_stats']
            lines.append(f"\n추적 관찰 기간:")
            lines.append(f"  - 평균: {fu['mean_years']}년 ({fu['mean_months']}개월)")
            lines.append(f"  - 범위: {round(fu['min_days']/365.25, 1)}년 ~ {round(fu['max_days']/365.25, 1)}년")
            lines.append(f"  - 데이터 수: {fu['count']}명")

        text = "\n".join(lines)

        return text
