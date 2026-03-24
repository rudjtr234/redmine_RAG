"""CRF 데이터 통계 계산 모듈 - 8개 암종 통합"""
import logging
import re
from collections import Counter
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================
# 병원 코드 매핑 (암종별)
# ============================================================

HOSPITAL_MAP_BREAST = {
    "01": "세브란스", "02": "계명대", "03": "분당차",
    "04": "강남세브란스", "05": "강남차", "06": "단국대", "07": "이화여대"
}

HOSPITAL_MAP_STOMACH = {
    "1": "세브란스", "2": "분당차", "3": "이대", "4": "단국대", "5": "계명대",
    "01": "신촌세브란스", "02": "계명대", "03": "분당차", "04": "단국대", "07": "이대목동"
}

HOSPITAL_MAP_DEFAULT = {
    "01": "병원01", "02": "병원02", "03": "병원03",
    "04": "병원04", "05": "병원05", "06": "병원06", "07": "병원07"
}

CATEGORY_LABEL = {
    "breast": "유방암",
    "stomach": "위암",
    "thyroid": "갑상선암",
    "colorectal": "대장암",
    "hrd": "난소암(HRD)",
    "cervix": "자궁경부암",
    "prostate": "전립선암",
    "liver": "간암",
}


def _get_hospital_name(hospital_code: str, category: str = "") -> str:
    if category == "breast":
        return HOSPITAL_MAP_BREAST.get(str(hospital_code), f"병원 {hospital_code}")
    elif category == "stomach":
        return HOSPITAL_MAP_STOMACH.get(str(hospital_code), f"병원 {hospital_code}")
    else:
        return HOSPITAL_MAP_DEFAULT.get(str(hospital_code), f"병원 {hospital_code}")


def _normalize_colorectal_stage(raw_value: str, stage_type: str):
    """
    대장암 T/N stage 값 정규화.
    - 숫자 코드(예: 3, 2, 0) -> 표준 stage 라벨
    - 문자열 stage(예: T4, N1, T4a, N2b) -> 표준화
    - x/미상/pCR/No residual 등 특수값 처리
    - 코드북/헤더 값(예: "T stage", "Tis: 0")은 제외
    """
    if raw_value is None:
        return None

    st = stage_type.upper()
    if st not in ("T", "N"):
        return None

    value = str(raw_value).strip()
    if not value:
        return None

    value_l = value.lower()

    # 코드북/헤더 row 방어
    if value_l in ("t stage", "n stage"):
        return None
    if re.match(r'^[tn][^:]*:\s*\d+\s*$', value_l):
        return None

    # 명시적 stage 표기 우선 처리 (T4, N1, T4a, N2b, Tis, Tx ...)
    if st == "T":
        explicit = re.search(r'\bT\s*(IS|X|0|1|2|3|4A|4B|4)\b', value, re.IGNORECASE)
        if explicit:
            token = explicit.group(1).upper()
            if token == "IS":
                return "Tis"
            return f"T{token}"
    else:
        explicit = re.search(r'\bN\s*(X|0|1A|1B|1C|1|2A|2B|2)\b', value, re.IGNORECASE)
        if explicit:
            token = explicit.group(1).upper()
            return f"N{token}"

    # 숫자 코드 -> stage 라벨 매핑
    if re.fullmatch(r'[0-5]', value):
        if st == "T":
            code_map = {
                "0": "Tis",
                "1": "T1",
                "2": "T2",
                "3": "T3",
                "4": "T4a",
                "5": "T4b",
            }
        else:
            code_map = {
                "0": "N0",
                "1": "N1a",
                "2": "N1b",
                "3": "N1c",
                "4": "N2a",
                "5": "N2b",
            }
        return code_map.get(value)

    # 특수/결측값 처리
    if value_l in ("x", "미상", "unknown", "unk"):
        return f"{st}x"

    if st == "T" and (
        "pcr" in value_l
        or "no residual" in value_l
        or "ccrt후 no residual cancer" in value_l
    ):
        return "T0"

    return None


# ============================================================
# 암종별 통계 파서
# ============================================================

def _parse_breast(doc: str, meta: dict, accum: dict):
    """유방암 필드 파싱"""
    a = accum

    age = re.search(r'나이\s*\(진단시\)\s*:\s*(\d+)', doc)
    if age:
        a['ages'].append(int(age.group(1)))

    size = re.search(r'암\s*size\s*\(mm\)_장경\s*:\s*(\d+(?:\.\d+)?)', doc)
    if size:
        a['tumor_sizes'].append(float(size.group(1)))

    er = re.search(r'ER\s*\(-/\+\)\s*:\s*([01])', doc)
    if er:
        a['er_total'] += 1
        if er.group(1) == '1':
            a['er_positive'] += 1

    pr = re.search(r'PR\s*\(-/\+\)\s*:\s*([01])', doc)
    if pr:
        a['pr_total'] += 1
        if pr.group(1) == '1':
            a['pr_positive'] += 1

    her2 = re.search(r'HER2\s*\(-/\+\)\s*:\s*([01])', doc)
    if her2:
        a['her2_total'] += 1
        if her2.group(1) == '1':
            a['her2_positive'] += 1

    her2_ihc = re.search(r'HER2_IHC\s*\(0/\+1/\s*\+2/\s*\+3\)\s*:\s*([0123])', doc)
    if her2_ihc:
        ihc_map = {'0': 'IHC 0', '1': 'IHC 1+', '2': 'IHC 2+', '3': 'IHC 3+'}
        a['her2_ihc'].append(ihc_map.get(her2_ihc.group(1), her2_ihc.group(1)))

    ki67 = re.search(r'KI-67\s*LI\s*\(%\)\s*:\s*(\d+(?:\.\d+)?)', doc, re.IGNORECASE)
    if ki67:
        a['ki67_values'].append(float(ki67.group(1)))

    stage = re.search(r'AJCC\s*stage\s*\(8판\)\s*:\s*(\d+)', doc, re.IGNORECASE)
    if stage:
        stage_map = {'1': 'Stage I', '2': 'Stage II', '3': 'Stage III', '4': 'Stage IV'}
        a['stages'].append(stage_map.get(stage.group(1), f'Stage {stage.group(1)}'))

    ng = re.search(r'NG\s*\(1/2/3\)\s*:\s*([123])', doc)
    if ng:
        a['ng_grades'].append(f'Grade {ng.group(1)}')

    hg = re.search(r'HG\s*\(1/2/3/4\)\s*:\s*([1234])', doc)
    if hg:
        a['hg_grades'].append(f'Grade {hg.group(1)}')

    ln = re.search(r'림프절\s*전이여부_수술당시.*?:\s*([0123])', doc)
    if ln:
        a['ln_total'] += 1
        if ln.group(1) != '0':
            a['ln_positive'] += 1

    t_cat = re.search(r'T\s*category\s*:\s*(\d+)', doc)
    if t_cat:
        a['t_categories'].append(f'T{t_cat.group(1)}')

    n_cat = re.search(r'N\s*category\s*:\s*(\d+)', doc)
    if n_cat:
        a['n_categories'].append(f'N{n_cat.group(1)}')

    hist = re.search(r'진단명\s*\(histologic\s*type.*?\)\s*:\s*:\s*([1234])', doc)
    if hist:
        type_map = {'1': 'Ductal', '2': 'Lobular', '3': 'Mucinous', '4': 'Other'}
        a['histologic_types'].append(type_map.get(hist.group(1), f'Type {hist.group(1)}'))

    surg = re.search(r'수술명\s*\(partial/total\)\s*:\s*([12])', doc)
    if surg:
        a['surgery_types'].append('Total mastectomy' if surg.group(1) == '2' else 'Partial mastectomy')

    location = re.search(r'암의\s*위치\s*\(Rt\./Lt\./Both\)\s*:\s*([123])', doc)
    if location:
        loc_map = {'1': 'Right', '2': 'Left', '3': 'Both'}
        a['tumor_locations'].append(loc_map.get(location.group(1), location.group(1)))

    num = re.search(r'암의\s*개수\s*\(single/multiple\)\s*:\s*([12])', doc)
    if num:
        a['tumor_numbers'].append('Single' if num.group(1) == '1' else 'Multiple')

    recur_ax = re.search(r'Axillary\s*LN\s*재발\s*여부\s*\(0/1\)\s*:\s*([01])', doc)
    if recur_ax:
        a['recur_total'] += 1
        if recur_ax.group(1) == '1':
            a['recur_axillary'] += 1

    recur_site = re.search(r'수술부위\s*재발여부\s*\(0/1\)\s*:\s*([01])', doc)
    if recur_site and recur_site.group(1) == '1':
        a['recur_site'] += 1

    distant = re.search(r'다른\s*장기로\s*전이\s*여부\s*\(0/1\)\s*:\s*([01])', doc)
    if distant and distant.group(1) == '1':
        a['distant_meta'] += 1

    survival = re.search(r'이\s*질병으로\s*사망여부.*?:\s*([012])', doc)
    if survival:
        a['survival_total'] += 1
        s = survival.group(1)
        if s == '0': a['alive'] += 1
        elif s == '1': a['dead_disease'] += 1
        elif s == '2': a['dead_other'] += 1

    surg_date = re.search(r'수술연월일\s*:\s*(\d{4}-\d{2}-\d{2})', doc)
    if surg_date:
        try:
            a['surgery_years'].append(int(surg_date.group(1)[:4]))
        except: pass

    neoadj = re.search(r'neoadjuvantCTx\s*\(0:무,\s*1:유\)\s*:\s*([01])', doc, re.IGNORECASE)
    if neoadj:
        a['neoadjuvant_ctx'].append('유' if neoadj.group(1) == '1' else '무')

    adj_endo = re.search(r'adjuvant\s*Endocrine/Hormonal\s*Tx\s*:\s*([012])', doc, re.IGNORECASE)
    if adj_endo:
        a['adjuvant_endocrine'].append(adj_endo.group(1))

    adj_rtx = re.search(r'adjuvant\s*RTx\s*:\s*([012])', doc, re.IGNORECASE)
    if adj_rtx:
        a['adjuvant_rtx'].append(adj_rtx.group(1))


def _parse_stomach(doc: str, meta: dict, accum: dict):
    """위암 필드 파싱"""
    a = accum

    age = re.search(r'진단나이\s*:\s*(\d+)', doc)
    if age:
        a['ages'].append(int(age.group(1)))

    gender = re.search(r'성별\s*:\s*([01])', doc)
    if gender:
        a['genders'].append('Male' if gender.group(1) == '1' else 'Female')

    treatment = re.search(r'수술\s*vs\s*ESD\s*:\s*([01])', doc)
    if treatment:
        a['treatment_types'].append('ESD' if treatment.group(1) == '1' else '수술')

    size = re.search(r'크기\(cm\)\s*:\s*(\d+(?:\.\d+)?)', doc)
    if size:
        a['tumor_sizes'].append(float(size.group(1)))

    t_stage = re.search(r'T\s*stage\s*:\s*([1-6])', doc)
    if t_stage:
        t_map = {'1': 'T1a', '2': 'T1b', '3': 'T2', '4': 'T3', '5': 'T4a', '6': 'T4b'}
        a['t_stages'].append(t_map.get(t_stage.group(1), f'T{t_stage.group(1)}'))

    n_stage = re.search(r'N\s*stage\s*:\s*([01234])', doc)
    if n_stage:
        n_map = {'0': 'N0', '1': 'N1', '2': 'N2', '3': 'N3a', '4': 'N3b'}
        a['n_stages'].append(n_map.get(n_stage.group(1), f'N{n_stage.group(1)}'))

    her2 = re.search(r'HER2\s*:\s*([012])', doc)
    if her2:
        a['her2_total'] += 1
        if her2.group(1) == '2': a['her2_positive'] += 1

    msi = re.search(r'MSI\s*:\s*([019])', doc)
    if msi and msi.group(1) != '9':
        a['msi_statuses'].append('MSI-H' if msi.group(1) == '1' else 'Intact')

    lvi = re.search(r'LVI\s*:\s*([01])', doc)
    if lvi:
        a['lvi_total'] += 1
        if lvi.group(1) == '1': a['lvi_positive'] += 1

    pni = re.search(r'PNI\s*:\s*([01])', doc)
    if pni:
        a['pni_total'] += 1
        if pni.group(1) == '1': a['pni_positive'] += 1

    location = re.search(r'종양위치\s*:\s*([1234])', doc)
    if location:
        loc_map = {'1': 'Upper third', '2': 'Middle third', '3': 'Lower third', '4': 'Whole/Others'}
        a['tumor_locations'].append(loc_map.get(location.group(1), location.group(1)))

    event = re.search(r'event\s*:\s*([012])', doc)
    if event and event.group(1) != '9':
        a['survival_total'] += 1
        e = event.group(1)
        if e == '0': a['alive'] += 1
        elif e == '1': a['recur_total'] += 1
        elif e == '2': a['dead_disease'] += 1

    surg_date = re.search(r'수술/ESD\s*날짜\s*:\s*(\d{4})', doc)
    if surg_date:
        a['surgery_years'].append(int(surg_date.group(1)))


def _parse_thyroid(doc: str, meta: dict, accum: dict):
    """갑상선암 필드 파싱"""
    a = accum

    age = re.search(r'나이\s*\(진단시\)\s*:\s*(\d+)', doc)
    if age:
        a['ages'].append(int(age.group(1)))

    gender = re.search(r'성별\s*:\s*([12])', doc)
    if gender:
        a['genders'].append('Male' if gender.group(1) == '1' else 'Female')

    size = re.search(r'암\s*size\s*\(mm\)_장경\s*:\s*(\d+(?:\.\d+)?)', doc)
    if size:
        a['tumor_sizes'].append(float(size.group(1)))

    surg_method = re.search(r'수술방법\s*\(excision/lobectomy/total\)\s*:\s*([123])', doc)
    if surg_method:
        method_map = {'1': 'Excision', '2': 'Lobectomy', '3': 'Total thyroidectomy'}
        a['surgery_types'].append(method_map.get(surg_method.group(1), surg_method.group(1)))

    ete = re.search(r'Extrathyroid\s*extension.*?:\s*([012])', doc, re.IGNORECASE)
    if ete:
        ete_map = {'0': 'No', '1': 'Yes(micro)', '2': 'Yes(gross)'}
        a['ete_statuses'].append(ete_map.get(ete.group(1), ete.group(1)))

    t_cat = re.search(r'T\s*category\s*:\s*(\d+)', doc)
    if t_cat:
        a['t_categories'].append(f'T{t_cat.group(1)}')

    n_cat = re.search(r'N\s*category\s*:\s*(\d+)', doc)
    if n_cat:
        a['n_categories'].append(f'N{n_cat.group(1)}')

    stage = re.search(r'AJCC\s*stage\s*\(8판\)\s*:\s*(\d+)', doc, re.IGNORECASE)
    if stage:
        stage_map = {'1': 'Stage I', '2': 'Stage II', '3': 'Stage III', '4': 'Stage IV'}
        a['stages'].append(stage_map.get(stage.group(1), f'Stage {stage.group(1)}'))

    braf = re.search(r'BRAF\s*mutation\s*\(0/1\)\s*:\s*([01])', doc)
    if braf:
        a['braf_total'] += 1
        if braf.group(1) == '1': a['braf_positive'] += 1

    ln = re.search(r'림프절\s*전이여부_수술당시.*?:\s*([0123])', doc)
    if ln:
        a['ln_total'] += 1
        if ln.group(1) != '0': a['ln_positive'] += 1

    recur = re.search(r'Neck\s*LN\s*재발\s*여부.*?:\s*([01])', doc)
    if recur:
        a['recur_total'] += 1
        if recur.group(1) == '1': a['recur_axillary'] += 1

    distant = re.search(r'다른\s*장기로\s*전이\s*여부.*?:\s*([01])', doc)
    if distant and distant.group(1) == '1':
        a['distant_meta'] += 1

    survival = re.search(r'이\s*질병으로\s*사망여부.*?:\s*([0123])', doc)
    if survival:
        a['survival_total'] += 1
        s = survival.group(1)
        if s == '0': a['alive'] += 1
        elif s == '1': a['dead_disease'] += 1
        elif s == '2': a['dead_other'] += 1

    surg_date = re.search(r'수술연월일\s*:\s*(\d{4})', doc)
    if surg_date:
        a['surgery_years'].append(int(surg_date.group(1)))

    location = re.search(r'암의\s*위치\s*\(Rt\./Lt\./isthmus/Both\)\s*:\s*([1234])', doc)
    if location:
        loc_map = {'1': 'Right', '2': 'Left', '3': 'Isthmus', '4': 'Both'}
        a['tumor_locations'].append(loc_map.get(location.group(1), location.group(1)))

    num = re.search(r'암의\s*개수\s*\(single/multiple\)\s*:\s*([12])', doc)
    if num:
        a['tumor_numbers'].append('Single' if num.group(1) == '1' else 'Multiple')

    lympho = re.search(r'Non-neoplastic\s*parenchyma.*?:\s*([01])', doc)
    if lympho:
        a['lymphocytic_thyroiditis'].append('Yes' if lympho.group(1) == '1' else 'No')


def _parse_colorectal(doc: str, meta: dict, accum: dict):
    """대장암 필드 파싱 (컬럼 설명 시트 구조)"""
    a = accum

    gender = re.search(r'Level\s*I\.3\s*:\s*(Female|Male|\d+)', doc, re.IGNORECASE)
    if gender:
        a['genders'].append(gender.group(1))

    age = re.search(r'Level\s*I\.4\s*:\s*(\d+)', doc)
    if age:
        try: a['ages'].append(int(age.group(1)))
        except: pass

    t_stage = re.search(r'Level\s*I\.8\s*:\s*([^\n\r]+)', doc, re.IGNORECASE)
    if t_stage:
        normalized_t = _normalize_colorectal_stage(t_stage.group(1), "T")
        if normalized_t:
            a['t_stages'].append(normalized_t)

    n_stage = re.search(r'Level\s*I\.9\s*:\s*([^\n\r]+)', doc, re.IGNORECASE)
    if n_stage:
        normalized_n = _normalize_colorectal_stage(n_stage.group(1), "N")
        if normalized_n:
            a['n_stages'].append(normalized_n)

    lvi = re.search(r'Level\s*I\.10\s*:\s*([01])', doc)
    if lvi:
        a['lvi_total'] += 1
        if lvi.group(1) == '1': a['lvi_positive'] += 1

    pni = re.search(r'Level\s*I\.11\s*:\s*([01])', doc)
    if pni:
        a['pni_total'] += 1
        if pni.group(1) == '1': a['pni_positive'] += 1

    msi = re.search(r'Level\s*I\.13\s*:\s*(\w+)', doc)
    if msi:
        a['msi_statuses'].append(msi.group(1))

    kras = re.search(r'Level\s*I\.14\s*:\s*([01mutMUT\w]+)', doc)
    if kras:
        a['kras_statuses'].append(kras.group(1))

    braf = re.search(r'Level\s*I\.16\s*:\s*([01mutMUT\w]+)', doc)
    if braf:
        a['braf_statuses'].append(braf.group(1))

    survival = re.search(r'Level\s*I\.18\s*:\s*([01])', doc)
    if survival:
        a['survival_total'] += 1
        if survival.group(1) == '1': a['alive'] += 1
        else: a['dead_disease'] += 1

    local_recur = re.search(r'Level\s*I\.20\s*:\s*([01])', doc)
    if local_recur and local_recur.group(1) == '1':
        a['recur_total'] += 1

    distant = re.search(r'Level\s*I\.22\s*:\s*([01])', doc)
    if distant and distant.group(1) == '1':
        a['distant_meta'] += 1


def _parse_hrd(doc: str, meta: dict, accum: dict):
    """난소암 HRD 필드 파싱"""
    a = accum

    age = re.search(r'Age\s*:\s*(\d+)', doc)
    if age:
        a['ages'].append(int(age.group(1)))

    hrd_group = re.search(r'HRD\s*group\s*:\s*([012])', doc)
    if hrd_group:
        a['hrd_groups'].append(int(hrd_group.group(1)))

    hrd_score = re.search(r'HRD\s*score\s*:\s*(-?\d+(?:\.\d+)?)', doc)
    if hrd_score:
        a['hrd_scores'].append(float(hrd_score.group(1)))

    brca1 = re.search(r'BRCA1\s*:\s*([01])', doc)
    if brca1:
        a['brca1_total'] += 1
        if brca1.group(1) == '1': a['brca1_positive'] += 1

    brca2 = re.search(r'BRCA2\s*:\s*([01])', doc)
    if brca2:
        a['brca2_total'] += 1
        if brca2.group(1) == '1': a['brca2_positive'] += 1

    figo = re.search(r'FIGO\s*stage\s*:\s*(\d+)', doc)
    if figo:
        figo_map = {
            '1': 'I', '2': 'II', '3': 'III', '4': 'IV',
            '5': 'IA', '6': 'IB', '7': 'IC', '8': 'IC1', '9': 'IC2', '10': 'IC3',
            '11': 'IIA', '12': 'IIB', '13': 'IIIA', '14': 'IIIB', '15': 'IIIC',
            '16': 'IVA', '17': 'IVB',
        }
        a['stages'].append(f'FIGO {figo_map.get(figo.group(1), figo.group(1))}')

    histology = re.search(r'Histology\s*:\s*(\d+)', doc)
    if histology:
        hist_map = {'1': 'HGSC', '2': 'LGSC', '3': 'Endometrioid', '4': 'Clear cell', '5': 'Mucinous'}
        a['histologic_types'].append(hist_map.get(histology.group(1), f'Type {histology.group(1)}'))

    tp53 = re.search(r'TP53\s*mutation\s*:\s*([01])', doc)
    if tp53:
        a['tp53_total'] += 1
        if tp53.group(1) == '1': a['tp53_positive'] += 1

    msi = re.search(r'MSI\s*:\s*([01])', doc)
    if msi:
        a['msi_statuses'].append('MSI-H' if msi.group(1) == '1' else 'Intact')

    death = re.search(r'사망여부\s*:\s*([01])', doc)
    if death:
        a['survival_total'] += 1
        if death.group(1) == '0': a['alive'] += 1
        else: a['dead_disease'] += 1

    neoadj = re.search(r'선행항암치료\s*여부\s*:\s*([01])', doc)
    if neoadj:
        a['neoadjuvant_ctx'].append('유' if neoadj.group(1) == '1' else '무')

    chemo_line = re.search(r'총\s*항암횟수\s*\(1st\s*line\)\s*:\s*(\d+)', doc)
    if chemo_line:
        a['chemo_cycles'].append(int(chemo_line.group(1)))


def _parse_cervix(doc: str, meta: dict, accum: dict):
    """자궁경부암 필드 파싱"""
    a = accum

    age = re.search(r'나이\s*\(진단시\)\s*:\s*(\d+)', doc)
    if age:
        a['ages'].append(int(age.group(1)))

    hpv = re.search(r'HPV\s*유무\s*\(0:\s*No,\s*1:\s*Yes\)\s*:\s*([01])', doc)
    if hpv:
        a['hpv_total'] += 1
        if hpv.group(1) == '1': a['hpv_positive'] += 1

    hpv16 = re.search(r'HPV\s*16.*?:\s*([01])', doc)
    if hpv16 and hpv16.group(1) == '1':
        a['hpv16_positive'] += 1

    hpv18 = re.search(r'HPV\s*18.*?:\s*([01])', doc)
    if hpv18 and hpv18.group(1) == '1':
        a['hpv18_positive'] += 1

    hpv_hr = re.search(r'HPV\s*other\s*high\s*risk.*?:\s*([01])', doc)
    if hpv_hr and hpv_hr.group(1) == '1':
        a['hpv_other_hr_positive'] += 1

    surg = re.search(r'수술\s*여부.*?:\s*([01234])', doc)
    if surg:
        surg_map = {'0': 'None', '1': 'LEEP', '2': 'Hysterectomy',
                    '3': 'LEEP+Hysterectomy', '4': 'Trachelectomy'}
        a['surgery_types'].append(surg_map.get(surg.group(1), surg.group(1)))

    lvi = re.search(r'LVI\s*:\s*([01])', doc)
    if lvi:
        a['lvi_total'] += 1
        if lvi.group(1) == '1': a['lvi_positive'] += 1

    margin = re.search(r'Margin\s*status\s*\(-/\+\)\s*:\s*([01])', doc)
    if margin:
        a['margin_total'] += 1
        if margin.group(1) == '1': a['margin_positive'] += 1

    surg_date = re.search(r'수술연월일\s*:\s*(\d{4})', doc)
    if surg_date:
        a['surgery_years'].append(int(surg_date.group(1)))


def _parse_prostate(doc: str, meta: dict, accum: dict):
    """전립선암 필드 파싱"""
    a = accum

    age = re.search(r'Age\s*:\s*(\d+)', doc)
    if age:
        a['ages'].append(int(age.group(1)))

    pgg = re.search(r'PGG\s*:\s*([12345])', doc)
    if pgg:
        a['pgg_grades'].append(f'Grade Group {pgg.group(1)}')

    stage = re.search(r'Stage\s*:\s*([12345])', doc)
    if stage:
        a['stages'].append(f'Stage {stage.group(1)}')

    pni = re.search(r'PNI\s*:\s*([01])', doc)
    if pni:
        a['pni_total'] += 1
        if pni.group(1) == '1': a['pni_positive'] += 1

    lvi = re.search(r'LVI\s*:\s*([01])', doc)
    if lvi:
        a['lvi_total'] += 1
        if lvi.group(1) == '1': a['lvi_positive'] += 1

    bcr = re.search(r'BCR\s*:\s*([01])', doc)
    if bcr:
        a['bcr_total'] += 1
        if bcr.group(1) == '1': a['bcr_positive'] += 1

    distant = re.search(r'Distant_meta\s*:\s*([01])', doc)
    if distant and distant.group(1) == '1':
        a['distant_meta'] += 1

    overall_mortality = re.search(r'Overall_Mortality\s*:\s*([01])', doc)
    if overall_mortality:
        a['survival_total'] += 1
        if overall_mortality.group(1) == '0': a['alive'] += 1
        else: a['dead_disease'] += 1

    op_date = re.search(r'Op_Date\s*:\s*(\d{4})', doc)
    if op_date:
        a['surgery_years'].append(int(op_date.group(1)))


def _parse_liver(doc: str, meta: dict, accum: dict):
    """간암 필드 파싱"""
    a = accum

    age = re.search(r'나이\s*\(진단시\)\s*:\s*(\d+)', doc)
    if age:
        a['ages'].append(int(age.group(1)))

    gender = re.search(r'성별\s*:\s*([12])', doc)
    if gender:
        a['genders'].append('Male' if gender.group(1) == '1' else 'Female')

    grade = re.search(r'조직학적등급\s*:\s*([1234])', doc)
    if grade:
        a['hg_grades'].append(f'Grade {grade.group(1)}')

    size = re.search(r'암\s*size\s*\(c1\)_장경\s*:\s*(\d+(?:\.\d+)?)', doc)
    if size:
        a['tumor_sizes'].append(float(size.group(1)))

    tumor_num = re.search(r'암의\s*개수\s*:\s*(\d+)', doc)
    if tumor_num:
        n = int(tumor_num.group(1))
        a['tumor_numbers'].append('Single' if n == 1 else 'Multiple')

    hbsag = re.search(r'혈청\s*HBsAg\s*유무\s*:\s*([01])', doc)
    if hbsag:
        a['hbsag_total'] += 1
        if hbsag.group(1) == '1': a['hbsag_positive'] += 1

    hcv = re.search(r'혈청\s*anti-HCV\s*유무\s*:\s*([01])', doc)
    if hcv:
        a['hcv_total'] += 1
        if hcv.group(1) == '1': a['hcv_positive'] += 1

    recur = re.search(r'재발\s*\(간\)\s*여부\s*:\s*([01])', doc)
    if recur:
        a['recur_total'] += 1
        if recur.group(1) == '1': a['recur_axillary'] += 1

    distant = re.search(r'다른\s*장기로\s*전이\s*여부\s*:\s*([01])', doc)
    if distant and distant.group(1) == '1':
        a['distant_meta'] += 1

    survival = re.search(r'이\s*질병으로\s*사망여부\s*:\s*([01])', doc)
    if survival:
        a['survival_total'] += 1
        if survival.group(1) == '0': a['alive'] += 1
        else: a['dead_disease'] += 1

    biopsy_date = re.search(r'생검\s*\(수술\)\s*연월일\s*:\s*(\d{4})', doc)
    if biopsy_date:
        a['surgery_years'].append(int(biopsy_date.group(1)))


# ============================================================
# 공통 누산기 초기화
# ============================================================

def _make_accum():
    return {
        # 공통
        'ages': [], 'genders': [], 'tumor_sizes': [], 'surgery_years': [],
        'tumor_locations': [], 'tumor_numbers': [],
        'stages': [], 't_categories': [], 'n_categories': [],
        't_stages': [], 'n_stages': [],
        'histologic_types': [], 'surgery_types': [],
        'lvi_total': 0, 'lvi_positive': 0,
        'pni_total': 0, 'pni_positive': 0,
        'ln_total': 0, 'ln_positive': 0,
        'survival_total': 0, 'alive': 0, 'dead_disease': 0, 'dead_other': 0,
        'recur_total': 0, 'recur_axillary': 0, 'recur_site': 0,
        'distant_meta': 0,
        # breast
        'er_total': 0, 'er_positive': 0,
        'pr_total': 0, 'pr_positive': 0,
        'her2_total': 0, 'her2_positive': 0,
        'her2_ihc': [], 'ki67_values': [],
        'ng_grades': [], 'hg_grades': [],
        'adjuvant_endocrine': [], 'adjuvant_rtx': [],
        'neoadjuvant_ctx': [],
        # stomach
        'treatment_types': [], 'msi_statuses': [],
        'kras_statuses': [], 'braf_statuses': [],
        # thyroid
        'braf_total': 0, 'braf_positive': 0,
        'ete_statuses': [], 'lymphocytic_thyroiditis': [],
        # hrd
        'hrd_groups': [], 'hrd_scores': [],
        'brca1_total': 0, 'brca1_positive': 0,
        'brca2_total': 0, 'brca2_positive': 0,
        'tp53_total': 0, 'tp53_positive': 0,
        'chemo_cycles': [],
        # cervix
        'hpv_total': 0, 'hpv_positive': 0,
        'hpv16_positive': 0, 'hpv18_positive': 0, 'hpv_other_hr_positive': 0,
        'margin_total': 0, 'margin_positive': 0,
        # prostate
        'pgg_grades': [], 'bcr_total': 0, 'bcr_positive': 0,
        # liver
        'hbsag_total': 0, 'hbsag_positive': 0,
        'hcv_total': 0, 'hcv_positive': 0,
    }


# ============================================================
# 결과 포맷터
# ============================================================

def _format_stats(accum: dict, category: str, hospital_label: str, total_docs: int, unique_count: int) -> str:
    a = accum
    cat_label = CATEGORY_LABEL.get(category, category)
    lines = [f"=== {hospital_label} {cat_label} CRF 데이터 통계 ===",
             f"\n총 환자 수: {unique_count}명",
             f"총 문서 수: {total_docs}개"]

    def pct(pos, total):
        return f"{round(pos/total*100,1)}%" if total else "N/A"

    def mean_range(lst, unit=""):
        if not lst: return None
        return f"평균 {round(sum(lst)/len(lst),1)}{unit} (범위: {min(lst)}~{max(lst)}{unit}, n={len(lst)})"

    # 나이
    if a['ages']:
        lines.append(f"\n진단 시 나이: {mean_range(a['ages'], '세')}")

    # 성별
    if a['genders']:
        gc = Counter(a['genders'])
        lines.append("\n성별 분포:")
        for g, cnt in gc.items():
            lines.append(f"  - {g}: {cnt}명 ({pct(cnt, len(a['genders']))})")

    # 종양 크기
    if a['tumor_sizes']:
        unit = "cm" if category in ("stomach", "liver") else "mm"
        lines.append(f"\n종양 크기: {mean_range(a['tumor_sizes'], unit)}")

    # 종양 위치
    if a['tumor_locations']:
        lines.append("\n종양 위치:")
        for loc, cnt in Counter(a['tumor_locations']).items():
            lines.append(f"  - {loc}: {cnt}명")

    # 종양 개수
    if a['tumor_numbers']:
        lines.append("\n종양 개수:")
        for num, cnt in Counter(a['tumor_numbers']).items():
            lines.append(f"  - {num}: {cnt}명")

    # Stage
    if a['stages']:
        lines.append("\nStage 분포:")
        for s, cnt in sorted(Counter(a['stages']).items()):
            lines.append(f"  - {s}: {cnt}명")

    # T/N category (breast/thyroid)
    if a['t_categories']:
        lines.append("\nT category:")
        for t, cnt in sorted(Counter(a['t_categories']).items()):
            lines.append(f"  - {t}: {cnt}명")
    if a['n_categories']:
        lines.append("\nN category:")
        for n, cnt in sorted(Counter(a['n_categories']).items()):
            lines.append(f"  - {n}: {cnt}명")

    # T/N stage (stomach/colorectal)
    if a['t_stages']:
        lines.append("\nT stage:")
        for t, cnt in sorted(Counter(a['t_stages']).items()):
            lines.append(f"  - {t}: {cnt}명")
    if a['n_stages']:
        lines.append("\nN stage:")
        for n, cnt in sorted(Counter(a['n_stages']).items()):
            lines.append(f"  - {n}: {cnt}명")

    # 조직학적 타입
    if a['histologic_types']:
        lines.append("\n조직학적 타입:")
        for ht, cnt in Counter(a['histologic_types']).items():
            lines.append(f"  - {ht}: {cnt}명")

    # 수술 방법
    if a['surgery_types']:
        lines.append("\n수술/치료 방법:")
        for st, cnt in Counter(a['surgery_types']).items():
            lines.append(f"  - {st}: {cnt}명")

    # 수술 연도
    if a['surgery_years']:
        lines.append("\n수술/시술 연도 분포:")
        for yr, cnt in sorted(Counter(a['surgery_years']).items()):
            lines.append(f"  - {yr}: {cnt}명")

    # --- breast 전용 ---
    if category == 'breast':
        if a['er_total']:
            lines.append(f"\nER: 양성 {a['er_positive']}명 ({pct(a['er_positive'], a['er_total'])}) / 전체 {a['er_total']}명")
        if a['pr_total']:
            lines.append(f"PR: 양성 {a['pr_positive']}명 ({pct(a['pr_positive'], a['pr_total'])}) / 전체 {a['pr_total']}명")
        if a['her2_total']:
            lines.append(f"HER2: 양성 {a['her2_positive']}명 ({pct(a['her2_positive'], a['her2_total'])}) / 전체 {a['her2_total']}명")
        if a['her2_ihc']:
            lines.append("\nHER2 IHC 등급:")
            for g, cnt in sorted(Counter(a['her2_ihc']).items()):
                lines.append(f"  - {g}: {cnt}명")
        if a['ki67_values']:
            lines.append(f"\nKi-67: {mean_range(a['ki67_values'], '%')}")
        if a['ng_grades']:
            lines.append("\nNuclear Grade:")
            for g, cnt in sorted(Counter(a['ng_grades']).items()):
                lines.append(f"  - {g}: {cnt}명")
        if a['hg_grades']:
            lines.append("\nHistologic Grade:")
            for g, cnt in sorted(Counter(a['hg_grades']).items()):
                lines.append(f"  - {g}: {cnt}명")
        if a['neoadjuvant_ctx']:
            lines.append(f"\n수술 전 항암치료: {Counter(a['neoadjuvant_ctx'])}")
        if a['adjuvant_endocrine']:
            ac = Counter(a['adjuvant_endocrine'])
            yes = ac.get('1', 0) + ac.get('2', 0)
            lines.append(f"보조 호르몬 치료: 받음 {yes}명 / 전체 {len(a['adjuvant_endocrine'])}명")
        if a['adjuvant_rtx']:
            ac = Counter(a['adjuvant_rtx'])
            yes = ac.get('1', 0) + ac.get('2', 0)
            lines.append(f"보조 방사선 치료: 받음 {yes}명 / 전체 {len(a['adjuvant_rtx'])}명")

    # --- stomach 전용 ---
    if category == 'stomach':
        if a['her2_total']:
            lines.append(f"\nHER2 (3+ 양성): {a['her2_positive']}명 / 전체 {a['her2_total']}명 ({pct(a['her2_positive'], a['her2_total'])})")
        if a['treatment_types']:
            lines.append("\n치료 방법 (수술 vs ESD):")
            for t, cnt in Counter(a['treatment_types']).items():
                lines.append(f"  - {t}: {cnt}명")
        if a['msi_statuses']:
            lines.append("\nMSI 상태:")
            for m, cnt in Counter(a['msi_statuses']).items():
                lines.append(f"  - {m}: {cnt}명")

    # --- thyroid 전용 ---
    if category == 'thyroid':
        if a['braf_total']:
            lines.append(f"\nBRAF 변이: 양성 {a['braf_positive']}명 ({pct(a['braf_positive'], a['braf_total'])}) / 전체 {a['braf_total']}명")
        if a['ete_statuses']:
            lines.append("\nExtrathyroid Extension:")
            for e, cnt in Counter(a['ete_statuses']).items():
                lines.append(f"  - {e}: {cnt}명")
        if a['lymphocytic_thyroiditis']:
            lines.append("\nLymphocytic Thyroiditis:")
            for l, cnt in Counter(a['lymphocytic_thyroiditis']).items():
                lines.append(f"  - {l}: {cnt}명")

    # --- colorectal 전용 ---
    if category == 'colorectal':
        if a['msi_statuses']:
            lines.append("\nMSI 상태:")
            for m, cnt in Counter(a['msi_statuses']).items():
                lines.append(f"  - {m}: {cnt}명")
        if a['kras_statuses']:
            lines.append(f"\nKRAS 변이: {Counter(a['kras_statuses'])}")
        if a['braf_statuses']:
            lines.append(f"BRAF 변이: {Counter(a['braf_statuses'])}")

    # --- hrd 전용 ---
    if category == 'hrd':
        if a['hrd_groups']:
            gc = Counter(a['hrd_groups'])
            hrd_pos = gc.get(1, 0) + gc.get(2, 0)
            hrd_neg = gc.get(0, 0)
            lines.append(f"\nHRD 상태: 양성(1·2) {hrd_pos}명 / 음성(0) {hrd_neg}명 / 전체 {len(a['hrd_groups'])}명")
        if a['hrd_scores']:
            lines.append(f"HRD Score: {mean_range(a['hrd_scores'])}")
        if a['brca1_total']:
            lines.append(f"\nBRCA1 변이: {a['brca1_positive']}명 ({pct(a['brca1_positive'], a['brca1_total'])}) / {a['brca1_total']}명")
        if a['brca2_total']:
            lines.append(f"BRCA2 변이: {a['brca2_positive']}명 ({pct(a['brca2_positive'], a['brca2_total'])}) / {a['brca2_total']}명")
        if a['tp53_total']:
            lines.append(f"TP53 변이: {a['tp53_positive']}명 ({pct(a['tp53_positive'], a['tp53_total'])}) / {a['tp53_total']}명")
        if a['msi_statuses']:
            lines.append(f"\nMSI: {Counter(a['msi_statuses'])}")
        if a['neoadjuvant_ctx']:
            lines.append(f"\n선행항암치료: {Counter(a['neoadjuvant_ctx'])}")
        if a['chemo_cycles']:
            lines.append(f"1차 항암 횟수: {mean_range(a['chemo_cycles'], '회')}")

    # --- cervix 전용 ---
    if category == 'cervix':
        if a['hpv_total']:
            lines.append(f"\nHPV 양성: {a['hpv_positive']}명 ({pct(a['hpv_positive'], a['hpv_total'])}) / {a['hpv_total']}명")
            if a['hpv_positive']:
                lines.append(f"  - HPV 16: {a['hpv16_positive']}명")
                lines.append(f"  - HPV 18: {a['hpv18_positive']}명")
                lines.append(f"  - 기타 고위험군: {a['hpv_other_hr_positive']}명")
        if a['margin_total']:
            lines.append(f"\n절제연 양성: {a['margin_positive']}명 ({pct(a['margin_positive'], a['margin_total'])}) / {a['margin_total']}명")

    # --- prostate 전용 ---
    if category == 'prostate':
        if a['pgg_grades']:
            lines.append("\nPathologic Gleason Grade Group:")
            for g, cnt in sorted(Counter(a['pgg_grades']).items()):
                lines.append(f"  - {g}: {cnt}명")
        if a['bcr_total']:
            lines.append(f"\nBCR(생화학적 재발): {a['bcr_positive']}명 ({pct(a['bcr_positive'], a['bcr_total'])}) / {a['bcr_total']}명")

    # --- liver 전용 ---
    if category == 'liver':
        if a['hg_grades']:
            lines.append("\n조직학적 등급:")
            for g, cnt in sorted(Counter(a['hg_grades']).items()):
                lines.append(f"  - {g}: {cnt}명")
        if a['hbsag_total']:
            lines.append(f"\nHBsAg 양성(B형간염): {a['hbsag_positive']}명 ({pct(a['hbsag_positive'], a['hbsag_total'])}) / {a['hbsag_total']}명")
        if a['hcv_total']:
            lines.append(f"anti-HCV 양성(C형간염): {a['hcv_positive']}명 ({pct(a['hcv_positive'], a['hcv_total'])}) / {a['hcv_total']}명")

    # 공통 - LVI/PNI
    if a['lvi_total']:
        lines.append(f"\nLVI(림프혈관 침윤): 양성 {a['lvi_positive']}명 ({pct(a['lvi_positive'], a['lvi_total'])}) / {a['lvi_total']}명")
    if a['pni_total']:
        lines.append(f"PNI(신경주위 침윤): 양성 {a['pni_positive']}명 ({pct(a['pni_positive'], a['pni_total'])}) / {a['pni_total']}명")

    # 림프절 전이 (breast/thyroid)
    if a['ln_total']:
        lines.append(f"\n림프절 전이: {a['ln_positive']}명 ({pct(a['ln_positive'], a['ln_total'])}) / {a['ln_total']}명")

    # 재발
    if a['recur_total']:
        lines.append(f"\n재발: {a['recur_axillary']}명 / {a['recur_total']}명 ({pct(a['recur_axillary'], a['recur_total'])})")
    if a['distant_meta']:
        lines.append(f"원격 전이: {a['distant_meta']}명")
    if a['recur_site']:
        lines.append(f"수술 부위 재발: {a['recur_site']}명")

    # 생존
    if a['survival_total']:
        lines.append(f"\n생존 현황:")
        lines.append(f"  - 생존: {a['alive']}명 ({pct(a['alive'], a['survival_total'])})")
        lines.append(f"  - 질병 사망: {a['dead_disease']}명")
        if a['dead_other']:
            lines.append(f"  - 기타 사망: {a['dead_other']}명")
        lines.append(f"  - 전체: {a['survival_total']}명")

    return "\n".join(lines)


# ============================================================
# Mixin 클래스
# ============================================================

PARSER_MAP = {
    'breast': _parse_breast,
    'stomach': _parse_stomach,
    'thyroid': _parse_thyroid,
    'colorectal': _parse_colorectal,
    'hrd': _parse_hrd,
    'cervix': _parse_cervix,
    'prostate': _parse_prostate,
    'liver': _parse_liver,
}


class CRFStatisticsMixin:
    """CRF 데이터 통계 계산을 위한 Mixin 클래스 (8개 암종 통합)"""

    def calculate_crf_statistics(self, documents: list, metadatas: list, hospital_code: str = None) -> dict:
        """
        CRF 문서 리스트에서 암종별 통계 계산

        Returns:
            암종별 통계 딕셔너리 (formatted_text 포함)
        """
        logger.info(f"📊 통계 계산 시작: {len(documents)}개 문서")

        # 암종별로 문서 분리
        cat_docs = {}
        cat_metas = {}
        cat_records = {}

        for doc, meta in zip(documents, metadatas):
            cat = meta.get('category', 'unknown')
            st = meta.get('sheet_type', '')
            if st != 'data':
                continue
            if cat not in cat_docs:
                cat_docs[cat] = []
                cat_metas[cat] = []
                cat_records[cat] = set()
            cat_docs[cat].append(doc)
            cat_metas[cat].append(meta)
            rid = meta.get('record_id')
            if rid:
                cat_records[cat].add(rid)

        # 암종별 통계 계산
        results = {}
        all_texts = []

        for cat, docs in cat_docs.items():
            parser = PARSER_MAP.get(cat)
            if not parser:
                continue

            accum = _make_accum()
            hosp_counter = Counter()

            for doc, meta in zip(docs, cat_metas[cat]):
                parser(doc, meta, accum)
                hc = meta.get('hospital', '')
                if hc:
                    hosp_counter[hc] += 1

            unique_count = len(cat_records.get(cat, set())) or len(docs)
            hosp_label = _get_hospital_name(hospital_code, cat) if hospital_code else "전체 병원"

            text = _format_stats(accum, cat, hosp_label, len(docs), unique_count)

            # 병원별 건수 추가
            if hosp_counter:
                hospital_lines = ["\n병원별 건수:"]
                for hc, cnt in sorted(hosp_counter.items()):
                    hn = _get_hospital_name(hc, cat)
                    hospital_lines.append(f"  - {hn} (코드: {hc}): {cnt}명")
                text += "\n" + "\n".join(hospital_lines)

            results[cat] = {
                'accum': accum,
                'formatted_text': text,
                'total_docs': len(docs),
                'unique_count': unique_count,
                'hospital_counts': dict(hosp_counter),
            }
            all_texts.append(text)

        # 레거시 호환 반환값 (단일 암종인 경우 기존 형식 유지)
        if len(results) == 1:
            cat = list(results.keys())[0]
            r = results[cat]
            return {
                'total_patients': r['unique_count'],
                'total_documents': r['total_docs'],
                'hospital_name': "전체 병원",
                'formatted_text': r['formatted_text'],
                'category': cat,
                'multi_category': False,
                **r['accum'],
            }

        # 다중 암종
        combined_text = "\n\n".join(all_texts)
        return {
            'total_patients': sum(r['unique_count'] for r in results.values()),
            'total_documents': sum(r['total_docs'] for r in results.values()),
            'hospital_name': "전체 병원",
            'formatted_text': combined_text,
            'categories': list(results.keys()),
            'multi_category': True,
            'per_category': {cat: r['formatted_text'] for cat, r in results.items()},
        }

    def format_statistics_for_llm(self, stats: dict) -> str:
        """통계 딕셔너리를 LLM용 텍스트로 변환"""
        return stats.get('formatted_text', '통계 데이터 없음')

    def _get_hospital_name(self, hospital_code: str) -> str:
        """병원 코드 → 이름 (유방암 기본값)"""
        return _get_hospital_name(hospital_code, 'breast')

    def get_dataset_metadata(self, all_docs: list) -> dict:
        """CRF 데이터셋 메타 정보 반환"""
        metadata = {
            "total_records": len(all_docs),
            "hospitals": {},
            "data_collection_period": {},
            "available_fields": [],
            "record_id_range": {}
        }

        hospital_counts = {}
        all_dates = []
        field_set = set()
        record_ids = []

        for doc in all_docs:
            fields = doc.get("fields") if isinstance(doc, dict) else {}
            if not isinstance(fields, dict):
                fields = {}

            hospital = doc.get("hospital", "Unknown")
            hospital_counts[hospital] = hospital_counts.get(hospital, 0) + 1

            record_id = doc.get("record_id")
            if record_id:
                record_ids.append(record_id)

            surgery_date = (doc.get("수술연월일") or doc.get("surgery_date")
                            or fields.get("수술연월일") or fields.get("surgery_date"))
            if surgery_date:
                try:
                    if isinstance(surgery_date, str):
                        date_obj = datetime.strptime(surgery_date[:10], "%Y-%m-%d")
                        all_dates.append(date_obj)
                except: pass

            for key in doc.keys():
                if key not in ["hospital", "record_id", "sheet", "path_no"]:
                    field_set.add(key)
            for key in fields.keys():
                field_set.add(key)

        for hospital_code, count in hospital_counts.items():
            hospital_name = _get_hospital_name(hospital_code)
            metadata["hospitals"][hospital_name] = {"code": hospital_code, "count": count}

        if all_dates:
            metadata["data_collection_period"] = {
                "earliest": min(all_dates).strftime("%Y-%m-%d"),
                "latest": max(all_dates).strftime("%Y-%m-%d"),
                "total_days": (max(all_dates) - min(all_dates)).days
            }

        if record_ids:
            metadata["record_id_range"] = {
                "first": min(record_ids),
                "last": max(record_ids),
                "total": len(set(record_ids))
            }

        metadata["available_fields"] = sorted(list(field_set))
        return metadata

    def format_metadata_for_llm(self, metadata: dict) -> str:
        """메타데이터 → LLM용 텍스트"""
        lines = ["=== CRF 데이터셋 메타 정보 (8개 암종 통합) ===",
                 f"\n총 레코드 수: {metadata.get('total_records', 0)}개"]

        hospitals = metadata.get('hospitals', {})
        if hospitals:
            lines.append(f"\n수집 병원 목록 ({len(hospitals)}개):")
            for hospital_name, info in sorted(hospitals.items()):
                lines.append(f"  - {hospital_name} (코드: {info['code']}): {info['count']}개 레코드")

        period = metadata.get('data_collection_period', {})
        if period:
            lines.append(f"\n데이터 수집 기간:")
            lines.append(f"  - 가장 오래된 수술일: {period.get('earliest', 'N/A')}")
            lines.append(f"  - 가장 최신 수술일: {period.get('latest', 'N/A')}")
            lines.append(f"  - 총 수집 기간: 약 {period.get('total_days', 0)}일")

        record_range = metadata.get('record_id_range', {})
        if record_range:
            lines.append(f"\nRecord ID 범위:")
            lines.append(f"  - 첫 번째: {record_range.get('first', 'N/A')}")
            lines.append(f"  - 마지막: {record_range.get('last', 'N/A')}")
            lines.append(f"  - 고유 환자 수: {record_range.get('total', 0)}명")

        fields = metadata.get('available_fields', [])
        if fields:
            display_fields = fields[:30]
            lines.append(f"\n사용 가능한 데이터 필드 ({len(fields)}개, 상위 30개):")
            for field in display_fields:
                lines.append(f"  - {field}")
            if len(fields) > 30:
                lines.append(f"  ...외 {len(fields)-30}개")

        return "\n".join(lines)
