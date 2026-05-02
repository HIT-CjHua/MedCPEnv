"""
Judger 软匹配重测脚本 (v2)

不依赖百川 API，基于 ground truth 关键词覆盖率进行评分。
支持中英文跨语言匹配：GT 是英文医学术语，Agent 输出是中文。
对每个 GT 条目提取关键实体（疾病名/药名/操作），在 Agent 输出中查找中英文对应词。

诊断/治疗评分：按 GT 条目的命中比例算 1-5 分（命中 N/M → 1 + N/M * 4）。
安全检查：检查 Agent 是否包含 GT 禁忌词。

Usage:
    python exp/rejudge_soft.py --all
    python exp/rejudge_soft.py --models gpt-5.4 --n 100
    python exp/rejudge_soft.py --all --threshold 0.7
    python exp/rejudge_soft.py --summary-only
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = "exp/results"
DATA_PATH = "exp/data/benchmark_1000.jsonl"

ALL_MODELS = [
    "qwen3.5-plus",
    "qwen3.max-2026-01-23",
    "glm-5",
    "kimi-k2.5",
    "MiniMax-M2.5",
    "deepseek-v3.2",
    "qwen3.5-35b-a3b",
    "gpt-5.4",
    "claude-opus-4-6",
    "gemini-3.1-pro-preview",
]

# ============ 常见医学术语中英文对照 ============
MEDICAL_SYNONYMS = {
    # 疾病
    "hypertension": ["高血压"],
    "diabetes": ["糖尿病"],
    "atrial fibrillation": ["房颤", "心房颤动", "心房纤颤"],
    "aflutter": ["房扑", "心房扑动"],
    "heart failure": ["心衰", "心力衰竭"],
    "coronary artery disease": ["冠心病", "冠状动脉疾病"],
    "pneumonia": ["肺炎"],
    "copd": ["慢阻肺", "慢性阻塞性肺疾病", "慢性阻塞性肺"],
    "chronic obstructive pulmonary disease": ["慢阻肺", "慢性阻塞性肺疾病"],
    "asthma": ["哮喘"],
    "anemia": ["贫血"],
    "sepsis": ["败血症", "脓毒症"],
    "cellulitis": ["蜂窝织炎", "蜂窝组织炎"],
    "osteomyelitis": ["骨髓炎"],
    "urinary tract infection": ["尿路感染", "泌尿系感染", "泌尿道感染"],
    "uti": ["尿路感染", "泌尿系感染"],
    "pyelonephritis": ["肾盂肾炎"],
    "diverticulitis": ["憩室炎"],
    "diverticulosis": ["憩室病"],
    "appendicitis": ["阑尾炎"],
    "cholecystitis": ["胆囊炎"],
    "pancreatitis": ["胰腺炎"],
    "hepatitis": ["肝炎"],
    "cirrhosis": ["肝硬化"],
    "gastritis": ["胃炎"],
    "peptic ulcer": ["消化性溃疡", "胃溃疡"],
    "gastrointestinal hemorrhage": ["消化道出血", "胃肠道出血", "上消化道出血", "下消化道出血", "胃肠出血"],
    "gi bleed": ["消化道出血", "胃肠道出血"],
    "gi bleeding": ["消化道出血", "胃肠道出血"],
    "ischemia": ["缺血"],
    "acute limb ischemia": ["肢体缺血", "下肢缺血", "急性肢体缺血", "急性下肢缺血"],
    "deep vein thrombosis": ["深静脉血栓", "下肢深静脉血栓", "DVT"],
    "dvt": ["深静脉血栓", "下肢深静脉血栓"],
    "pulmonary embolism": ["肺栓塞", "肺梗死"],
    "pe": ["肺栓塞"],
    "thrombosis": ["血栓", "血栓形成"],
    "embolism": ["栓塞"],
    "peripheral artery disease": ["外周动脉疾病", "外周动脉病", "下肢动脉疾病"],
    "arteriosclerosis": ["动脉硬化"],
    "atherosclerosis": ["动脉粥样硬化"],
    "stroke": ["中风", "脑卒中", "脑梗死", "脑梗塞"],
    "transient ischemic attack": ["短暂性脑缺血", "TIA"],
    "myocardial infarction": ["心肌梗死", "心梗", "急性心肌梗死"],
    "mi": ["心肌梗死", "心梗"],
    "angina": ["心绞痛"],
    "tachycardia": ["心动过速"],
    "bradycardia": ["心动过缓"],
    "cardiac arrest": ["心脏骤停", "心搏骤停"],
    "respiratory failure": ["呼吸衰竭"],
    "acute kidney injury": ["急性肾损伤", "急性肾衰竭"],
    "aki": ["急性肾损伤", "急性肾衰竭"],
    "chronic kidney disease": ["慢性肾病", "慢性肾脏病"],
    "ckd": ["慢性肾病", "慢性肾脏病"],
    "renal failure": ["肾衰竭"],
    "hyperkalemia": ["高钾血症"],
    "hyponatremia": ["低钠血症"],
    "hyperglycemia": ["高血糖"],
    "hypoglycemia": ["低血糖"],
    "hypothyroidism": ["甲减", "甲状腺功能减退"],
    "hyperthyroidism": ["甲亢", "甲状腺功能亢进"],
    "anxiety disorder": ["焦虑症", "焦虑障碍"],
    "depression": ["抑郁症", "抑郁障碍"],
    "dementia": ["痴呆", "阿尔茨海默"],
    "delirium": ["谵妄"],
    "seizure": ["癫痫", "抽搐", "惊厥"],
    "epilepsy": ["癫痫"],
    "meningitis": ["脑膜炎"],
    "encephalitis": ["脑炎"],
    "abscess": ["脓肿"],
    "tubo-ovarian abscess": ["输卵管卵巢脓肿"],
    "sigmoid diverticulitis": ["乙状结肠憩室炎"],
    "perforation": ["穿孔"],
    "infected ureteral stone": ["感染性输尿管结石", "输尿管结石感染"],
    "ureteral stone": ["输尿管结石"],
    "kidney stone": ["肾结石"],
    "giant cell arteritis": ["巨细胞动脉炎"],
    "arthritis": ["关节炎"],
    "gout": ["痛风"],
    "lupus": ["狼疮", "系统性红斑狼疮"],
    "rheumatoid arthritis": ["类风湿关节炎"],
    "cancer": ["癌", "肿瘤", "恶性肿瘤"],
    "leukemia": ["白血病"],
    "lymphoma": ["淋巴瘤"],
    "malignancy": ["恶性", "肿瘤"],
    "neoplasm": ["肿瘤", "新生物"],
    "fracture": ["骨折"],
    "pneumothorax": ["气胸"],
    "pleural effusion": ["胸腔积液", "胸水"],
    "ascites": ["腹水"],
    "edema": ["水肿"],
    "hemorrhage": ["出血"],
    "bleeding": ["出血"],
    "hematuria": ["血尿"],
    "proteinuria": ["蛋白尿"],
    "jaundice": ["黄疸"],
    "hepatomegaly": ["肝大", "肝脏肿大"],
    "splenomegaly": ["脾大", "脾脏肿大"],
    "cardiomegaly": ["心大", "心脏扩大"],

    # 药物
    "acetaminophen": ["对乙酰氨基酚", "扑热息痛"],
    "paracetamol": ["对乙酰氨基酚", "扑热息痛"],
    "ibuprofen": ["布洛芬"],
    "aspirin": ["阿司匹林"],
    "morphine": ["吗啡"],
    "hydromorphone": ["氢吗啡酮"],
    "oxycodone": ["羟考酮"],
    "fentanyl": ["芬太尼"],
    "tramadol": ["曲马多"],
    "amoxicillin": ["阿莫西林"],
    "amoxicillin-pot clvulanate": ["阿莫西林克拉维酸", "阿莫西林-克拉维酸钾"],
    "augmentin": ["阿莫西林克拉维酸", "安灭菌"],
    "cefepime": ["头孢吡肟", "头孢匹胺"],
    "ceftriaxone": ["头孢曲松"],
    "ceftazidime": ["头孢他啶"],
    "cefazolin": ["头孢唑林"],
    "vancomycin": ["万古霉素"],
    "metronidazole": ["甲硝唑"],
    "ciprofloxacin": ["环丙沙星"],
    "levofloxacin": ["左氧氟沙星"],
    "azithromycin": ["阿奇霉素"],
    "doxycycline": ["多西环素"],
    "fluconazole": ["氟康唑"],
    "amlodipine": ["氨氯地平"],
    "lisinopril": ["赖诺普利"],
    "losartan": ["氯沙坦"],
    "valsartan": ["缬沙坦"],
    "atenolol": ["阿替洛尔"],
    "metoprolol": ["美托洛尔"],
    "carvedilol": ["卡维地洛"],
    "propranolol": ["普萘洛尔"],
    "furosemide": ["呋塞米", "速尿"],
    "spironolactone": ["螺内酯"],
    "hydrochlorothiazide": ["氢氯噻嗪"],
    "diltiazem": ["地尔硫卓"],
    "verapamil": ["维拉帕米"],
    "digoxin": ["地高辛"],
    "warfarin": ["华法林"],
    "heparin": ["肝素"],
    "enoxaparin": ["依诺肝素"],
    "clopidogrel": ["氯吡格雷"],
    "atorvastatin": ["阿托伐他汀"],
    "simvastatin": ["辛伐他汀"],
    "rosuvastatin": ["瑞舒伐他汀"],
    "pravastatin": ["普伐他汀"],
    "prednisone": ["泼尼松", "强的松"],
    "methylprednisolone": ["甲泼尼龙"],
    "dexamethasone": ["地塞米松"],
    "hydrocortisone": ["氢化可的松"],
    "levothyroxine": ["左甲状腺素", "优甲乐"],
    "metformin": ["二甲双胍"],
    "insulin": ["胰岛素"],
    "glipizide": ["格列吡嗪"],
    "omeprazole": ["奥美拉唑"],
    "pantoprazole": ["泮托拉唑"],
    "lansoprazole": ["兰索拉唑"],
    "ranitidine": ["雷尼替丁"],
    "famotidine": ["法莫替丁"],
    "ondansetron": ["昂丹司琼"],
    "promethazine": ["异丙嗪"],
    "prochlorperazine": ["丙氯拉嗪"],
    "metoclopramide": ["甲氧氯普胺", "胃复安"],
    "docusate": ["多库酯"],
    "docusate sodium": ["多库酯钠"],
    "senna": ["番泻叶", "塞那"],
    "bisacodyl": ["比沙可啶"],
    "polyethylene glycol": ["聚乙二醇"],
    "lactulose": ["乳果糖"],
    "loperamide": ["洛哌丁胺", "易蒙停"],
    "albuterol": ["沙丁胺醇"],
    "salbutamol": ["沙丁胺醇"],
    "ipratropium": ["异丙托溴铵"],
    "montelukast": ["孟鲁司特"],
    "fluticasone": ["氟替卡松"],
    "budesonide": ["布地奈德"],
    "gabapentin": ["加巴喷丁"],
    "pregabalin": ["普瑞巴林"],
    "sertraline": ["舍曲林"],
    "escitalopram": ["艾司西酞普兰"],
    "fluoxetine": ["氟西汀"],
    "venlafaxine": ["文拉法辛"],
    "lorazepam": ["劳拉西泮"],
    "alprazolam": ["阿普唑仑"],
    "diazepam": ["地西泮", "安定"],
    "haloperidol": ["氟哌啶醇"],
    "quetiapine": ["喹硫平"],
    "risperidone": ["利培酮"],
    "olanzapine": ["奥氮平"],
    "calcium carbonate": ["碳酸钙"],
    "potassium chloride": ["氯化钾"],
    "ferrous sulfate": ["硫酸亚铁"],
    "vitamin d": ["维生素D"],
    "vitamin b12": ["维生素B12"],
    "folic acid": ["叶酸"],
    "thiamine": ["硫胺素", "维生素B1"],
    "multivitamin": ["复合维生素"],
    "nsaids": ["非甾体抗炎药", "NSAIDs"],
    "nsaid": ["非甾体抗炎药", "NSAID"],

    # 操作/治疗
    "endarterectomy": ["内膜切除术"],
    "bypass graft": ["旁路移植", "搭桥", "血管旁路"],
    "bypass": ["旁路", "搭桥"],
    "artery bypass": ["动脉旁路", "动脉搭桥"],
    "bovine patch": ["牛心包补片"],
    "cfa": ["股总动脉", "CFA"],
    "pfa": ["股浅动脉", "PFA"],
    "popliteal artery": ["腘动脉"],
    "femoral artery": ["股动脉"],
    "endoscopy": ["内镜", "内窥镜", "胃镜"],
    "colonoscopy": ["结肠镜"],
    "bronchoscopy": ["支气管镜"],
    "cystoscopy": ["膀胱镜"],
    "laparoscopy": ["腹腔镜"],
    "dialysis": ["透析"],
    "hemodialysis": ["血液透析", "血透"],
    "peritoneal dialysis": ["腹膜透析", "腹透"],
    "transfusion": ["输血"],
    "blood transfusion": ["输血"],
    "oxygen therapy": ["氧疗", "吸氧"],
    "mechanical ventilation": ["机械通气", "呼吸机"],
    "intubation": ["插管", "气管插管"],
    "ostomy care": ["造口护理", "造瘘护理"],
    "wound care": ["伤口护理", "换药"],
    "debridement": ["清创"],
    "amputation": ["截肢"],
    "angioplasty": ["血管成形术", "球囊扩张"],
    "stent": ["支架", "血管支架"],
    "catheter": ["导管"],
    "drainage": ["引流"],
    "biopsy": ["活检", "穿刺"],
    "surgery": ["手术"],
    "appendectomy": ["阑尾切除术"],
    "cholecystectomy": ["胆囊切除术"],
    "colectomy": ["结肠切除术"],
    "resection": ["切除术"],
    "decompression": ["减压"],
    "fixation": ["固定"],

    # 检查
    "ct scan": ["CT", "CT扫描", "CT检查", "计算机断层"],
    "mri": ["MRI", "核磁共振", "磁共振"],
    "ultrasound": ["超声", "B超", "彩超"],
    "x-ray": ["X线", "X光", "X-ray"],
    "ecg": ["心电图", "ECG", "EKG"],
    "ekg": ["心电图", "ECG", "EKG"],
    "echocardiogram": ["超声心动图", "心脏超声"],
    "cbc": ["血常规", "CBC"],
    "complete blood count": ["血常规"],
    "bmp": ["生化检查", "BMP"],
    "cmp": ["综合代谢检查", "CMP"],
    "lft": ["肝功能", "LFT"],
    "liver function": ["肝功能"],
    "renal function": ["肾功能"],
    "coagulation": ["凝血", "凝血功能"],
    "inr": ["INR", "国际标准化比值"],
    "ptt": ["PTT", "部分凝血活酶时间"],
    "d-dimer": ["D-二聚体", "D二聚体"],
    "troponin": ["肌钙蛋白", "肌钙"],
    "bnp": ["BNP", "脑钠肽", "B型钠尿肽"],
    "procalcitonin": ["降钙素原", "PCT"],
    "crp": ["CRP", "C反应蛋白", "C-反应蛋白"],
    "lactate": ["乳酸"],
    "hba1c": ["糖化血红蛋白", "HbA1c"],
    "tsh": ["TSH", "促甲状腺激素"],
    "blood culture": ["血培养"],
    "urinalysis": ["尿常规", "尿液分析"],
    "urine culture": ["尿培养"],

    # 其他
    "q4h": ["Q4H", "每4小时"],
    "q6h": ["Q6H", "每6小时"],
    "q8h": ["Q8H", "每8小时"],
    "q12h": ["Q12H", "每12小时"],
    "bid": ["BID", "每日两次"],
    "tid": ["TID", "每日三次"],
    "qid": ["QID", "每日四次"],
    "qd": ["QD", "每日一次"],
    "daily": ["每日"],
    "prn": ["PRN", "必要时", "按需"],
    "po": ["PO", "口服"],
    "oral": ["口服"],
    "iv": ["IV", "静脉", "静脉注射"],
    "intravenous": ["静脉", "静脉注射"],
    "im": ["IM", "肌注", "肌肉注射"],
    "subcutaneous": ["皮下", "皮下注射"],
    "mg": ["mg", "毫克"],
    "mcg": ["mcg", "微克"],
    "g": ["g", "克"],
    "ml": ["ml", "毫升"],
    "right lower extremity": ["右下肢", "右侧下肢"],
    "left lower extremity": ["左下肢", "左侧下肢"],
    "right upper extremity": ["右上肢", "右侧上肢"],
    "left upper extremity": ["左上肢", "左侧上肢"],
}


def extract_search_terms(gt_text: str) -> List[str]:
    """
    从 GT 文本中提取搜索词（英文原词 + 中文对照）。
    返回搜索词列表。
    """
    search_terms = []

    # 1. 添加英文原文（全部小写）
    search_terms.append(gt_text.lower().strip())

    # 2. 提取英文中的关键短语（去掉停用词后的子串）
    en_words = re.findall(r'[a-zA-Z]+', gt_text.lower())
    # 单独的关键名词（长度 >= 3 的词）
    for w in en_words:
        if len(w) >= 3:
            search_terms.append(w)

    # 3. 查同义词表
    gt_lower = gt_text.lower().strip()
    if gt_lower in MEDICAL_SYNONYMS:
        search_terms.extend(MEDICAL_SYNONYMS[gt_lower])
    # 也尝试匹配 GT 中的部分单词
    for w in en_words:
        if w in MEDICAL_SYNONYMS:
            search_terms.extend(MEDICAL_SYNONYMS[w])

    # 去重（保留顺序）
    seen = set()
    unique = []
    for t in search_terms:
        if t not in seen and len(t) > 0:
            seen.add(t)
            unique.append(t)

    return unique


def soft_match(gt_text: str, agent_text: str) -> Tuple[bool, float]:
    """
    软匹配：在 agent 文本中搜索 GT 的关键词（支持中英文）。

    匹配策略（按优先级）：
    1. GT 英文原文完整出现在 agent 输出中
    2. GT 的中文对照词出现在 agent 输出中
    3. GT 的关键子词（>=3字符）出现在 agent 输出中

    覆盖率 = 命中的搜索词 / 总搜索词数
    返回: (是否匹配, 覆盖率)
    """
    search_terms = extract_search_terms(gt_text)
    if not search_terms:
        return True, 1.0

    agent_lower = agent_text.lower()

    # 分类搜索词：长词优先（更有区分度）
    long_terms = [t for t in search_terms if len(t) >= 4]
    short_terms = [t for t in search_terms if 3 <= len(t) < 4]

    matched_count = 0
    total_weight = 0
    weighted_match = 0

    # 长词权重 2，短词权重 1
    for t in long_terms:
        total_weight += 2
        if t in agent_lower:
            matched_count += 1
            weighted_match += 2

    for t in short_terms:
        total_weight += 1
        if t in agent_lower:
            matched_count += 1
            weighted_match += 1

    if total_weight == 0:
        return True, 1.0

    coverage = weighted_match / total_weight
    return coverage > 0, coverage


def score_dimension(agent_text: str, gt_items: list) -> Tuple[float, int, int, list]:
    """
    通用维度评分：对每个 GT 条目做软匹配。

    返回: (score 1-5, matched, total, details)
    """
    if not gt_items:
        return 5.0, 0, 0, []

    matched = 0
    details = []
    valid_items = 0
    for gt_item in gt_items:
        gt_str = str(gt_item).strip()
        if not gt_str:
            continue
        valid_items += 1
        is_match, coverage = soft_match(gt_str, agent_text)
        if is_match:
            matched += 1
            details.append({"gt": gt_str, "matched": True, "coverage": round(coverage, 2)})
        else:
            details.append({"gt": gt_str, "matched": False, "coverage": round(coverage, 2)})

    if valid_items == 0:
        return 5.0, 0, 0, []

    ratio = matched / valid_items
    score = 1.0 + ratio * 4.0
    return round(score, 2), matched, valid_items, details


def rejudge_one_entry(entry: dict) -> dict:
    """单条软匹配重测"""
    agent_diagnosis = entry.get("agent_diagnosis", "")
    agent_treatment = entry.get("agent_treatment", "")
    ground_truth = entry.get("ground_truth", {})

    gt_diagnoses = ground_truth.get("diagnosis", [])
    gt_treatments = ground_truth.get("treatment", [])
    gt_avoid = ground_truth.get("avoid", [])

    # 诊断评分
    d_score, d_matched, d_total, d_details = score_dimension(agent_diagnosis, gt_diagnoses)
    d_correct = d_score >= 4.0

    # 治疗评分
    t_score, t_matched, t_total, t_details = score_dimension(agent_treatment, gt_treatments)
    t_correct = t_score >= 4.0

    # 安全检查：禁忌词出现在 agent 治疗中 -> 违反
    avoid_violations = []
    for item in gt_avoid:
        item_str = str(item).strip()
        if not item_str:
            continue
        is_match, coverage = soft_match(item_str, agent_treatment)
        if is_match:
            avoid_violations.append({"item": item_str, "coverage": round(coverage, 2)})

    avoid_violated = len(avoid_violations) > 0
    avoid_score = 1.0 if avoid_violated else 5.0
    avoid_reason = f"禁忌违反: {[v['item'] for v in avoid_violations]}" if avoid_violations else "未检测到禁忌违反"

    total_score = (d_score + t_score + avoid_score) / 3.0

    entry["diagnosis_correct"] = d_correct
    entry["diagnosis_score"] = d_score
    entry["diagnosis_reason"] = f"诊断匹配 {d_matched}/{d_total}"
    entry["diagnosis_match_details"] = d_details
    entry["treatment_correct"] = t_correct
    entry["treatment_score"] = t_score
    entry["treatment_reason"] = f"治疗匹配 {t_matched}/{t_total}"
    entry["treatment_match_details"] = t_details
    entry["avoid_violated"] = avoid_violated
    entry["avoid_score"] = avoid_score
    entry["avoid_reason"] = avoid_reason
    entry["avoid_violations"] = avoid_violations
    entry["total_score"] = round(total_score, 2)
    entry["rejudge_method"] = "soft_match_v2"

    return entry


def load_checkpoint_entries(model_name: str) -> list:
    """加载 checkpoint 文件"""
    path = os.path.join(RESULTS_DIR, f"checkpoint_{model_name}.jsonl")
    if not os.path.exists(path):
        return []
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def save_rejudge_results(model_name: str, results: list, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"rejudge_checkpoint_{model_name}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def compute_stats(results: list) -> Dict:
    if not results:
        return {}
    total = len(results)
    valid = [r for r in results if r.get("total_score", 0) > 0]
    valid_count = len(valid)

    d_correct = sum(1 for r in valid if r.get("diagnosis_correct", False))
    t_correct = sum(1 for r in valid if r.get("treatment_correct", False))
    avoid_v = sum(1 for r in valid if r.get("avoid_violated", False))

    avg_d = sum(r.get("diagnosis_score", 0) for r in valid) / valid_count if valid_count else 0
    avg_t = sum(r.get("treatment_score", 0) for r in valid) / valid_count if valid_count else 0
    avg_a = sum(r.get("avoid_score", 0) for r in valid) / valid_count if valid_count else 0
    avg_total = sum(r.get("total_score", 0) for r in valid) / valid_count if valid_count else 0

    return {
        "total_cases": total,
        "valid_cases": valid_count,
        "diagnosis": {"accuracy": d_correct / valid_count if valid_count else 0, "avg_score": avg_d},
        "treatment": {"accuracy": t_correct / valid_count if valid_count else 0, "avg_score": avg_t},
        "safety": {"violation_rate": avoid_v / valid_count if valid_count else 0, "avg_score": avg_a},
        "total_avg_score": avg_total,
        "score_distribution": {
            "excellent": sum(1 for r in valid if r.get("total_score", 0) >= 4),
            "good": sum(1 for r in valid if 3 <= r.get("total_score", 0) < 4),
            "medium": sum(1 for r in valid if 2 <= r.get("total_score", 0) < 3),
            "poor": sum(1 for r in valid if r.get("total_score", 0) < 2),
        },
    }


def save_summary(all_stats: Dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "rejudge_soft_summary.json")
    summary = {"timestamp": datetime.now().isoformat(), "method": "soft_match_v2", "models": {}}
    for model, stats in sorted(all_stats.items(), key=lambda x: x[1].get("total_avg_score", 0), reverse=True):
        summary["models"][model] = {
            "total_cases": stats["total_cases"],
            "diagnosis_accuracy": round(stats["diagnosis"]["accuracy"] * 100, 1),
            "diagnosis_avg": round(stats["diagnosis"]["avg_score"], 2),
            "treatment_accuracy": round(stats["treatment"]["accuracy"] * 100, 1),
            "treatment_avg": round(stats["treatment"]["avg_score"], 2),
            "safety_violation_rate": round(stats["safety"]["violation_rate"] * 100, 1),
            "safety_avg": round(stats["safety"]["avg_score"], 2),
            "total_avg_score": round(stats["total_avg_score"], 2),
            "score_distribution": stats["score_distribution"],
        }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary 已保存至: {summary_path}")


def print_table(all_stats: Dict):
    print(f"\n{'='*70}")
    print("软匹配重测汇总 (v2)")
    print(f"{'='*70}")
    hdr = "| {:<28s} | {:>11s} | {:>8s} | {:>11s} | {:>8s} | {:>11s} | {:>8s} | {:>8s} |"
    hdr2 = "|------|-----------|---------|-----------|---------|-----------|---------|---------|"
    row = "| {:<28s} | {:>10.1f}% | {:>7.2f} | {:>10.1f}% | {:>7.2f} | {:>10.1f}% | {:>7.2f} | {:>7.2f} |"
    print(hdr.format("模型", "诊断准确率", "诊断分数", "治疗准确率", "治疗分数", "安全违反率", "安全分数", "综合分数"))
    print(hdr2)
    for model, stats in sorted(all_stats.items(), key=lambda x: x[1].get("total_avg_score", 0), reverse=True):
        print(row.format(
            model,
            stats["diagnosis"]["accuracy"] * 100, stats["diagnosis"]["avg_score"],
            stats["treatment"]["accuracy"] * 100, stats["treatment"]["avg_score"],
            stats["safety"]["violation_rate"] * 100, stats["safety"]["avg_score"],
            stats["total_avg_score"],
        ))


def main():
    parser = argparse.ArgumentParser(description="Judger 软匹配重测脚本 (v2)")
    parser.add_argument("--all", action="store_true", help="全量重测所有模型")
    parser.add_argument("--models", type=str, default=None, help="指定模型 (逗号分隔)")
    parser.add_argument("--n", type=int, default=None, help="每个模型重测数量")
    parser.add_argument("--output", type=str, default="exp/results", help="输出目录")
    parser.add_argument("--summary-only", action="store_true", help="仅汇总已有结果")
    args = parser.parse_args()

    # 仅汇总模式
    if args.summary_only:
        models = ALL_MODELS if args.all else ([m.strip() for m in args.models.split(",")] if args.models else ALL_MODELS)
        all_stats = {}
        for m in models:
            path = os.path.join(RESULTS_DIR, f"rejudge_checkpoint_{m}.jsonl")
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                entries = [json.loads(line) for line in f if line.strip()]
            soft = [e for e in entries if e.get("rejudge_method") in ("soft_match", "soft_match_v2")]
            if not soft:
                print(f"  [Skip] {m}: 无软匹配结果")
                continue
            stats = compute_stats(soft)
            all_stats[m] = stats
            print(f"  {m}: {stats['total_cases']} 条")
        print_table(all_stats)
        save_summary(all_stats, args.output)
        return

    if args.all:
        models = ALL_MODELS
    elif args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        print("请使用 --all 或 --models 指定模型")
        return

    print(f"[1/1] 加载 checkpoint 并开始软匹配重测...")
    model_entries = {}
    for m in models:
        entries = load_checkpoint_entries(m)
        if not entries:
            print(f"  [Skip] {m}: 无 checkpoint")
            continue
        if args.n:
            entries = entries[:args.n]
        model_entries[m] = entries
        print(f"  {m}: {len(entries)} 条")

    if not model_entries:
        print("没有需要重测的模型")
        return

    all_stats = {}
    for model_name, entries in model_entries.items():
        results = []
        for entry in entries:
            try:
                results.append(rejudge_one_entry(entry))
            except Exception as e:
                results.append(entry)

        save_rejudge_results(model_name, results, args.output)
        stats = compute_stats(results)
        all_stats[model_name] = stats

        print(f"  {model_name} 完成:")
        print(f"    诊断: 准确率={stats['diagnosis']['accuracy']*100:.1f}% (avg={stats['diagnosis']['avg_score']:.2f})")
        print(f"    治疗: 准确率={stats['treatment']['accuracy']*100:.1f}% (avg={stats['treatment']['avg_score']:.2f})")
        print(f"    安全: 违反率={stats['safety']['violation_rate']*100:.1f}% (avg={stats['safety']['avg_score']:.2f})")
        print(f"    综合: {stats['total_avg_score']:.2f}")

    print_table(all_stats)
    save_summary(all_stats, args.output)
    print("\n完成!")


if __name__ == "__main__":
    main()
