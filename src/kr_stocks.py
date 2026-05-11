"""
Korean Stock Support: 한국 주식(KRX) 매핑 및 yfinance 연동

기능:
    1. 한글 종목명 → yfinance 티커 자동 변환 (예: "삼성전자" → "005930.KS")
    2. 종목코드 → yfinance 포맷 변환 (예: "005930" → "005930.KS")
    3. 한국 주요 종목 섹터 매핑 내장
    4. KOSPI/KOSDAQ 벤치마크 지원
"""

from typing import Dict, Optional, Tuple

# =============================================================================
# 한국 주요 종목 매핑 테이블 (Top 80+ 종목)
# =============================================================================

# 종목명 → (종목코드, 섹터)  — 시가총액 상위 100종목 (2025년 기준)
KR_STOCK_MAP: Dict[str, Tuple[str, str]] = {
    # ── 반도체/전자 ──
    "삼성전자": ("005930", "반도체"),
    "삼성전자우": ("005935", "반도체"),
    "SK하이닉스": ("000660", "반도체"),
    "삼성SDI": ("006400", "2차전지"),
    "삼성전기": ("009150", "전자부품"),
    "LG이노텍": ("011070", "전자부품"),
    "DB하이텍": ("000990", "반도체"),
    "리노공업": ("058470", "반도체장비"),
    "한미반도체": ("042700", "반도체장비"),
    "이수페타시스": ("007660", "전자부품"),

    # ── 2차전지 ──
    "LG에너지솔루션": ("373220", "2차전지"),
    "에코프로비엠": ("247540", "2차전지"),
    "에코프로": ("086520", "2차전지"),
    "포스코퓨처엠": ("003670", "2차전지"),
    "엘앤에프": ("066970", "2차전지"),
    "SK아이이테크놀로지": ("361610", "2차전지"),

    # ── 자동차 ──
    "현대차": ("005380", "자동차"),
    "기아": ("000270", "자동차"),
    "현대모비스": ("012330", "자동차부품"),
    "현대위아": ("011210", "자동차부품"),
    "만도": ("204320", "자동차부품"),
    "HL만도": ("204320", "자동차부품"),

    # ── 바이오/헬스케어 ──
    "삼성바이오로직스": ("207940", "바이오"),
    "셀트리온": ("068270", "바이오"),
    "SK바이오팜": ("326030", "바이오"),
    "유한양행": ("000100", "제약"),
    "녹십자": ("006280", "제약"),
    "삼성바이오에피스": ("206640", "바이오"),
    "알테오젠": ("196170", "바이오"),
    "HLB": ("028300", "바이오"),
    "리가켐바이오": ("141080", "바이오"),
    "씨젠": ("096530", "진단키트"),
    "SK바이오사이언스": ("302440", "바이오"),

    # ── IT/플랫폼 ──
    "네이버": ("035420", "IT/플랫폼"),
    "카카오": ("035720", "IT/플랫폼"),
    "카카오뱅크": ("323410", "핀테크"),
    "카카오페이": ("377300", "핀테크"),
    "크래프톤": ("259960", "게임"),
    "엔씨소프트": ("036570", "게임"),
    "펄어비스": ("263750", "게임"),
    "넷마블": ("251270", "게임"),
    "위메이드": ("112040", "게임"),
    "더존비즈온": ("012510", "소프트웨어"),
    "카카오게임즈": ("293490", "게임"),

    # ── 금융 ──
    "KB금융": ("105560", "금융"),
    "신한지주": ("055550", "금융"),
    "하나금융지주": ("086790", "금융"),
    "우리금융지주": ("316140", "금융"),
    "삼성생명": ("032830", "보험"),
    "삼성화재": ("000810", "보험"),
    "미래에셋증권": ("006800", "증권"),
    "한국투자금융지주": ("071050", "증권"),
    "메리츠금융지주": ("138040", "금융"),
    "DB손해보험": ("005830", "보험"),
    "NH투자증권": ("005940", "증권"),

    # ── 화학/소재 ──
    "LG화학": ("051910", "화학"),
    "롯데케미칼": ("011170", "화학"),
    "포스코홀딩스": ("005490", "철강"),
    "현대제철": ("004020", "철강"),
    "고려아연": ("010130", "비철금속"),
    "금양": ("001570", "화학"),
    "OCI홀딩스": ("010060", "화학"),
    "한화솔루션": ("009830", "화학"),

    # ── 건설/인프라 ──
    "현대건설": ("000720", "건설"),
    "삼성물산": ("028260", "건설/지주"),
    "대우건설": ("047040", "건설"),
    "GS건설": ("006360", "건설"),
    "HDC현대산업개발": ("294870", "건설"),
    "DL이앤씨": ("375500", "건설"),

    # ── 에너지 ──
    "SK이노베이션": ("096770", "에너지"),
    "S-Oil": ("010950", "에너지"),
    "한국전력": ("015760", "유틸리티"),
    "한국가스공사": ("036460", "유틸리티"),
    "한전KPS": ("051600", "유틸리티"),

    # ── 통신 ──
    "SK텔레콤": ("017670", "통신"),
    "KT": ("030200", "통신"),
    "LG유플러스": ("032640", "통신"),
    "SK스퀘어": ("402340", "통신/지주"),

    # ── 식품/소비재 ──
    "CJ제일제당": ("097950", "식품"),
    "오뚜기": ("007310", "식품"),
    "아모레퍼시픽": ("090430", "화장품"),
    "LG생활건강": ("051900", "화장품"),
    "호텔신라": ("008770", "여행/레저"),
    "하이트진로": ("000080", "식품"),
    "오리온": ("271560", "식품"),
    "농심": ("004370", "식품"),
    "CJ ENM": ("035760", "엔터/미디어"),
    "HYBE": ("352820", "엔터/미디어"),
    "하이브": ("352820", "엔터/미디어"),
    "JYP Ent.": ("035900", "엔터/미디어"),
    "SM": ("041510", "엔터/미디어"),
    "에스엠": ("041510", "엔터/미디어"),

    # ── 유통 ──
    "신세계": ("004170", "유통"),
    "이마트": ("139480", "유통"),
    "롯데쇼핑": ("023530", "유통"),
    "BGF리테일": ("282330", "유통"),
    "쿠팡": ("CPNG", "유통"),  # NYSE 상장

    # ── 방산/조선/항공 ──
    "한화에어로스페이스": ("012450", "방산"),
    "한화오션": ("042660", "조선"),
    "HD한국조선해양": ("009540", "조선"),
    "현대중공업": ("329180", "조선"),
    "대한항공": ("003490", "항공"),
    "HMM": ("011200", "해운"),
    "한화시스템": ("272210", "방산"),
    "LIG넥스원": ("079550", "방산"),

    # ── 지주/기타 ──
    "SK": ("034730", "지주"),
    "LG": ("003550", "지주"),
    "CJ": ("001040", "지주"),
    "두산에너빌리티": ("034020", "중공업"),
    "한화": ("000880", "지주"),
    "GS": ("078930", "지주"),
    "두산밥캣": ("241560", "기계"),
    "HD현대": ("267250", "지주"),
}

# 종목코드 → 종목명 역방향 매핑
KR_CODE_TO_NAME: Dict[str, str] = {
    code: name for name, (code, _) in KR_STOCK_MAP.items()
}

# 종목코드 → 섹터
KR_CODE_TO_SECTOR: Dict[str, str] = {
    code: sector for _, (code, sector) in KR_STOCK_MAP.items()
}

# 영문명 / 약칭 → 한글 종목명 매핑 (대소문자 무시)
KR_ENGLISH_ALIAS: Dict[str, str] = {
    # 대형주
    "samsung": "삼성전자", "samsung electronics": "삼성전자",
    "sk hynix": "SK하이닉스", "hynix": "SK하이닉스",
    "hyundai": "현대차", "hyundai motor": "현대차",
    "kia": "기아", "kia motors": "기아",
    "naver": "네이버",
    "kakao": "카카오",
    "kakaobank": "카카오뱅크", "kakao bank": "카카오뱅크",
    "kakaopay": "카카오페이", "kakao pay": "카카오페이",
    "celltrion": "셀트리온",
    "samsung bio": "삼성바이오로직스", "samsung biologics": "삼성바이오로직스",
    "samsung sdi": "삼성SDI",
    "lg chem": "LG화학", "lg chemical": "LG화학",
    "lg energy": "LG에너지솔루션", "lg energy solution": "LG에너지솔루션",
    "posco": "포스코홀딩스", "posco holdings": "포스코홀딩스",
    "kb financial": "KB금융", "kb": "KB금융",
    "shinhan": "신한지주", "shinhan financial": "신한지주",
    "hana financial": "하나금융지주",
    "ncsoft": "엔씨소프트", "nc soft": "엔씨소프트",
    "krafton": "크래프톤",
    "netmarble": "넷마블",
    "pearl abyss": "펄어비스",
    "sk telecom": "SK텔레콤", "skt": "SK텔레콤",
    "kt": "KT",
    "lg uplus": "LG유플러스", "lg u+": "LG유플러스",
    "coupang": "쿠팡",
    "hybe": "HYBE",
    "sm entertainment": "SM", "sm ent": "SM",
    "jyp": "JYP Ent.", "jyp entertainment": "JYP Ent.",
    "korean air": "대한항공",
    "kepco": "한국전력", "korea electric": "한국전력",
    "hyundai heavy": "현대중공업", "hhi": "현대중공업",
    "hanwha aerospace": "한화에어로스페이스", "hanwha": "한화",
    "hanwha ocean": "한화오션",
    "doosan": "두산에너빌리티", "doosan enerbility": "두산에너빌리티",
    "amorepacific": "아모레퍼시픽", "amore": "아모레퍼시픽",
    "lg household": "LG생활건강", "lg h&h": "LG생활건강",
    "samsung fire": "삼성화재", "samsung life": "삼성생명",
    "samsung engineering": "삼성물산",
    "ecopro": "에코프로", "ecopro bm": "에코프로비엠",
    "alteogen": "알테오젠",
    "sk innovation": "SK이노베이션",
    "sk square": "SK스퀘어",
    "meritz": "메리츠금융지주", "meritz financial": "메리츠금융지주",
    "hyundai mobis": "현대모비스",
    "hyundai steel": "현대제철",
    "hyundai construction": "현대건설",
    "lotte chemical": "롯데케미칼",
    "korea zinc": "고려아연",
    "s-oil": "S-Oil", "s oil": "S-Oil",
    "hmm": "HMM",
    "lg innotek": "LG이노텍",
    "samsung electro": "삼성전기", "samsung electro-mechanics": "삼성전기",
    "hd hyundai": "HD현대", "hd korea shipbuilding": "HD한국조선해양",
    "sk biopharm": "SK바이오팜", "sk bio": "SK바이오팜",
    "yuhan": "유한양행",
    "shinsegae": "신세계",
    "orion": "오리온",
    "nongshim": "농심",
    "ottogi": "오뚜기",
    "lig nex1": "LIG넥스원", "lignex1": "LIG넥스원",
    "db hitek": "DB하이텍",
    "woori financial": "우리금융지주",
}

# 글로벌 주식 한글 별칭 → yfinance 티커 (한글로 해외 주식 검색용)
GLOBAL_KOREAN_ALIAS: Dict[str, Tuple[str, str]] = {
    # Mega Cap
    "애플": ("AAPL", "Technology"),
    "마이크로소프트": ("MSFT", "Technology"),
    "엔비디아": ("NVDA", "Technology"),
    "테슬라": ("TSLA", "Consumer Cyclical"),
    "아마존": ("AMZN", "Consumer Cyclical"),
    "구글": ("GOOGL", "Technology"),
    "알파벳": ("GOOGL", "Technology"),
    "메타": ("META", "Technology"),
    "페이스북": ("META", "Technology"),
    "넷플릭스": ("NFLX", "Communication"),
    "브로드컴": ("AVGO", "Technology"),
    "오라클": ("ORCL", "Technology"),
    "세일즈포스": ("CRM", "Technology"),
    "어도비": ("ADBE", "Technology"),
    "퀄컴": ("QCOM", "Technology"),
    "인텔": ("INTC", "Technology"),
    "마이크론": ("MU", "Technology"),
    "팔란티어": ("PLTR", "Technology"),
    "우버": ("UBER", "Technology"),
    "쇼피파이": ("SHOP", "Technology"),
    "스노우플레이크": ("SNOW", "Technology"),
    "서비스나우": ("NOW", "Technology"),
    "시스코": ("CSCO", "Technology"),
    "ARM": ("ARM", "Technology"),
    # Financial
    "버크셔해서웨이": ("BRK-B", "Financial"),
    "JP모건": ("JPM", "Financial"),
    "제이피모건": ("JPM", "Financial"),
    "비자": ("V", "Financial"),
    "마스터카드": ("MA", "Financial"),
    "골드만삭스": ("GS", "Financial"),
    "모건스탠리": ("MS", "Financial"),
    "블랙록": ("BLK", "Financial"),
    "아메리칸익스프레스": ("AXP", "Financial"),
    "시티그룹": ("C", "Financial"),
    # Healthcare
    "유나이티드헬스": ("UNH", "Healthcare"),
    "존슨앤드존슨": ("JNJ", "Healthcare"),
    "존슨앤존슨": ("JNJ", "Healthcare"),
    "일라이릴리": ("LLY", "Healthcare"),
    "노보노디스크": ("NVO", "Healthcare"),
    "화이자": ("PFE", "Healthcare"),
    "애브비": ("ABBV", "Healthcare"),
    "머크": ("MRK", "Healthcare"),
    # Consumer
    "월마트": ("WMT", "Consumer Defensive"),
    "코스트코": ("COST", "Consumer Defensive"),
    "코카콜라": ("KO", "Consumer Defensive"),
    "펩시코": ("PEP", "Consumer Defensive"),
    "맥도날드": ("MCD", "Consumer Cyclical"),
    "스타벅스": ("SBUX", "Consumer Cyclical"),
    "나이키": ("NKE", "Consumer Cyclical"),
    "디즈니": ("DIS", "Communication"),
    "홈디포": ("HD", "Consumer Cyclical"),
    # Industrial / Energy
    "엑슨모빌": ("XOM", "Energy"),
    "셰브론": ("CVX", "Energy"),
    "보잉": ("BA", "Industrials"),
    "캐터필러": ("CAT", "Industrials"),
    "록히드마틴": ("LMT", "Industrials"),
    "레이시온": ("RTX", "Industrials"),
    "허니웰": ("HON", "Industrials"),
    # Asia ADR
    "알리바바": ("BABA", "Technology"),
    "TSMC": ("TSM", "Technology"),
    "대만반도체": ("TSM", "Technology"),
    "텐센트": ("TCEHY", "Technology"),
    "핀둬둬": ("PDD", "Consumer Cyclical"),
    "테무": ("PDD", "Consumer Cyclical"),
    "소니": ("SONY", "Technology"),
    "토요타": ("TM", "Consumer Cyclical"),
    "니오": ("NIO", "Consumer Cyclical"),
}

# 벤치마크 매핑
KR_BENCHMARKS = {
    "KOSPI": "^KS11",
    "KOSDAQ": "^KQ11",
    "KOSPI 200": "^KS200",
}


# 검색용 전체 종목 리스트: "삼성전자 (005930) - 반도체" 형식
ALL_STOCK_OPTIONS = [
    f"{name} ({code}) - {sector}"
    for name, (code, sector) in KR_STOCK_MAP.items()
]

# 글로벌 주요 주식/ETF 100선 — 시가총액 상위 + 인기 ETF
GLOBAL_POPULAR = {
    # ── Mega Cap Tech ──
    "AAPL - Apple": ("AAPL", "Technology"),
    "MSFT - Microsoft": ("MSFT", "Technology"),
    "GOOGL - Alphabet (Google)": ("GOOGL", "Technology"),
    "AMZN - Amazon": ("AMZN", "Consumer Cyclical"),
    "NVDA - NVIDIA": ("NVDA", "Technology"),
    "TSLA - Tesla": ("TSLA", "Consumer Cyclical"),
    "META - Meta Platforms": ("META", "Technology"),
    "AVGO - Broadcom": ("AVGO", "Technology"),
    "ORCL - Oracle": ("ORCL", "Technology"),
    "CRM - Salesforce": ("CRM", "Technology"),
    "ADBE - Adobe": ("ADBE", "Technology"),
    "AMD - Advanced Micro Devices": ("AMD", "Technology"),
    "INTC - Intel": ("INTC", "Technology"),
    "QCOM - Qualcomm": ("QCOM", "Technology"),
    "NFLX - Netflix": ("NFLX", "Communication"),
    "CSCO - Cisco": ("CSCO", "Technology"),
    "IBM - IBM": ("IBM", "Technology"),
    "NOW - ServiceNow": ("NOW", "Technology"),
    "UBER - Uber": ("UBER", "Technology"),
    "SHOP - Shopify": ("SHOP", "Technology"),
    "SNOW - Snowflake": ("SNOW", "Technology"),
    "PLTR - Palantir": ("PLTR", "Technology"),
    "MU - Micron Technology": ("MU", "Technology"),
    "ARM - ARM Holdings": ("ARM", "Technology"),

    # ── Financial ──
    "BRK-B - Berkshire Hathaway": ("BRK-B", "Financial"),
    "JPM - JPMorgan Chase": ("JPM", "Financial"),
    "V - Visa": ("V", "Financial"),
    "MA - Mastercard": ("MA", "Financial"),
    "BAC - Bank of America": ("BAC", "Financial"),
    "GS - Goldman Sachs": ("GS", "Financial"),
    "MS - Morgan Stanley": ("MS", "Financial"),
    "BLK - BlackRock": ("BLK", "Financial"),
    "AXP - American Express": ("AXP", "Financial"),
    "C - Citigroup": ("C", "Financial"),

    # ── Healthcare ──
    "UNH - UnitedHealth": ("UNH", "Healthcare"),
    "JNJ - Johnson & Johnson": ("JNJ", "Healthcare"),
    "LLY - Eli Lilly": ("LLY", "Healthcare"),
    "NVO - Novo Nordisk": ("NVO", "Healthcare"),
    "PFE - Pfizer": ("PFE", "Healthcare"),
    "ABBV - AbbVie": ("ABBV", "Healthcare"),
    "MRK - Merck": ("MRK", "Healthcare"),
    "TMO - Thermo Fisher": ("TMO", "Healthcare"),
    "ISRG - Intuitive Surgical": ("ISRG", "Healthcare"),

    # ── Consumer ──
    "WMT - Walmart": ("WMT", "Consumer Defensive"),
    "COST - Costco": ("COST", "Consumer Defensive"),
    "PG - Procter & Gamble": ("PG", "Consumer Defensive"),
    "KO - Coca-Cola": ("KO", "Consumer Defensive"),
    "PEP - PepsiCo": ("PEP", "Consumer Defensive"),
    "MCD - McDonald's": ("MCD", "Consumer Cyclical"),
    "SBUX - Starbucks": ("SBUX", "Consumer Cyclical"),
    "NKE - Nike": ("NKE", "Consumer Cyclical"),
    "DIS - Walt Disney": ("DIS", "Communication"),
    "HD - Home Depot": ("HD", "Consumer Cyclical"),

    # ── Industrial / Energy ──
    "XOM - ExxonMobil": ("XOM", "Energy"),
    "CVX - Chevron": ("CVX", "Energy"),
    "BA - Boeing": ("BA", "Industrials"),
    "CAT - Caterpillar": ("CAT", "Industrials"),
    "GE - GE Aerospace": ("GE", "Industrials"),
    "UNP - Union Pacific": ("UNP", "Industrials"),
    "LMT - Lockheed Martin": ("LMT", "Industrials"),
    "RTX - RTX (Raytheon)": ("RTX", "Industrials"),
    "HON - Honeywell": ("HON", "Industrials"),
    "DE - John Deere": ("DE", "Industrials"),

    # ── 중국/일본/글로벌 ADR ──
    "BABA - Alibaba": ("BABA", "Technology"),
    "TSM - TSMC": ("TSM", "Technology"),
    "TCEHY - Tencent (OTC)": ("TCEHY", "Technology"),
    "PDD - PDD Holdings (Temu)": ("PDD", "Consumer Cyclical"),
    "SONY - Sony": ("SONY", "Technology"),
    "TM - Toyota": ("TM", "Consumer Cyclical"),
    "NIO - NIO": ("NIO", "Consumer Cyclical"),

    # ── Index ETF ──
    "SPY - S&P 500 ETF": ("SPY", "Index ETF"),
    "QQQ - Nasdaq 100 ETF": ("QQQ", "Index ETF"),
    "IWM - Russell 2000 ETF": ("IWM", "Index ETF"),
    "VTI - Total US Market": ("VTI", "Index ETF"),
    "VOO - Vanguard S&P 500": ("VOO", "Index ETF"),
    "DIA - Dow Jones ETF": ("DIA", "Index ETF"),
    "VT - Total World Stock": ("VT", "Index ETF"),
    "EFA - Developed Markets": ("EFA", "Index ETF"),
    "EEM - Emerging Markets": ("EEM", "Index ETF"),
    "VWO - Vanguard EM": ("VWO", "Index ETF"),

    # ── Sector ETF ──
    "XLK - Technology Select": ("XLK", "Sector ETF"),
    "XLF - Financials Select": ("XLF", "Sector ETF"),
    "XLE - Energy Select": ("XLE", "Sector ETF"),
    "XLV - Healthcare Select": ("XLV", "Sector ETF"),
    "XLI - Industrials Select": ("XLI", "Sector ETF"),
    "XLRE - Real Estate Select": ("XLRE", "Sector ETF"),
    "SOXX - Semiconductor ETF": ("SOXX", "Sector ETF"),
    "SMH - VanEck Semiconductor": ("SMH", "Sector ETF"),
    "ARKK - ARK Innovation": ("ARKK", "Thematic ETF"),
    "ARKW - ARK Next Gen Internet": ("ARKW", "Thematic ETF"),

    # ── Bond / Commodity / Alternative ──
    "TLT - 20+ Year Treasury Bond": ("TLT", "Bond ETF"),
    "BND - Total Bond Market": ("BND", "Bond ETF"),
    "HYG - High Yield Corporate Bond": ("HYG", "Bond ETF"),
    "LQD - Investment Grade Bond": ("LQD", "Bond ETF"),
    "GLD - Gold ETF (SPDR)": ("GLD", "Commodity ETF"),
    "SLV - Silver ETF": ("SLV", "Commodity ETF"),
    "USO - Oil ETF": ("USO", "Commodity ETF"),
    "BITX - Bitcoin 2x ETF": ("BITX", "Crypto ETF"),
    "VNQ - Real Estate ETF": ("VNQ", "Real Estate ETF"),
    "SCHD - Schwab Dividend ETF": ("SCHD", "Dividend ETF"),
}

GLOBAL_STOCK_OPTIONS = list(GLOBAL_POPULAR.keys())

# 통합 검색 리스트
ALL_SEARCHABLE = ["직접 입력 (티커 직접 타이핑)"] + ALL_STOCK_OPTIONS + GLOBAL_STOCK_OPTIONS


def parse_stock_selection(selection: str) -> Tuple[str, str, str]:
    """
    검색 드롭다운 선택값을 (ticker, display_name, sector)로 변환

    Handles:
    - "삼성전자 (005930) - 반도체" → ("005930.KS", "삼성전자", "반도체")
    - "AAPL - Apple" → ("AAPL", "AAPL", "Technology")
    - "직접 입력 ..." → ("", "", "")
    """
    if not selection or selection.startswith("직접 입력"):
        return "", "", ""

    # 한국 주식: "삼성전자 (005930) - 반도체"
    if "(" in selection and ")" in selection:
        name = selection.split("(")[0].strip()
        code = selection.split("(")[1].split(")")[0].strip()
        sector = selection.split("-")[-1].strip() if "-" in selection else "Unknown"
        return f"{code}.KS", name, sector

    # 글로벌: "AAPL - Apple"
    if selection in GLOBAL_POPULAR:
        ticker, sector = GLOBAL_POPULAR[selection]
        return ticker, ticker, sector

    return selection, selection, "Unknown"


# =============================================================================
# 변환 함수
# =============================================================================

def resolve_kr_ticker(input_str: str) -> Tuple[str, str, str]:
    """
    다양한 형식의 한국 주식 입력을 yfinance 티커로 변환

    Parameters:
        input_str: "삼성전자" or "005930" or "005930.KS"

    Returns:
        (yfinance_ticker, display_name, sector)
        예: ("005930.KS", "삼성전자", "반도체")
    """
    input_str = input_str.strip()

    # 이미 yfinance 포맷인 경우 (.KS, .KQ)
    if input_str.endswith(".KS") or input_str.endswith(".KQ"):
        code = input_str.split(".")[0]
        name = KR_CODE_TO_NAME.get(code, input_str)
        sector = KR_CODE_TO_SECTOR.get(code, "Unknown")
        return input_str, name, sector

    # 한글 종목명인 경우 — 정확히 일치
    if input_str in KR_STOCK_MAP:
        code, sector = KR_STOCK_MAP[input_str]
        return f"{code}.KS", input_str, sector

    # 글로벌 주식 한글 별칭 매칭 (예: "애플" → AAPL)
    if input_str in GLOBAL_KOREAN_ALIAS:
        ticker, sector = GLOBAL_KOREAN_ALIAS[input_str]
        return ticker, input_str, sector

    # 글로벌 한글 별칭 퍼지 (공백 제거)
    normalized_input = input_str.replace(" ", "")
    if normalized_input in GLOBAL_KOREAN_ALIAS:
        ticker, sector = GLOBAL_KOREAN_ALIAS[normalized_input]
        return ticker, normalized_input, sector

    # 영문 별칭 매칭 (대소문자 무시)
    lower_input = input_str.lower().strip()
    if lower_input in KR_ENGLISH_ALIAS:
        kr_name = KR_ENGLISH_ALIAS[lower_input]
        if kr_name in KR_STOCK_MAP:
            code, sector = KR_STOCK_MAP[kr_name]
            return f"{code}.KS", kr_name, sector

    # 영문 부분 매칭 (예: "samsung" → "samsung electronics" → 삼성전자)
    for eng_key, kr_name in KR_ENGLISH_ALIAS.items():
        if lower_input in eng_key or eng_key in lower_input:
            if kr_name in KR_STOCK_MAP:
                code, sector = KR_STOCK_MAP[kr_name]
                return f"{code}.KS", kr_name, sector

    # 퍼지 매칭: 띄어쓰기 제거, 부분 일치, 오타 허용
    normalized = input_str.replace(" ", "").replace("\t", "")

    # 1) 공백 제거 후 정확히 일치 (예: "삼성 전자" → "삼성전자")
    if normalized in KR_STOCK_MAP:
        code, sector = KR_STOCK_MAP[normalized]
        return f"{code}.KS", normalized, sector

    # 2) 부분 문자열 매칭 (예: "삼성" → "삼성전자", "한화에어로" → "한화에어로스페이스")
    candidates = []
    for name, (code, sector) in KR_STOCK_MAP.items():
        name_normalized = name.replace(" ", "")
        if normalized in name_normalized:
            # 입력이 종목명에 포함: "한화에어로" in "한화에어로스페이스" ✓
            # 우선도: 길이 차이가 작을수록 좋음 (0 = 시작부분 일치)
            candidates.append((name, code, sector, 0, abs(len(name_normalized) - len(normalized))))
        elif name_normalized in normalized:
            # 종목명이 입력에 포함: "한화" in "한화에어로" — 덜 정확하므로 후순위
            candidates.append((name, code, sector, 1, abs(len(name_normalized) - len(normalized))))

    if candidates:
        # 정렬: (1) 입력⊂종목명 우선, (2) 길이 차이 작은 것 우선
        best = min(candidates, key=lambda x: (x[3], x[4]))
        return f"{best[1]}.KS", best[0], best[2]

    # 숫자만 있으면 종목코드로 간주
    if input_str.isdigit():
        code = input_str.zfill(6)  # 6자리로 패딩
        name = KR_CODE_TO_NAME.get(code, code)
        sector = KR_CODE_TO_SECTOR.get(code, "Unknown")
        return f"{code}.KS", name, sector

    # 매칭 실패 → 글로벌 티커로 취급 (AAPL 등)
    return input_str, input_str, "Unknown"


def is_korean_ticker(input_str: str) -> bool:
    """한국 주식인지 판별"""
    input_str = input_str.strip()

    if input_str.endswith(".KS") or input_str.endswith(".KQ"):
        return True
    if input_str in KR_STOCK_MAP:
        return True
    if input_str.isdigit() and len(input_str) <= 6:
        return True

    return False


def resolve_benchmark(benchmark_name: str) -> str:
    """벤치마크 이름을 yfinance 티커로 변환"""
    if benchmark_name in KR_BENCHMARKS:
        return KR_BENCHMARKS[benchmark_name]
    return benchmark_name  # 글로벌 벤치마크는 그대로


def build_kr_portfolio(
    inputs: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, str]]:
    """
    사용자 입력(한글/코드 혼합)을 정리된 포트폴리오로 변환

    Parameters:
        inputs: {"삼성전자": 0.3, "AAPL": 0.2, "005930": 0.1, ...}

    Returns:
        (weights, sector_map, display_names)
        weights: {yfinance_ticker: weight}
        sector_map: {yfinance_ticker: sector}
        display_names: {yfinance_ticker: display_name}
    """
    weights = {}
    sector_map = {}
    display_names = {}

    for input_str, weight in inputs.items():
        ticker, name, sector = resolve_kr_ticker(input_str)
        weights[ticker] = weight
        sector_map[ticker] = sector
        display_names[ticker] = name

    # 정규화
    total = sum(weights.values())
    if total > 0:
        weights = {t: w / total for t, w in weights.items()}

    return weights, sector_map, display_names


def get_popular_kr_portfolios() -> Dict[str, Dict[str, float]]:
    """한국 주식 샘플 포트폴리오"""
    return {
        "국민 포트폴리오 (대형주)": {
            "삼성전자": 0.25,
            "SK하이닉스": 0.15,
            "현대차": 0.10,
            "네이버": 0.10,
            "카카오": 0.08,
            "LG에너지솔루션": 0.08,
            "삼성바이오로직스": 0.07,
            "KB금융": 0.07,
            "포스코홀딩스": 0.05,
            "SK텔레콤": 0.05,
        },
        "2차전지 테마": {
            "LG에너지솔루션": 0.25,
            "삼성SDI": 0.20,
            "에코프로비엠": 0.15,
            "에코프로": 0.15,
            "포스코퓨처엠": 0.15,
            "SK이노베이션": 0.10,
        },
        "배당주 포트폴리오": {
            "삼성전자": 0.15,
            "KB금융": 0.12,
            "신한지주": 0.12,
            "하나금융지주": 0.12,
            "SK텔레콤": 0.10,
            "KT": 0.10,
            "한국전력": 0.08,
            "삼성화재": 0.08,
            "포스코홀딩스": 0.07,
            "S-Oil": 0.06,
        },
        "성장주 포트폴리오": {
            "네이버": 0.15,
            "카카오": 0.12,
            "크래프톤": 0.12,
            "삼성바이오로직스": 0.12,
            "셀트리온": 0.10,
            "에코프로": 0.10,
            "한화에어로스페이스": 0.10,
            "한화오션": 0.10,
            "HD한국조선해양": 0.09,
        },
    }


def search_kr_stock(query: str, max_results: int = 10) -> list:
    """한글/영문 검색으로 종목 찾기"""
    query = query.strip().lower()
    results = []

    for name, (code, sector) in KR_STOCK_MAP.items():
        if query in name.lower() or query in code:
            results.append({
                "name": name,
                "code": code,
                "ticker": f"{code}.KS",
                "sector": sector,
            })
            if len(results) >= max_results:
                break

    return results
