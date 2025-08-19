# patterns.py

from talib import *

# Список всех доступных паттернов TA-Lib
CANDLE_PATTERNS = {
    
    #"CDL2CROWS": {
        #"name": "2CROWS",
        #"function": CDL2CROWS,
        #"description": "Медвежий разворот"
    #},
    "CDL3BLACKCROWS": {
        "name": "3BLACKCROWS",
        "function": CDL3BLACKCROWS,
        "description": "Медвежий, Бычий разворот"
    },
    "CDL3INSIDE": {
        "name": "3INSIDE",
        "function": CDL3INSIDE,
        "description": "Бычий разворот"
    },
    #"CDL3LINESTRIKE": {
        #"name": "3LINESTRIKE",
        #"function": CDL3LINESTRIKE,
        #"description": "Бычий, медвежий разворот"
    #},
     #"CDL3OUTSIDE": {
        #"name": "3OUTSIDE",
        #"function": CDL3OUTSIDE,
        #"description": "Бычий, медвежий разворот"
    #},
     #"CDL3STARSINSOUTH": {
        #"name": "3STARSINSOUTH",
        #"function": CDL3STARSINSOUTH,
        #"description": "Бычий разворот"
    #},
     "CDL3WHITESOLDIERS": {
        "name": "3WHITESOLDIERS",
        "function": CDL3WHITESOLDIERS,
        "description": "Бычий разворот"
    },
     #"CDLABANDONEDBABY": {
        #"name": "ABANDONEDBABY",
        #"function": CDLABANDONEDBABY,
        #"description": "Бычий разворот"
    #},
    #"CDLADVANCEBLOCK": {
        #"name": "ADVANCEBLOCK",
        #"function": CDLADVANCEBLOCK,
        #"description": "Медвежий разворот"
    #},
    #"CDLBELTHOLD": {
        #"name": "BELTHOLD",
        #"function": CDLBELTHOLD,
        #"description": "Медвкжий, бычий разворот"
    #},
    #"CDLBREAKAWAY": {
        #"name": "BREAKAWAY",
        #"function": CDLBREAKAWAY,
        #"description": "Бычий разворот"
    #},
    #"CDLCLOSINGMARUBOZU": {
        #"name": "marubozuclosing",
        #"function": CDLCLOSINGMARUBOZU,
        #"description": "Разворот или продолжение тренда"
    #},
     #"CDLCONCEALBABYSWALL": {
        #"name": "CONCEALBABYSWALL",
        #"function": CDLCONCEALBABYSWALL,
        #"description": "Бычий разворот"
    #},
     #"CDLCOUNTERATTACK": {
        #"name": "COUNTERATTACK",
        #"function": CDLCOUNTERATTACK,
        #"description": "Бычий медвежий разворот"
    #},
     #"CDLDARKCLOUDCOVER": {
        #"name": "DARKCLOUDCOVER",
        #"function": CDLDARKCLOUDCOVER,
        #"description": "Медвежий разворот"
    #},
     #"CDLDOJI": {
        #"name": "CDLDOJI",
        #"function": CDLDOJI,
        #"description": ""
    #},
    #"CDLDOJISTAR": {
       # "name": "CDLDOJISTAR",
        #"function": CDLDOJISTAR,
        #"description": ""
    #},
    #"CDLDRAGONFLYDOJI": {
        #"name": "CDLDRAGONFLYDOJI",
        #"function": CDLDRAGONFLYDOJI,
        #"description": ""
    #},
    "CDLENGULFING": {
        "name": "ENGULFING",
        "function": CDLENGULFING,
        "description": ""
    },
    #"CDLEVENINGDOJISTAR": {
        #"name": "EVENINGDOJISTAR",
        #"function": CDLEVENINGDOJISTAR,
        #"description": "Медвежий разворот"
    #},
     "CDLEVENINGSTAR": {
        "name": "EVENINGSTAR",
        "function": CDLEVENINGSTAR,
        "description": "Медвежий разворот"
    },
     #"CDLGAPSIDESIDEWHITE": {
        #"name": "GAPSIDESIDEWHITE",
        #"function": CDLGAPSIDESIDEWHITE,
        #"description": "Бычий разворот"
    #},
     #"CDLGRAVESTONEDOJI": {
        #"name": "GRAVESTONEDOJI",
        #"function": CDLGRAVESTONEDOJI,
        #"description": "Медвежий разворот"
    #},
     #"CDLHAMMER": {
        #"name": "CDLHAMMER",
        #"function": CDLHAMMER,
        #"description": ""
    #},
    "CDLHANGINGMAN": {
        "name": "Г#",
        "function": CDLHANGINGMAN,
        "description": "Медвежий паттерн"
    },
    "CDLHARAMI": {
        "name": "harami",
        "function": CDLHARAMI,
        "description": ""
    },
    #"CDLHARAMICROSS": {
       # "name": "CDLHARAMICROSS",
        #"function": CDLHARAMICROSS,
        #"description": ""
    #},
    #"CDLHIGHWAVE": {
        #"name": "HIGHWAVE",
        #"function": CDLHIGHWAVE,
        #"description": "Нерешительность"
    #},
     #"CDLHIKKAKE": {
        #"name": "HI-KE",
        #"function": CDLHIKKAKE,
        #"description": ""
    #},
     #"CDLHIKKAKEMOD": {
        #"name": "hi-kemod",
        #"function": CDLHIKKAKEMOD,
        #"description": ""
    #},
     #"CDLHOMINGPIGEON": {
        #"name": "CDLHOMINGPIGEON",
        #"function": CDLHOMINGPIGEON,
        #"description": ""
    #},
     "CDLIDENTICAL3CROWS": {
        "name": "CDLIDENTICAL3CROWS",
        "function": CDLIDENTICAL3CROWS,
        "description": ""
    },
    #"CDLINNECK": {
        #"name": "CDLINNECK",
        #"function": CDLINNECK,
        #"description": ""
    #},
    #"CDLINVERTEDHAMMER": {
        #"name": "CDLINVERTEDHAMMER",
        #"function": CDLINVERTEDHAMMER,
        #"description": ""
    #},
    #"CDLKICKING": {
        #"name": "CDLKICKING",
        #"function": CDLKICKING,
        #"description": ""
    #},
    #"CDLKICKINGBYLENGTH": {
        #"name": "CDLKICKINGBYLENGTH",
        #"function": CDLKICKINGBYLENGTH,
        #"description": ""
    #},
     #"CDLLADDERBOTTOM": {
        #"name": "LADDERBOTTOM",
        #"function": CDLLADDERBOTTOM,
        #"description": ""
    #},
     #"CDLLONGLEGGEDDOJI": {
        #"name": "LONGLEGGEDDOJI",
        #"function": CDLLONGLEGGEDDOJI,
        #"description": ""
    #},
     #"CDLLONGLINE": {
        #"name": "LONGLINE",
        #"function": CDLLONGLINE,
        #"description": ""
    #},
     "CDLMARUBOZU": {
        "name": "MARUBOZU",
        "function": CDLMARUBOZU,
        "description": ""
    },
    #"CDLMATCHINGLOW": {
        #"name": "MATCHINGLOW",
        #"function": CDLMATCHINGLOW,
        #"description": ""
    #},
     #"CDLMATHOLD": {
        #"name": "CDLMATHOLD",
        #"function": CDLMATHOLD,
        #"description": ""
    #},
     #"CDLMORNINGDOJISTAR": {
        #"name": "MORNINGDOJISTAR",
        #"function": CDLMORNINGDOJISTAR,
        #"description": ""
    #},
     "CDLMORNINGSTAR": {
        "name": "MORNINGSTAR",
        "function": CDLMORNINGSTAR,
        "description": ""
    },
     #"CDLONNECK": {
        #"name": "CDLONNECK",
        #"function": CDLONNECK,
        #"description": ""
    #},
    "CDLPIERCING": {
        "name": "PIERCING",
        "function": CDLPIERCING,
        "description": ""
    },
     #"CDLRICKSHAWMAN": {
        #"name": "CDLRICKSHAWMAN",
        #"function": CDLRICKSHAWMAN,
        #"description": ""
    #},
     #"CDLRISEFALL3METHODS": {
        #"name": "RISEFALL3METHODS",
        #"function": CDLRISEFALL3METHODS,
        #"description": ""
    #},
     #"CDLSEPARATINGLINES": {
        #"name": "CDLSEPARATINGLINES",
        #"function": CDLSEPARATINGLINES,
        #"description": ""
    #},
     #"CDLSHOOTINGSTAR": {
        #"name": "SHOOTINGSTAR",
        #"function": CDLSHOOTINGSTAR,
        #"description": ""
    #},
    "CDLSHORTLINE": {
        "name": "SHORTLINE",
        "function": CDLSHORTLINE,
        "description": ""
    },
     #"CDLSPINNINGTOP": {
        #"name": "PINNINGTOP",
        #"function": CDLSPINNINGTOP,
        #"description": ""
    #},
     #"CDLSTALLEDPATTERN": {
        #"name": "STALLEDPATTERN",
        #"function": CDLSTALLEDPATTERN,
        #"description": ""
    #},
     #"CDLSTICKSANDWICH": {
        #"name": "CDLSTICKSANDWICH",
        #"function": CDLSTICKSANDWICH,
        #"description": ""
    #},
     #"CDLTAKURI": {
        #"name": "TAKURI",
        #"function": CDLTAKURI,
        #"description": ""
    #},
    #"CDLTASUKIGAP": {
        #"name": "TASUKIGAP",
        #"function": CDLTASUKIGAP,
        #"description": ""
    #},
     "CDLTHRUSTING": {
        "name": "THRUSTING",
        "function": CDLTHRUSTING,
        "description": ""
    },
     #"CDLTRISTAR": {
        #"name": "TRISTAR",
        #"function": CDLTRISTAR,
        #"description": ""
    #},
     #"CDLUNIQUE3RIVER": {
        #"name": "UNIQUE3RIVER",
        #"function": CDLUNIQUE3RIVER,
        #"description": ""
    #},
     #"CDLUPSIDEGAP2CROWS": {
        #"name": "UPSIDEGAP2CROWS",
        #"function": CDLUPSIDEGAP2CROWS,
        #"description": ""
    #},
    #"CDLXSIDEGAP3METHODS": {
        #"name": "XSIDEGAP3METHODS",
        #"function": CDLXSIDEGAP3METHODS,
        #"description": ""
    #},
    # Добавьте сюда другие паттерны по мере необходимости
}
