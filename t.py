import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from talib import SMA, AD, STOCH, MA, RSI, MACD, BBANDS, ATR, WCLPRICE, AVGPRICE, TYPPRICE, SAR
from patterns import CANDLE_PATTERNS  # Импорт базы паттернов
import csv

# Логирование действий бота
log_file = 'trading_log.csv'

def log_trade(action, price, timestamp):
    with open(log_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, action, price])

# Функция для подготовки данных для обучения модели
def prepare_data(data):
    features = data[['RSI', 'MACD_Line', 'Signal_Line', 'SMA_50', 'BB_Width', 'ATR']].dropna()
    labels = (data['close'].shift(-1) > data['close']).astype(int)  # 1 - рост, 0 - падение
    # Убедимся, что длины совпадают
    min_length = min(len(features), len(labels))
    features = features.iloc[:min_length]
    labels = labels.iloc[:min_length]
    return features, labels

# Функция для обучения модели
def train_model(data):
    features, labels = prepare_data(data)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
    joblib.dump(model, 'trading_model.pkl')  # Сохранение модели
    return model

# Функция для переобучения модели
def retrain_model(data):
    print("Переобучение модели...")
    return train_model(data)

# Функция для прогнозирования сигналов с использованием модели
def predict_signals(model, data):
    features = data[['RSI', 'MACD_Line', 'Signal_Line', 'SMA_50', 'BB_Width', 'ATR']].dropna()
    predictions = model.predict(features)
    data['ML_Signal'] = np.nan
    data.loc[features.index, 'ML_Signal'] = predictions
    return data

# Функция для построения линий тренда
def draw_trendlines(data, ax):
    highs = data['high']
    lows = data['low']

    # Линия тренда по максимумам
    ax.plot(highs.index, highs, label="High Trendline", color="green", linestyle="--", alpha=0.7)

    # Линия тренда по минимумам
    ax.plot(lows.index, lows, label="Low Trendline", color="red", linestyle="--", alpha=0.7)

    ax.legend()

# Функция для получения данных с биржи через ccxt
def fetch_ohlcv(exchange, symbol, timeframe, limit=900):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.set_index('timestamp', inplace=True)
    return data

# Проверка доступности пары
def is_symbol_available(exchange, symbol):
    markets = exchange.load_markets()
    return symbol in markets

# Подключение к бирже через ccxt
exchange = ccxt.binance({
    "rateLimit": 1200,
    "enableRateLimit": True,
})

# Выбор криптовалютной пары и таймфрейма
symbol = input("Введите торговую пару (например, BTC/USDT): ").strip().upper()
if not is_symbol_available(exchange, symbol):
    raise ValueError(f"Пара {symbol} недоступна на Binance. Проверьте символ и попробуйте снова.")

timeframe = input("Введите таймфрейм (например, 5m, 15m): ").strip()

# Обучение модели на исторических данных
print("Загрузка данных для обучения модели...")
data = fetch_ohlcv(exchange, symbol, timeframe)
data['SMA_50'] = SMA(data['close'], timeperiod=50)
data['RSI'] = RSI(data['close'], timeperiod=14)
data['MACD_Line'], data['Signal_Line'], _ = MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)
data['upper_band'], data['middle_band'], data['lower_band'] = BBANDS(data['close'], timeperiod=20)
data['ATR'] = ATR(data['high'], data['low'], data['close'], timeperiod=14)
data['BB_Width'] = data['upper_band'] - data['lower_band']

print("Обучение модели...")
try:
    model = joblib.load('trading_model.pkl')
    print("Модель загружена из файла.")
except (FileNotFoundError, IndexError, EOFError):
    print("Файл модели отсутствует или повреждён. Обучение новой модели...")
    model = train_model(data)
    print("Модель обучена и сохранена.")
# Функция для расчета и отображения всех паттернов
def detect_and_plot_patterns(data, ax):
    for pattern_name, pattern_info in CANDLE_PATTERNS.items():
        # Расчет текущего паттерна
        data[pattern_name] = pattern_info["function"](data['open'], data['high'], data['low'], data['close'])

        # Отображаем только те паттерны, которые найдены
        if data[pattern_name].any():
            # Координаты маркеров для найденных паттернов
            pattern_indices = data.index[data[pattern_name] != 0]
            pattern_prices = data['close'][data[pattern_name] != 0]

            # Добавляем маркеры
            ax.scatter(
                pattern_indices,  # Индексы, где найден паттерн
                pattern_prices,   # Соответствующие цены закрытия
                marker='o',
                s=100
            )

            # Добавляем текст под каждым маркером
            for i, price in zip(pattern_indices, pattern_prices):
                ax.text(
                    i,  # Индекс (время/дата)
                    price - (price * 0.005),  # Чуть ниже цены закрытия (регулируйте сдвиг)
                    f"{pattern_info['name']}",  # Название паттерна
                    fontsize=8,
                    color='black',
                    ha='center'
                )

# Инициализация отдельных окон для индикаторов
fig_price, ax_price = plt.subplots(figsize=(14, 6))  # Основной график
fig_rsi, ax_rsi = plt.subplots(figsize=(10, 4))  # RSI
fig_macd, ax_macd = plt.subplots(figsize=(10, 4))  # MACD
fig_atr, ax_atr = plt.subplots(figsize=(10, 4))  # ATR
fig_stoch, ax_stoch = plt.subplots(figsize=(20, 8))  # AD

# Функция обновления графиков
def update(frame):
    global data, model
    data = fetch_ohlcv(exchange, symbol, timeframe)
    
    # Расчет индикаторов
    data['SMA_50'] = SMA(data['close'], timeperiod=50)
    data['RSI'] = RSI(data['close'], timeperiod=14)
    data['MACD_Line'], data['Signal_Line'], _ = MACD(data['close'], fastperiod=13, slowperiod=26, signalperiod=9)
    data['upper_band'], data['middle_band'], data['lower_band'] = BBANDS(data['close'], timeperiod=20)
    data['ATR'] = ATR(data['high'], data['low'], data['close'], timeperiod=14)
    data['BB_Width'] = data['upper_band'] - data['lower_band']
    data['WCLPRICE'] = WCLPRICE(data['high'], data['low'], data['close'])
    data['AVGPRICE'] = AVGPRICE(data['open'], data['high'], data['low'], data['close'])
    data['TYPPRICE'] = TYPPRICE(data['high'], data['low'], data['close'])
    data['SAR'] = SAR(data['high'], data['low'], acceleration=0.2, maximum=0.2)
    data['slowk'], data['slowd'] = STOCH(data['high'], data['low'], data['close'], fastk_period=21, slowk_period=9, slowk_matype=0, slowd_period=9, slowd_matype=0)

    # Переобучение модели
    model = retrain_model(data)

    # Прогнозирование сигналов
    data = predict_signals(model, data)

    # Дополнительные сигналы
    data['Buy_Signal'] = (data['close'] < data['lower_band']) & (data['RSI'] < 30)
    data['Sell_Signal'] = (data['close'] > data['upper_band']) & (data['RSI'] > 70)

    # Логирование действий
    if data['Buy_Signal'].iloc[-1]:
        log_trade('buy', data['close'].iloc[-1], data.index[-1])
    if data['Sell_Signal'].iloc[-1]:
        log_trade('sell', data['close'].iloc[-1], data.index[-1])

    # Основной график
    ax_price.clear()
    ax_price.plot(data.index, data['close'], label='Price', color='black')
    ax_price.plot(data.index, data['SMA_50'], label='SMA 50', color='blue', linestyle='--')
    ax_price.plot(data.index, data['upper_band'], label='Upper BB', color='orange', linestyle='--')
    ax_price.plot(data.index, data['lower_band'], label='Lower BB', color='orange', linestyle='--')
    ax_price.plot(data.index, data['WCLPRICE'], color='DeepPink', linestyle=':')
    ax_price.plot(data.index, data['AVGPRICE'], color='DarkBlue', linestyle=':')
    ax_price.plot(data.index, data['TYPPRICE'], color='Yellow', linestyle=':')
    ax_price.plot(data.index, data['SAR'], color='DarkRed', linestyle='-.')

    draw_trendlines(data, ax_price)
    detect_and_plot_patterns(data, ax_price)
    ax_price.scatter(data.index[data['ML_Signal'] == 1], data['close'][data['ML_Signal'] == 1], label='Buy Signal (ML)', color='green', marker='^')
    ax_price.scatter(data.index[data['ML_Signal'] == 0], data['close'][data['ML_Signal'] == 0], label='Sell Signal (ML)', color='red', marker='v')
    ax_price.set_title(f'Trading Strategy with ML for {symbol} on {timeframe}')
    ax_price.legend()

    # RSI
    ax_rsi.clear()
    ax_rsi.plot(data.index, data['RSI'], label='RSI', color='purple')
    ax_rsi.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    ax_rsi.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    ax_rsi.legend()
    ax_rsi.set_title("RSI Indicator")

    # MACD
    ax_macd.clear()
    ax_macd.plot(data.index, data['MACD_Line'], label='MACD Line', color='blue')
    ax_macd.plot(data.index, data['Signal_Line'], label='Signal Line', color='red')
    ax_macd.axhline(0, color='black', linestyle='--')
    ax_macd.legend()
    ax_macd.set_title("MACD Indicator")

    # ATR
    ax_atr.clear()
    ax_atr.plot(data.index, data['ATR'], label='ATR', color='cyan')
    ax_atr.legend()
    ax_atr.set_title("ATR Indicator")
    #AD
    ax_stoch.clear()
    ax_stoch.plot(data.index, data['slowk'], label='slowk', color='green')
    ax_stoch.plot(data.index, data['slowd'], label='slowd', color='red')
    ax_stoch.axhline(80, color='black', linestyle='--')
    ax_stoch.axhline(20, color='black', linestyle='--')

    ax_stoch.legend()
# Анимация для всех окон
ani_price = FuncAnimation(fig_price, update, interval=30000, save_count=200)
ani_rsi = FuncAnimation(fig_rsi, update, interval=30000, save_count=200)
ani_macd = FuncAnimation(fig_macd, update, interval=30000, save_count=200)
ani_atr = FuncAnimation(fig_atr, update, interval=30000, save_count=200)
ani_stoch = FuncAnimation(fig_stoch, update, interval=30000, save_count=200)

# Отображение всех окон
plt.show()
