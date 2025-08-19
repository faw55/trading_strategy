import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Важно для работы без GUI
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from talib import SMA, AD, STOCH, MA, RSI, MACD, BBANDS, ATR, WCLPRICE, AVGPRICE, TYPPRICE, SAR
import csv
import asyncio
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)
import logging
import requests
import json
# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Токен вашего бота
TOKEN = "bot token TELEGRAM"
OPENROUTER_API_KEY = "API_KEY"  # Получите на https://openrouter.ai/keys
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "cognitivecomputations/dolphin-mistral-24b-venice-edition:free"  # Бесплатная модель
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
    "rateLimit": 900,
    "enableRateLimit": True,
})

# Глобальная переменная для отслеживания состояния анализа
analysis_in_progress = False

def calculate_indicators(data):
    """Расчет технических индикаторов"""
    data['SMA_50'] = SMA(data['close'], timeperiod=50)
    data['RSI'] = RSI(data['close'], timeperiod=14)
    data['MACD_Line'], data['Signal_Line'], _ = MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    data['upper_band'], data['middle_band'], data['lower_band'] = BBANDS(data['close'], timeperiod=20)
    data['ATR'] = ATR(data['high'], data['low'], data['close'], timeperiod=14)
    data['BB_Width'] = data['upper_band'] - data['lower_band']
    data['WCLPRICE'] = WCLPRICE(data['high'], data['low'], data['close'])
    data['AVGPRICE'] = AVGPRICE(data['open'], data['high'], data['low'], data['close'])
    data['TYPPRICE'] = TYPPRICE(data['high'], data['low'], data['close'])
    data['SAR'] = SAR(data['high'], data['low'], acceleration=0.2, maximum=0.2)
    data['slowk'], data['slowd'] = STOCH(data['high'], data['low'], data['close'], fastk_period=21, slowk_period=9, slowk_matype=0, slowd_period=9, slowd_matype=0)
    return data

def generate_price_plot(data, symbol, timeframe):
    """Генерация графика цен"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data['close'], label='Price', color='black')
    ax.plot(data.index, data['SMA_50'], label='SMA 50', color='blue', linestyle='--')
    ax.plot(data.index, data['upper_band'], label='Upper BB', color='orange', linestyle='--')
    ax.plot(data.index, data['lower_band'], label='Lower BB', color='orange', linestyle='--')
    ax.plot(data.index, data['SAR'], color='red', marker='^', markersize=4, label='SAR')
    ax.set_title(f'{symbol} Price Analysis ({timeframe})')
    ax.legend()
    
    # Сохраняем в буфер
    buf = BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

def generate_rsi_plot(data):
    """Генерация графика RSI"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data.index, data['RSI'], label='RSI', color='purple')
    ax.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    ax.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    ax.set_title("RSI Indicator")
    ax.legend()
    
    buf = BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

def generate_stochastic_plot(data):
    """Генерация графика Stochastic"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data.index, data['slowk'], label='%K', color='green')
    ax.plot(data.index, data['slowd'], label='%D', color='red')
    ax.axhline(80, color='black', linestyle='--', label='Overbought')
    ax.axhline(20, color='black', linestyle='--', label='Oversold')
    ax.set_title("Stochastic Oscillator")
    ax.legend()
    
    buf = BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

def get_trading_signals(data):
    """Анализ торговых сигналов"""
    signals = []
    
    # Stochastic анализ
    last_k = data['slowk'].iloc[-1]
    last_d = data['slowd'].iloc[-1]
    
    if last_k > 80 and last_d > 80:
        signals.append("🟥 Stochastic: Сильная перекупленность!")
    elif last_k < 20 and last_d < 20:
        signals.append("🟩 Stochastic: Сильная перепроданность!")
    elif last_k > 70 or last_d > 70:
        signals.append("🟧 Stochastic: Зона перекупленности")
    elif last_k < 30 or last_d < 30:
        signals.append("🟩 Stochastic: Зона перепроданности")
    
    # RSI анализ
    last_rsi = data['RSI'].iloc[-1]
    if last_rsi > 70:
        signals.append("🔴 RSI: Перекупленность (>70)")
    elif last_rsi < 30:
        signals.append("🟢 RSI: Перепроданность (<30)")
    
    # Bollinger Bands анализ
    last_close = data['close'].iloc[-1]
    if last_close > data['upper_band'].iloc[-1]:
        signals.append("🚨 Цена выше верхней полосы Боллинджера!")
    elif last_close < data['lower_band'].iloc[-1]:
        signals.append("🚨 Цена ниже нижней полосы Боллинджера!")
    
    # MACD анализ
    if data['MACD_Line'].iloc[-1] > data['Signal_Line'].iloc[-1]:
        signals.append("📈 MACD: Бычий сигнал (линия MACD выше сигнальной)")
    else:
        signals.append("📉 MACD: Медвежий сигнал (линия MACD ниже сигнальной)")
    
    return "\n".join(signals)

async def analyze_market(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Асинхронный анализ рынка и отправка результатов"""
    global analysis_in_progress
    
    if analysis_in_progress:
        await update.message.reply_text("⏳ Анализ уже выполняется, пожалуйста подождите...")
        return
    
    try:
        analysis_in_progress = True
        
        # Получаем параметры из команды
        args = context.args
        symbol = args[0] if len(args) > 0 else "BTC/USDT"
        timeframe = args[1] if len(args) > 1 else "15m"
        
        await update.message.reply_text(f"⏳ Начинаю анализ {symbol} на таймфрейме {timeframe}...")
        
        # Получение данных
        data = fetch_ohlcv(exchange, symbol, timeframe)
        data = calculate_indicators(data)
        
        # Переобучение модели
        model = retrain_model(data)
        
        # Прогнозирование сигналов
        data = predict_signals(model, data)
        
        # Логирование действий
        data['Buy_Signal'] = (data['close'] < data['lower_band']) & (data['RSI'] < 30)
        data['Sell_Signal'] = (data['close'] > data['upper_band']) & (data['RSI'] > 70)
        
        if data['Buy_Signal'].iloc[-1]:
            log_trade('buy', data['close'].iloc[-1], data.index[-1])
        if data['Sell_Signal'].iloc[-1]:
            log_trade('sell', data['close'].iloc[-1], data.index[-1])
        
        # Генерация графиков
        price_plot = generate_price_plot(data, symbol, timeframe)
        rsi_plot = generate_rsi_plot(data)
        stoch_plot = generate_stochastic_plot(data)
        
        # Анализ сигналов
        signals = get_trading_signals(data)
        
        # Формирование отчета
        report = (
            f"📊 <b>Анализ {symbol} ({timeframe})</b>\n\n"
            f"{signals}\n\n"
            f"📈 Последняя цена: ${data['close'].iloc[-1]:.2f}\n"
            f"🔄 RSI: {data['RSI'].iloc[-1]:.2f}\n"
            f"📉 Stochastic: %K={data['slowk'].iloc[-1]:.2f}, %D={data['slowd'].iloc[-1]:.2f}\n"
            f"📏 Волатильность (ATR): {data['ATR'].iloc[-1]:.2f}"
        )
        
        # Отправка результатов
        await update.message.reply_photo(photo=price_plot, caption=report, parse_mode='HTML')
        await update.message.reply_photo(photo=rsi_plot, caption="Индикатор RSI")
        await update.message.reply_photo(photo=stoch_plot, caption="Stochastic Oscillator")
        
    except Exception as e:
        logger.error(f"Ошибка анализа: {str(e)}")
        await update.message.reply_text(f"⚠️ Произошла ошибка при анализе: {str(e)}")
    finally:
        analysis_in_progress = False

# Обработчики для Telegram-бота
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    await update.message.reply_text(
        f"🎮 Привет, {user.first_name}! Я твой умный помощник. ✨\n\n"
        f"🛠️ <b>Доступные функции:</b>\n"
        f"• /gpt - Чат с ИИ (GPT)\n"
        f"• /games - Игровая коллекция\n"
        f"• /duck - Инфо про DUCK × MY × DUCK\n"
        f"• /analyze - Анализ крипторынка\n\n"
        f"💡 Напиши /help для всех команд",
        parse_mode='HTML'
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        "🌈 <b>Доступные команды:</b>\n\n"
        "🎯 /start - Начать работу\n"
        "📚 /help - Эта справка\n"
        "🤖 /gpt [вопрос] - Задать вопрос ИИ\n"
        "📈 /analyze [пара] [таймфрейм] - Анализ крипторынка\n"
        "🎮 /games - Игровая коллекция\n"
        "🦆 /duck - Проект DUCK × MY × DUCK\n"
        "🔁 /echo [текст] - Эхо-ответ\n\n"
        "💬 Просто напиши сообщение для обычного чата"
    )
    await update.message.reply_text(help_text, parse_mode='HTML')

async def games_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    games_text = (
        "🎲 <b>Доступные игры:</b>\n\n"
        "1. <a href='https://example.com/game1'>Космические приключения(тест)</a> 🚀\n"
        "2. <a href='https://example.com/game2'>Загадки древности(тест)</a> 🏛️\n"
        "3. <a href='https://example.com/game3'>Математический квест(тест)</a> ➕\n"
        "4. <a href='https://t.me/duckmyduck_bot?start=r2864597aaec4ecce'>DUCK × MY × DUCK</a> 🦆\n\n"
        "🕹️ Выбери игру и нажми на ссылку, чтобы начать играть!\n"
        "✨ Приятной игры!"
    )
    await update.message.reply_text(games_text, parse_mode='HTML', disable_web_page_preview=False)

async def duck_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    duck_text = (
        "🦆 <b>Как заработать $DMD в DUCK × MY × DUCK</b> 🦆\n\n"
        "🪙 <b>Кормление</b>\nКорми Уток, чтобы они росли и зарабатывали Токены $DMD.\n\n"
        "🥚 <b>Яйца</b>\nПолучай Яйца во время кормления. Соединяй два Яйца.\n\n"
        "🐣 <b>Разведение</b>\nСкрещивай своих Уток с утками друзей.\n\n"
        "🚢 <a href='https://t.me/duckmyduck_official_ru/197'>Летний круиз</a>\n"
        "🧭 <a href='https://telegra.ph/Flipkoiny-07-17'>Как получить и использовать Флипкоины - FAQ </a>\n"
        "📊 <a href='https://t.me/duckmyduck_official_ru/184'>Аукцион - FAQ </a>"
    )
    await update.message.reply_text(duck_text, parse_mode='HTML', disable_web_page_preview=True)

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text.split(' ', 1)[1] if len(update.message.text.split(' ', 1)) > 1 else 'Вы не ввели текст'
    await update.message.reply_text(f"🔊 {text}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text
    await update.message.reply_text(
        f"✉️ Вы написали: <i>\"{text}\"</i>\n"
        f"ℹ️ Используйте /help для списка команд",
        parse_mode='HTML'
    )

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Exception while handling an update:", exc_info=context.error)
# Обработчик команды /ai (основной чат с ИИ)
async def ai_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text(
            "🤖 <b>ИИ Чат</b>\n\n"
            "Отправьте запрос после команды /ai\n"
            "Пример: <code>/ai Как заработать в DUCK × MY × DUCK?</code>\n\n"
            "Доступные модели:\n"
            "- /ai - стандартная модель\n"
            "- /gpt - GPT-3.5 Turbo\n"
            "- /mistral - Mistral 7B",
            parse_mode='HTML'
        )
        return

    prompt = " ".join(context.args)
    await handle_ai_request(update, prompt, DEFAULT_MODEL)

# Обработчик для GPT-3.5
async def gpt_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("ℹ️ Используйте: /gpt [ваш вопрос]")
        return
    await handle_ai_request(update, " ".join(context.args), "tngtech/deepseek-r1t2-chimera:free")

# Общая функция для запросов к ИИ
async def handle_ai_request(update: Update, prompt: str, model: str) -> None:
    await update.message.reply_chat_action(action="typing")
    
    try:
        response = requests.post(
            url=OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/yourusername/yourbot",  # Замените на ваш URL
                "X-Title": "DUCK × MY × DUCK Helper Bot"  # Замените на название вашего бота
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            }
        )

        if response.status_code == 200:
            answer = response.json()["choices"][0]["message"]["content"]
            await update.message.reply_text(f"🤖 <b>Ответ:</b>\n\n{answer}", parse_mode='HTML')
        else:
            error_msg = f"⚠️ Ошибка API (код {response.status_code})"
            logger.error(f"{error_msg}: {response.text}")
            await update.message.reply_text(error_msg)

    except Exception as e:
        logger.error(f"Ошибка в handle_ai_request: {str(e)}")
        await update.message.reply_text("🔴 Произошла ошибка при запросе к ИИ")

# Обработчик команды анализа
async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        help_text = (
            "📈 <b>Использование команды анализа:</b>\n\n"
            "<code>/analyze [пара] [таймфрейм]</code>\n\n"
            "Примеры:\n"
            "<code>/analyze BTC/USDT 15m</code> - анализ Bitcoin/USDT на 15-минутном графике\n"
            "<code>/analyze ETH/USDT 1h</code> - анализ Ethereum/USDT на часовом графике\n\n"
            "Доступные таймфреймы: 1m, 5m, 15m, 30m, 1h, 4h, 1d"
        )
        await update.message.reply_text(help_text, parse_mode='HTML')
        return
    
    # Запускаем анализ
    await analyze_market(update, context)

def main() -> None:
    application = Application.builder().token(TOKEN).build()
    
    # Регистрация обработчиков
    handlers = [
        CommandHandler("start", start),
        CommandHandler("help", help_command),
        CommandHandler("ai", ai_chat),
        CommandHandler("gpt", gpt_chat),
        CommandHandler("games", games_command),
        CommandHandler("duck", duck_command),
        CommandHandler("echo", echo),
        CommandHandler("analyze", analyze_command),
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    ]
    
    for handler in handlers:
        application.add_handler(handler)
    
    application.add_error_handler(error_handler)
    logger.info("🤖 Бот запущен с функцией анализа рынка!")
    application.run_polling()

if __name__ == "__main__":
    main()