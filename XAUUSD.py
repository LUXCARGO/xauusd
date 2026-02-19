# ================= IMPORTS =================
# ➕ ДОБАВЛЕНЫ: os, json, logging, datetime
import os
import json
import logging
import requests
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from apscheduler.schedulers.asyncio import AsyncIOScheduler


# ================= ENV =================
# 🔁 ИЗМЕНЁН (теперь через Railway Variables)
TOKEN = os.getenv("BOT_TOKEN")
API_KEY = os.getenv("API_KEY")
ADMIN_ID = int(os.getenv("ADMIN_ID"))

SYMBOL = "XAU/USD"
VIP_FILE = "vip_users.json"


# ================= LOGGING =================
# ➕ ДОБАВЛЕН
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# ================= VIP STORAGE =================
# ➕ ДОБАВЛЕН (раньше VIP не сохранялись в файл)
def load_vips():
    if not os.path.exists(VIP_FILE):
        return {ADMIN_ID}
    with open(VIP_FILE, "r") as f:
        return set(json.load(f))

def save_vips(vips):
    with open(VIP_FILE, "w") as f:
        json.dump(list(vips), f)

vip_users = load_vips()
vip_users.add(ADMIN_ID)
save_vips(vip_users)


# ================= CACHE =================
# ➕ ДОБАВЛЕН (раньше каждый раз обучалась модель)
signal_cache = {
    "data": None,
    "timestamp": None
}
CACHE_DURATION = 4  # минут


# ================= DATA =================
# 🔁 ИЗМЕНЁН (добавлено логирование ошибок)
def get_data(interval):
    url = f"https://api.twelvedata.com/time_series?symbol={SYMBOL}&interval={interval}&apikey={API_KEY}&outputsize=200"
    r = requests.get(url).json()

    if "values" not in r:
        logger.error("API Error or limit reached")
        return None

    df = pd.DataFrame(r["values"]).astype(float)
    df = df.iloc[::-1]

    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    df['ema'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['atr'] = ta.volatility.AverageTrueRange(
        df['high'], df['low'], df['close']).average_true_range()

    df = df.dropna()
    return df


# ================= AI =================
# ✅ БЕЗ ИЗМЕНЕНИЙ (логика осталась)
def train_ai(df):
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    X = df[['rsi','ema','macd']]
    y = df['target']

    model = RandomForestClassifier(n_estimators=300)
    model.fit(X[:-1], y[:-1])
    return model


# ================= SIGNAL =================
# 🔁 ИЗМЕНЁН (добавлено кеширование + логирование)
def generate_signal():
    global signal_cache

    if signal_cache["timestamp"] and \
       datetime.now() - signal_cache["timestamp"] < timedelta(minutes=CACHE_DURATION):
        logger.info("Using cached signal")
        return signal_cache["data"]

    logger.info("Generating new signal")

    tf_list = ["5min", "15min", "1h"]
    votes = []
    conf = []

    for tf in tf_list:
        df = get_data(tf)
        if df is None:
            return "⚠ API ERROR"

        model = train_ai(df)
        last = df[['rsi','ema','macd']].iloc[-1:]
        prob = model.predict_proba(last)[0]

        votes.append(np.argmax(prob))
        conf.append(max(prob))

    direction = round(np.mean(votes))
    confidence = round(np.mean(conf)*100,2)

    df_main = get_data("5min")
    price = df_main['close'].iloc[-1]
    atr = df_main['atr'].iloc[-1]

    if direction == 1:
        signal = "📈 BUY"
        sl = round(price - atr*1.5,2)
        tp = round(price + atr*3,2)
    else:
        signal = "📉 SELL"
        sl = round(price + atr*1.5,2)
        tp = round(price - atr*3,2)

    result = f"""
🔥 ELITE AI GOLD SIGNAL 🔥

Pair: XAUUSD
Entry: {price}

Direction: {signal}

Stop Loss: {sl}
Take Profit: {tp}

AI Confidence: {confidence}%
"""

    signal_cache["data"] = result
    signal_cache["timestamp"] = datetime.now()

    return result


# ================= TELEGRAM =================
# ✅ Логика почти без изменений
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("💎 Elite AI Bot Active")

async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if user_id not in vip_users:
        await update.message.reply_text("🚫 VIP ONLY")
        return

    result = generate_signal()
    await update.message.reply_text(result)


# ================= ADMIN CONTROL =================
# 🔁 ИЗМЕНЁН (добавлено сохранение VIP в файл)
async def addvip(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        return

    user_id = int(context.args[0])
    vip_users.add(user_id)
    save_vips(vip_users)
    await update.message.reply_text(f"✅ {user_id} added to VIP")

async def removevip(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        return

    user_id = int(context.args[0])
    vip_users.discard(user_id)
    save_vips(vip_users)
    await update.message.reply_text(f"❌ {user_id} removed from VIP")

async def viplist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        return

    await update.message.reply_text(f"VIP USERS:\n{vip_users}")


# ================= AUTO SIGNAL =================
# ✅ Логика та же
async def auto_signal(context: ContextTypes.DEFAULT_TYPE):
    result = generate_signal()
    for user_id in vip_users:
        await context.bot.send_message(chat_id=user_id, text=result)


# ================= RUN =================
# 🔁 ИЗМЕНЁН (Railway-safe)
app = ApplicationBuilder().token(TOKEN).build()

app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("signal", signal))
app.add_handler(CommandHandler("addvip", addvip))
app.add_handler(CommandHandler("removevip", removevip))
app.add_handler(CommandHandler("viplist", viplist))

scheduler = AsyncIOScheduler()
scheduler.add_job(auto_signal, "interval", minutes=5, args=[app])
scheduler.start()

logger.info("🚀 ELITE AI BOT STARTED")
app.run_polling()
