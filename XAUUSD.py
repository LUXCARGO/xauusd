import requests
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestClassifier
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from apscheduler.schedulers.asyncio import AsyncIOScheduler

TOKEN = "8515668400:AAFzSbPthjxnyeSZVjRP1622aUYf7K27vko"
API_KEY = "26d94e78e6354315972a2659b26d4734"
ADMIN_ID = 7849292154  # твой Telegram ID

SYMBOL = "XAU/USD"

vip_users = set()
vip_users.add(ADMIN_ID)

# ================= DATA =================
def get_data(interval):
    url = f"https://api.twelvedata.com/time_series?symbol={SYMBOL}&interval={interval}&apikey={API_KEY}&outputsize=200"
    r = requests.get(url).json()

    if "values" not in r:
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
def train_ai(df):
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    X = df[['rsi','ema','macd']]
    y = df['target']

    model = RandomForestClassifier(n_estimators=300)
    model.fit(X[:-1], y[:-1])
    return model


def generate_signal():
    tf_list = ["5min", "15min", "1h"]
    votes = []
    conf = []

    for tf in tf_list:
        df = get_data(tf)
        if df is None:
            return "⚠ API LIMIT или ошибка данных"

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

    return f"""
🔥 ELITE AI GOLD SIGNAL 🔥

Pair: XAUUSD
Entry: {price}

Direction: {signal}

Stop Loss: {sl}
Take Profit: {tp}

AI Confidence: {confidence}%
"""


# ================= TELEGRAM =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("💎 Elite AI Gold Bot Activated")


async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if user_id not in vip_users:
        await update.message.reply_text("🚫 VIP ACCESS ONLY")
        return

    result = generate_signal()
    await update.message.reply_text(result)


# ================= ADMIN CONTROL =================
async def addvip(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        return

    user_id = int(context.args[0])
    vip_users.add(user_id)
    await update.message.reply_text(f"✅ {user_id} добавлен в VIP")


async def removevip(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        return

    user_id = int(context.args[0])
    vip_users.discard(user_id)
    await update.message.reply_text(f"❌ {user_id} удалён из VIP")


async def viplist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        return

    await update.message.reply_text(f"VIP USERS:\n{vip_users}")


# ================= AUTO SIGNAL =================
async def auto_signal(context: ContextTypes.DEFAULT_TYPE):
    result = generate_signal()

    for user_id in vip_users:
        await context.bot.send_message(chat_id=user_id, text=result)


# ================= RUN =================
app = ApplicationBuilder().token(TOKEN).build()

app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("signal", signal))
app.add_handler(CommandHandler("addvip", addvip))
app.add_handler(CommandHandler("removevip", removevip))
app.add_handler(CommandHandler("viplist", viplist))

scheduler = AsyncIOScheduler()
scheduler.add_job(auto_signal, "interval", minutes=5, args=[app])
scheduler.start()

print("🔥 PRO ELITE AI BOT RUNNING 🔥")
app.run_polling()