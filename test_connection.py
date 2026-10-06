import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

load_dotenv()

api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")
paper = os.getenv("ALPACA_PAPER", "True").lower() == "true"

trading_client = TradingClient(api_key, secret_key, paper=paper)

account = trading_client.get_account()

print("====================================")
print("  ESTADO DE CONEXIÓN CON ALPACA     ")
print("====================================")
print(f"Estado de la cuenta : {account.status}")
print(f"Balance en efectivo : ${float(account.cash):,.2f} USD")
print(f"Poder de compra     : ${float(account.buying_power):,.2f} USD")
print(f"Cuenta bloqueada    : {account.account_blocked}")
print("====================================")