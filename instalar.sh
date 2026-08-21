git clone https://github.com ~/bot_temp
rm -rf ~/bot
mv ~/bot_temp ~/bot
cd ~/bot
python3 -m venv ~/venv_bot
source ~/venv_bot/bin/activate
pip install --upgrade pip
pip install pandas python-binance requests python-dotenv numpy pandas_ta
screen -S boteducativo -X quit >/dev/null 2>&1 || true
screen -dmS boteducativo bash -lc "source ~/venv_bot/bin/activate && cd ~/bot && python3 bot_futuros.py > /home/ubuntu/bot.log 2>&1"
echo "=== INSTALACION COMPLETADA EXITOSAMENTE ==="


