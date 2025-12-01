# Changelog

## 0.1.0 Alpha
- První alfa verze kryptoměny Droid (DRX)
- Základní blockchain skript (`droid-drx-alpha-0.1.0.py`)
- Blockchain explorer (`blockchain-explorer.py`)
- README a WHITEPAPER s instrukcemi
- CONTRIBUTING.md přidán s informacemi pro přispěvatele
- CHANGELOG.md vytvořen
- Logo projektu a MIT licence

Známé problémy:

Miner se zastaví když dojde ke ztrátě spojení se všemi dostupnými uzly, či se všemi dostupnými NTP servery, nebo když nejsou žádné transakce v mempoolu při nastavení ALLOW_EMPTY_BLOCKS = False

To je sice očekávané chování, ale chybí zde automatické znovuspuštění mineru, takže v takových případech je nutné miner spustit ručně.
