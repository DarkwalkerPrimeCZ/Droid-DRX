# Changelog

## 0.1.0 Alpha
- První alfa verze kryptoměny Droid (DRX)
- Základní blockchain skript (`droid-drx-alpha-0.1.0.py`)
- Blockchain explorer (`blockchain-explorer.py`)
- README a WHITEPAPER s instrukcemi
- CONTRIBUTING.md přidán s informacemi pro přispěvatele
- CHANGELOG.md vytvořen
- Logo projektu a MIT licence

## Známé problémy:

ALLOW_EMPTY_BLOCKS = True (default)

Dochází k zastavení mineru v případě přijetí bloku od jiného uzlu, i když by i v takovém případě měl miner pokračovat dál v těžbě, dokud ho uživatel ručně neukončí pomocí kláves CTRL+C

ALLOW_EMPTY_BLOCKS = False

Dochází k zastavení mineru v případě přijetí bloku od jiného uzlu, i když by i v takovém případě měl miner pokračovat dál v těžbě, dokud jsou transakce v mempoolu, nebo ho uživatel ručně neukončí pomocí kláves CTRL+C
