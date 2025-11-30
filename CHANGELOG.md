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

Logická chyba v metodě mine:

If self.get_last_block().hash != self.get_last_block().hash:

Stručné vysvětlení:
 * Co to je za chybu?
   * Podmínka porovnává aktuální hash posledního bloku se sebou samým.
   * Tato podmínka je matematicky vždy nepravdivá (X != X je vždy False).

 * Co způsobuje?
   * Tato podmínka měla sloužit jako kontrola, zda mezitím, co miner těžil, nepřišel nový blok od jiného uzlu sítě.
   * Protože je vždy False, kontrola nikdy nezachytí nový blok, který mezitím přišel, a miner tak pokračuje v těžbě na zastaralém/neplatném bloku (na starém konci řetězce), dokud sám blok nevytěžil.
   * To vede k plýtvání výpočetním výkonem a ve verzi 0.1.0 to v kombinaci s další logikou přispívá k zastavení mineru, místo aby se jen restartoval.
