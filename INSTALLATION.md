Instalace Droid (DRX) 0.1.0 Alpha 1

## Termux    

Stáhni Termux z F-Droid:      
[https://f-droid.org/packages/com.termux/](https://f-droid.org/packages/com.termux/)    

## Závislosti    

Otevři Termux a zadej:    

```bash    
pkg update && pkg upgrade    
pkg install python-cryptography    
pip install ecdsa colorama    
```    

## Instalace Droid    

1. V Termuxu vytvoř složku a nový soubor:    

```bash    
mkdir droid    
cd droid    
nano droid.py    
```    

2. Otevři soubor `droid-drx-0.1.0-alpha-1.py` v programu Totalcmd-Editor (či ekvivalentu) a zkopíruj obsah do schránky.    

3. Podrž prst na obrazovce, klikni na **Paste** a ulož změny klávesami **CTRL-X → Y → Enter**.    

4. Spusť program:    

```bash    
python droid.py    
```

5. K propojení jednotlivých uzlů doporučuji použít VPN Tailscale, nebo ZeroTierOne, protože projekt zatím nemá implementované peer discovery, ani šifrování P2P komunikace.

6. Pro zastavení mineru stiskni klávesy **CTRL+C**

Poznámka: Stisknutí kláves **CTRL+C** v hlavním menu programu ukončí celý program, stejně jako klávesa 12 v hlavním menu programu.
   
---    

# Instalace Blockchain Exploreru    

## Závislosti    

```bash    
pip install flask    
```    

1. V Termuxu vytvoř soubor:    

```bash    
cd droid    
nano be.py    
```    

2. Otevři soubor `blockchain-explorer.py` v Totalcmd-Editoru (či ekvivalentu) a zkopíruj obsah do schránky.    

3. Podrž prst na obrazovce, klikni na **Paste** a ulož změny klávesami **CTRL-X → Y → Enter**.    

4. Spusť program:    

```bash    
python be.py    
```    

5. Otevři IP adresu Flask serveru ve svém webovém prohlížeči.    

6. Pro ukončení Flask serveru stiskni **CTRL+C**.    

---    

**Autor:** DarkwalkerPrime
