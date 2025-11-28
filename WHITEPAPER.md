# Droid (DRX) Whitepaper

## Úvod

Droid (DRX) je experimentální CLI account-based kryptoměna napsaná v programovacím jazyce Python a je určená pro běh v aplikaci Termux na operačním systému Android.

Důvodem vzniku této kryptoměny je fakt, že naprostá většina současných kryptoměn je již netěžitelná tradičními způsoby, jelikož běžný PC nemůže konkurovat vysoce výkonným ASIC minerům.

Nejde jen o to, že tento přístup je extrémně neekologický; hlavně je problém v tom, že původním konceptem kryptoměn byla dostupnost, která se s příchodem ASIC minerů zcela vytratila. To, co bylo původně určeno pro všechny, je nyní záležitostí ASIC minerů.

Navíc některé kryptoměny přešly z konsenzuálního mechanismu Proof-of-Work (PoW) na mechanismus Proof-of-Stake (PoS), což stejně jako ASIC mining odřízlo běžné lidi od účasti na produkci nových mincí.

Sice stále existují kryptoměny, jako například VerusCoin, které je možné těžit na běžném PC, ale i v tomto případě je jen otázkou času, než je převálcují ASIC minery, případně dojde k přechodu na PoS. I když je těžba na PC či RIGu energeticky úspornější než pomocí ASIC minerů, stále jde o spotřebu v řádu stovek wattů na jedno zařízení.

Proto se zrodil Droid (DRX) – kryptoměna, která není jen další hračkou, ale jejím cílem je poskytnout skutečnou decentralizaci, svobodu, anonymitu a možnost těžby komukoliv s alespoň jedním zařízením s Androidem – což je dnes téměř každý.

Žádné KYC, žádná třetí strana, žádná nutnost investic do ASIC minerů nebo převodu kapitálu do kryptoměn kvůli stakování. Jen vy, váš Android a svoboda, kterou chtěl už Satoshi Nakamoto.

## Technické specifikace  

## Roadmap — stav vývoje a přechod na finální verzi

V současné fázi (Alpha 0.1.0) je zdrojový kód veřejně dostupný, aby bylo možné rychlé testování, debugování a komunitní přispívání i na stolních počítačích. Toto otevřené vydání slouží výhradně jako raná testovací verze a může obsahovat neúplné funkce a bezpečnostní nedostatky.

Plán pro finální verzi:

- Finální klient bude navržen jako Android/Termux‑only aplikace — implementována bude detekce platformy (ARM / Termux) tak, aby oficiální klient nebyl spustitelný na běžných PC nebo v emulátorech.  
- Bude zavedena integrační kontrola integrity klienta (např. kontrolní součet / checksum), která minimalizuje riziko používání upravených nebo kompromitovaných klientů při interakci s oficiální sítí.  
- Finální verze nebude zpětně kompatibilní s klienty alfa verze; blockchainová historie (ledger) však zůstane zachována, takže data a transakce z alfa fáze budou integrované do finální sítě.  
- Do finální sítě budou implementována opatření proti přijímání transakcí z neoficiálních forků a klonů, aby se zabránilo míchání provozu a záměně transakcí mezi odlišnými instancemi softwaru.  

**Upozornění:** I v současné Alfa 0.1.0 verzi je implementována základní ochrana proti transakcím s upravenými konstantami, takže blockchain odmítá transakce od uzlů, které by se snažily podvádět změnou konstant ve svém kódu.  

## Upozornění
Tento projekt slouží především pro vzdělávání a hobby; nejde o produkt určený pro produkční prostředí.  
V současné fázi je nutné používat VPN (ZeroTierOne), protože projekt zatím nemá implementované peer discovery a šifrování P2P komunikace.  

## Autor
DarkwalkerPrime