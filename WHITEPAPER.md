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

# Technické specifikace systému Droid (DRX Blockchain)

### 1.1 Popis systému
Droid je decentralizovaný blockchainový systém implementovaný v jazyce Python, navržený pro simulaci kryptoměny s tickerem "DRX". Systém podporuje těžbu bloků pomocí Proof-of-Work (PoW), vytváření a validaci transakcí, P2P synchronizaci mezi uzly, správu peněženek a ukládání dat v databázích SQLite. Cílem je poskytnout jednoduchou, bezpečnou a rozšiřitelnou platformu pro experimenty s blockchainovou technologií.

Systém je navržen s ohledem na bezpečnost, odolnost proti útokům (např. DoS, Sybil) a efektivitu (omezení velikosti dat v paměti). Podporuje maximální nabídku 100 000 000 DRX s halvingem každých 495 000 bloků.

### 1.2 Cíle systému
- **Decentralizace**: Podpora P2P sítě pro distribuci bloků a transakcí.
- **Bezpečnost**: Kryptografická ochrana transakcí (ECDSA), validace nonce a zůstatků, ochrana proti double-spending.
- **Efektivita**: Omezení velikosti bloků (1 MB) a mempoolu (10 MB), dynamická úprava obtížnosti.
- **Uživatelská přívětivost**: CLI menu pro interakci, šifrované ukládání peněženek.

### 1.3 Verze a závislosti
- **Verze Pythonu**: 3.12.3
- **Klíčové knihovny**:
  - `ecdsa`: Pro digitální podpisy (SECP256k1 křivka).
  - `hashlib`: Hashovací funkce (SHA3-256).
  - `cryptography`: Šifrování peněženek (AES-GCM, PBKDF2).
  - `sqlite3`: Ukládání blockchainu a mempoolu.
  - `multiprocessing`: Paralelní těžba bloků.
  - `socket`: P2P komunikace.
  - `colorama`: Barevný výstup v konzoli.
  - Další: `json`, `time`, `os`, `threading`, `queue`, `struct`, `getpass`, `heapq`, `collections`, `readline`, `math`, `signal`.

Systém nemá internetový přístup kromě vestavěných proxy pro některé API (není použito v jádru).

## 2. Architektura

### 2.1 Komponenty
- **Wallet**: Správa privátních/veřejných klíčů, generování adres, podepisování transakcí.
- **Transaction**: Struktura transakce (odesílatel, příjemce, částka, poplatek, nonce, timestamp, podpis).
- **Block**: Struktura bloku (index, transakce, Merkle root, previous hash, target, nonce, hash).
- **Blockchain**: Řetězec bloků, mempool, stav (zůstatky, nonce), validace a těžba.
- **P2PNode**: Síťová vrstva pro komunikaci mezi uzly (server, klient, synchronizace).
- **Databáze**:
  - `blockchain.db`: Ukládání bloků (SQLite).
  - `mempool.db`: Dočasné transakce (SQLite).
  - `wallets.json.enc`: Šifrované peněženky (AES-GCM).
  - `peers.json`: Seznam uzlů.

### 2.2 Datové toky
- **Těžba**: Uživatel vybere adresu → Vytvoření coinbase TX → Výběr TX z mempoolu (priorita podle poplatku) → PoW těžba (multiprocessing) → Přidání bloku → Broadcast do P2P.
- **Transakce**: Vytvoření TX → Validace (zůstatek, nonce, podpis) → Přidání do mempoolu → Broadcast do P2P.
- **Synchronizace**: Periodická kontrola (každých 10 s) → Požadavek na info/bloky/řetězec → Nahrazení delšího/lepšího řetězce (kumulativní práce, tie-breaker na hashu).
- **Časová synchronizace**: NTP (pool.ntp.org atd.) pro offset, fallback na read-only režim.

### 2.3 Bezpečnostní architektura
- **Kryptografie**: SHA3-256 pro hashe, ECDSA pro podpisy.
- **Validace**: Každá TX/blok je ověřena (podpis, nonce, zůstatek, timestamp, target).
- **Ochrana proti útokům**:
  - Rate limiting (10 požadavků/s, 100 TX/min).
  - Blacklist pro DoS.
  - Limit peerů (20) proti Sybil.
  - Expirace TX v mempoolu (24 h).
  - Potvrzení bloků (min. 6 pro pevnost).
- **Šifrování**: Peněženky šifrovány heslem (8-20 znaků).

## 3. Klíčové funkce

### 3.1 Těžba bloků (Proof-of-Work)
- **Algoritmus**: SHA3-256 hash musí být < target.
- **Paralelizace**: Používá všechna CPU jádra (uživatelsky nastavitelné).
- **Odměna**: 100 DRX + poplatky, halving každých 495 000 bloků.
- **Obtížnost**: Dynamická úprava každých 10 bloků (cílový čas 600 s).
- **Velikost bloku**: Max. 1 MB, priorita TX podle poplatku (heapq).
- **Podmínky**: Žádný prázdný blok (pokud ALLOW_EMPTY_BLOCKS=False), kontrola připojení k uzlům.

### 3.2 Transakce
- **Struktura**: From/To adresa, částka (min. 0.00000001 DRX), poplatek (0.00000001-0.01 DRX), nonce, timestamp, veřejný klíč, podpis.
- **Validace**: Podpis, nonce sekvence, zůstatek (potvrzený + pending), timestamp (±10 min), duplicitní TX ID/nonce.
- **Mempool**: Prioritní fronta (podle poplatku), expirace 24 h, max. velikost 10 MB.

### 3.3 Peněženky
- **Generování**: ECDSA SECP256k1, adresa = TICKER + SHA3-256(pubkey) + checksum (4 bajty).
- **Ukládání**: Šifrováno AES-GCM s PBKDF2 derivací klíče.
- **Import/Export**: Privátní klíč v hex.

### 3.4 P2P Síť
- **Komunikace**: TCP, JSON zprávy s délkou prefixem (4 bajty), max. 10 MB.
- **Typy zpráv**: transaction, new_block, request_chain_info, response_chain_info, request_blocks, response_blocks, request_full_chain, response_full_chain, request_mempool, response_mempool, new_peer.
- **Synchronizace**: Periodická (10 s), nahrazení na základě kumulativní práce/length/hash.
- **Orphan bloks**: Uložení a resoluce při příchodu rodiče.
- **Bezpečnost**: Rate limiting, blacklist, max. peers.

### 3.5 Ukládání a načítání dat
- **Blockchain**: SQLite, drží jen posledních 100 bloků v RAM.
- **Mempool**: SQLite, čištění každou minutu.
- **Peers**: JSON.

## 4. Algoritmy a výpočty

### 4.1 Hashování
- SHA3-256 pro TX ID, blok hash, Merkle root, adresy.

### 4.2 Merkle Tree
- Rekurzivní hashování TX ID (zdvojení pro lichý počet).

### 4.3 Proof-of-Work
- Nonce inkrementace, hash < target.
- Kumulativní práce: ∑ (2^256 / target) pro řetězec.

### 4.4 Úprava obtížnosti
- Každých 10 bloků: new_target = old_target * (elapsed_time / target_time).
- Min/max: 1 až 2^256 - 1.

### 4.5 Nonce a zůstatky
- Nonce: Sekvenční, kontrola duplicit v bloku/mempoolu/řetězci.
- Zůstatky: Mapa adres → int, aktualizována při přidání bloku.

## 5. Bezpečnostní opatření

- **Proti double-spending**: Nonce, validace zůstatků v bloku (kumulativní).
- **Proti forkům**: Nahrazení delšího řetězce, ale jen pokud fork není starší než 6 potvrzení.
- **Proti DoS**: Rate limiting, max. velikost zpráv/bloků/mempoolu.
- **Proti manipulaci**: Fixní genesis hash/timestamp, NTP synchronizace.
- **Proti read-only**: Pokud NTP selže, zakázána těžba/transakce.
- **Audit**: Logování P2P, validace celého řetězce při nahrazení.

## 6. Konstanty a parametry

- **TICKER**: DRX
- **DECIMALS**: 8
- **MAX_SUPPLY**: 100 000 000 DRX
- **BLOCK_REWARD**: 100 DRX
- **HALVING_INTERVAL_BLOCKS**: 495 000
- **BLOCK_TIME_SECONDS**: 60
- **TX_FEE_MIN/MAX**: 0.00000001 - 0.01 DRX
- **MIN_TX_AMOUNT**: 0.00000001 DRX
- **DIFFICULTY_ADJUSTMENT_INTERVAL**: 10
- **INITIAL_DIFFICULTY_BITS**: 20
- **MAX_BLOCK_SIZE_BYTES**: 1 MB
- **MAX_MEMPOOL_SIZE_BYTES**: 10 MB
- **CONFIRMATIONS_THRESHOLD**: 6
- **MAX_PEERS**: 20
- **RATE_LIMIT_REQUESTS**: 10/s
- **TX_RATE_LIMIT**: 100/min
- **MEMPOOL_TX_EXPIRATION**: 24 h

## 7. Limity a optimalizace

- Držení jen 100 bloků v RAM pro úsporu paměti.
- Paralelní těžba pro rychlost.
- Automatické čištění mempoolu.
- Žádná instalace balíčků (vestavěné).

Tyto specifikace jsou založeny na dostupném kódu a pokrývají klíčové aspekty systému.

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
 
V současné fázi je nutné používat VPN (ZeroTierOne), protože projekt zatím nemá implementované peer discovery, ani šifrování P2P komunikace.  

## Autor
DarkwalkerPrime
