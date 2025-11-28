import sys
import json
import time
import binascii
import os
import sqlite3
import hashlib
from flask import Flask, request, url_for, Response

# --- KONFIGURACE A KONSTANTY (Zkopírováno z merkle root.txt) ---
PROJECT_NAME = "Droid"
TICKER = "DRX"
DECIMALS = 8
MAX_SUPPLY = 100_000_000 * (10 ** DECIMALS)
BLOCK_REWARD = 100 * (10 ** DECIMALS)
HALVING_INTERVAL_BLOCKS = 495_000
BLOCK_TIME_SECONDS = 60
TX_FEE_MIN = int(0.00000001 * (10 ** DECIMALS))
TX_FEE_MAX = int(0.01 * (10 ** DECIMALS))
MIN_TX_AMOUNT = int(0.00000001 * (10 ** DECIMALS))
DIFFICULTY_ADJUSTMENT_INTERVAL = 10
TARGET_BLOCK_TIME = BLOCK_TIME_SECONDS * (DIFFICULTY_ADJUSTMENT_INTERVAL)
# V nové verzi se používá INITIAL_DIFFICULTY_BITS a FIXED_TARGET
INITIAL_DIFFICULTY_BITS = 20
FIXED_TARGET = (1 << 256) >> INITIAL_DIFFICULTY_BITS

BLOCKCHAIN_DB = 'blockchain.db'
MEMPOOL_DB = 'mempool.db'
LAST_BLOCKS_TO_KEEP = 100
CONFIRMATIONS_THRESHOLD = 6
MAX_BLOCK_SIZE_BYTES = 1 * 1024 * 1024
MAX_MEMPOOL_SIZE_BYTES = 10 * 1024 * 1024
GENESIS_ADDRESS = "DRXfc3d428153b2c71be82e84a04c8b70b3d5153c75cc7c75edc3323d6e9c7cb8d5"
GENESIS_TIMESTAMP = 1762819200
GENESIS_AMOUNT = 1_000_000 * (10 ** DECIMALS)
# Hash genesis bloku z nové verze
GENESIS_BLOCK_EXPECTED_HASH = "00000a22d1972da3eaaf1d9627dcc2d04d326d5a5030cd43f842d8b5f1bfd713"


# --- POMOCNÉ FUNKCE ---

def compute_merkle_root(transactions):
    """Vypoèítá Merkle Root z transakcí."""
    if not transactions:
        return hashlib.sha3_256(b'').hexdigest()

    tx_hashes = [tx.tx_id for tx in transactions]

    while len(tx_hashes) > 1:
        if len(tx_hashes) % 2 != 0:
            tx_hashes.append(tx_hashes[-1])

        new_hashes = []
        for i in range(0, len(tx_hashes), 2):
            combined = tx_hashes[i] + tx_hashes[i + 1]
            new_hash = hashlib.sha3_256(combined.encode()).hexdigest()
            new_hashes.append(new_hash)
        tx_hashes = new_hashes

    return tx_hashes[0]

# --- TØÍDY ---

class Transaction:
    def __init__(self, from_address, to_address, amount, fee=0, nonce=0, public_key=None, signature=None, timestamp=None, tx_id=None):
        self.from_address = from_address
        self.to_address = to_address
        self.amount = amount
        self.fee = fee
        self.nonce = nonce
        self.timestamp = timestamp or time.time()
        self.public_key = public_key
        self.signature = signature
        self.tx_id = tx_id or self.compute_hash()

    def to_dict_for_signing(self):
        return {
            'from_address': self.from_address,
            'to_address': self.to_address,
            'amount': self.amount,
            'fee': self.fee,
            'timestamp': self.timestamp,
            'nonce': self.nonce
        }

    def compute_hash(self):
        data = self.to_dict_for_signing()
        data_string = json.dumps(data, sort_keys=True)
        return hashlib.sha3_256(data_string.encode()).hexdigest()

    def to_dict(self):
        return {
            'from_address': self.from_address,
            'to_address': self.to_address,
            'amount': self.amount,
            'fee': self.fee,
            'timestamp': self.timestamp,
            'nonce': self.nonce,
            'public_key': self.public_key,
            'signature': self.signature,
            'tx_id': self.tx_id,
        }

    @staticmethod
    def from_dict(data):
        tx = Transaction(
            data['from_address'],
            data['to_address'],
            data['amount'],
            data['fee'],
            nonce=data.get('nonce', 0),
            public_key=data.get('public_key'),
            signature=data.get('signature'),
            timestamp=data.get('timestamp')
        )
        tx.tx_id = data.get('tx_id') or tx.compute_hash()
        return tx

    def get_size(self):
        return len(json.dumps(self.to_dict()).encode('utf-8'))

class Block:
    # Aktualizováno pro podporu target a merkle_root
    def __init__(self, index, transactions, previous_hash, target, nonce=0, timestamp=None, merkle_root=None):
        self.index = index
        self.timestamp = timestamp or time.time()
        self.transactions = transactions
        self.merkle_root = merkle_root or compute_merkle_root(transactions)
        self.previous_hash = previous_hash
        self.target = target
        self.nonce = nonce
        self.hash = self.compute_hash()

    def compute_hash(self):
        block_dict = {
            'index': self.index,
            'timestamp': self.timestamp,
            'merkle_root': self.merkle_root,
            'previous_hash': self.previous_hash,
            'target': hex(self.target)[2:], # Ukládá se jako hex string bez 0x
            'nonce': self.nonce
        }
        block_string = json.dumps(block_dict, sort_keys=True)
        return hashlib.sha3_256(block_string.encode()).hexdigest()

    def get_size(self):
        return len(json.dumps(self.to_dict()).encode('utf-8'))

    def to_dict(self):
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'transactions': [tx.to_dict() for tx in self.transactions],
            'merkle_root': self.merkle_root,
            'previous_hash': self.previous_hash,
            'target': hex(self.target)[2:],
            'nonce': self.nonce,
            'hash': self.hash
        }

    @staticmethod
    def from_dict(data):
        transactions = [Transaction.from_dict(tx_data) for tx_data in data['transactions']]
        # Ošetøení targetu (mùže být int nebo hex string)
        target_raw = data['target']
        if isinstance(target_raw, str):
            target = int(target_raw, 16)
        else:
            target = target_raw
            
        block = Block(
            data['index'], 
            transactions, 
            data['previous_hash'], 
            target, 
            data['nonce'], 
            timestamp=data['timestamp'],
            merkle_root=data.get('merkle_root')
        )
        block.hash = data['hash']
        return block

class Blockchain:
    def __init__(self, create_genesis=True):
        self.chain = []
        self.max_block_index = 0
        self.unconfirmed_transactions = []
        self.all_tx_ids = set()
        self.balance_map = {}
        if create_genesis:
            self.create_genesis_block()

    def create_genesis_block(self):
        genesis_tx = Transaction(
            from_address="COINBASE",
            to_address=GENESIS_ADDRESS,
            amount=GENESIS_AMOUNT,
            fee=0,
            nonce=0,
            public_key="COINBASE",
            signature="COINBASE",
            timestamp=GENESIS_TIMESTAMP
        )
        # Genesis blok v nové verzi má nonce 2748594
        genesis_block = Block(0, [genesis_tx], "0", FIXED_TARGET, nonce=2748594, timestamp=GENESIS_TIMESTAMP)
        self.chain.append(genesis_block)
        self.max_block_index = 0
        self.all_tx_ids.add(genesis_tx.tx_id)
        self.update_state_with_block(genesis_block)

    def update_state_with_block(self, block):
        for tx in block.transactions:
            self.all_tx_ids.add(tx.tx_id)
            if tx.from_address == "COINBASE":
                self.balance_map[tx.to_address] = self.balance_map.get(tx.to_address, 0) + tx.amount
            else:
                self.balance_map[tx.from_address] = self.balance_map.get(tx.from_address, 0) - tx.amount - tx.fee
                self.balance_map[tx.to_address] = self.balance_map.get(tx.to_address, 0) + tx.amount

    def rebuild_state(self):
        conn = sqlite3.connect(BLOCKCHAIN_DB)
        c = conn.cursor()
        c.execute("SELECT * FROM blocks ORDER BY block_index")
        self.balance_map = {}
        self.all_tx_ids = set()
        for row in c:
            # Nové schéma DB: 
            # 0:index, 1:time, 2:txs, 3:prev_hash, 4:target_hex, 5:nonce, 6:hash, 7:merkle_root
            block_data = {
                'index': row[0],
                'timestamp': row[1],
                'transactions': json.loads(row[2]),
                'previous_hash': row[3],
                'target': row[4],
                'nonce': row[5],
                'hash': row[6],
                'merkle_root': row[7]
            }
            block = Block.from_dict(block_data)
            self.update_state_with_block(block)
        conn.close()

    def get_block_from_db(self, index):
        conn = sqlite3.connect(BLOCKCHAIN_DB)
        c = conn.cursor()
        c.execute("SELECT * FROM blocks WHERE block_index = ?", (index,))
        row = c.fetchone()
        conn.close()
        if row:
            block_data = {
                'index': row[0],
                'timestamp': row[1],
                'transactions': json.loads(row[2]),
                'previous_hash': row[3],
                'target': row[4],
                'nonce': row[5],
                'hash': row[6],
                'merkle_root': row[7]
            }
            return Block.from_dict(block_data)
        return None

    def get_confirmed_balance(self, wallet_address):
        return self.balance_map.get(wallet_address, 0)

    def get_total_supply(self):
        total_supply = 0
        # Pouze odhad založený na max_index, protože explorer nemusí mít všechny bloky v RAM
        for i in range(self.max_block_index + 1):
            halvings = i // HALVING_INTERVAL_BLOCKS
            if i == 0:
                subsidy = GENESIS_AMOUNT
            else:
                subsidy = BLOCK_REWARD // (2 ** halvings)
            total_supply += subsidy
        return total_supply

    def get_confirmations(self, block_hash):
        conn = sqlite3.connect(BLOCKCHAIN_DB)
        c = conn.cursor()
        c.execute("SELECT block_index FROM blocks WHERE block_hash = ?", (block_hash,))
        row = c.fetchone()
        conn.close()
        if row:
            block_index = row[0]
            return self.max_block_index - block_index
        return 0

    def find_transaction_by_id(self, tx_id):
        for tx in self.unconfirmed_transactions:
            if tx.tx_id == tx_id:
                return tx, "Mempool"
        conn = sqlite3.connect(BLOCKCHAIN_DB)
        c = conn.cursor()
        c.execute("SELECT block_index, transactions FROM blocks")
        for row in c:
            transactions = json.loads(row[1])
            for tx_data in transactions:
                if tx_data['tx_id'] == tx_id:
                    conn.close()
                    return Transaction.from_dict(tx_data), f"Blok #{row[0]}"
        conn.close()
        return None, None

# --- NAÈÍTÁNÍ DAT ---

def load_data():
    droid_chain = Blockchain(create_genesis=False)
    if os.path.exists(BLOCKCHAIN_DB):
        try:
            conn = sqlite3.connect(BLOCKCHAIN_DB)
            c = conn.cursor()
            # Schéma musí odpovídat `merkle root.txt`
            c.execute('''
                CREATE TABLE IF NOT EXISTS blocks (
                    block_index INTEGER PRIMARY KEY,
                    timestamp REAL,
                    transactions TEXT,
                    previous_hash TEXT,
                    target_hex TEXT,
                    nonce INTEGER,
                    block_hash TEXT,
                    merkle_root TEXT
                )
            ''')
            
            c.execute("SELECT MAX(block_index) FROM blocks")
            result = c.fetchone()
            droid_chain.max_block_index = result[0] if result and result[0] is not None else 0
            
            c.execute("SELECT * FROM blocks WHERE block_index > ? ORDER BY block_index", (droid_chain.max_block_index - LAST_BLOCKS_TO_KEEP,))
            rows = c.fetchall()
            
            droid_chain.chain = []
            for row in rows:
                block_data = {
                    'index': row[0],
                    'timestamp': row[1],
                    'transactions': json.loads(row[2]),
                    'previous_hash': row[3],
                    'target': row[4],
                    'nonce': row[5],
                    'hash': row[6],
                    'merkle_root': row[7]
                }
                droid_chain.chain.append(Block.from_dict(block_data))
            
            droid_chain.rebuild_state()
            conn.close()
        except sqlite3.Error as e:
            print(f"Chyba pøi naèítání blockchain DB: {e}")

    droid_chain.unconfirmed_transactions = load_mempool(droid_chain)
    return droid_chain

def load_mempool(droid_chain):
    unconfirmed_transactions = []
    if os.path.exists(MEMPOOL_DB):
        try:
            conn = sqlite3.connect(MEMPOOL_DB)
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    tx_id TEXT PRIMARY KEY,
                    from_address TEXT,
                    to_address TEXT,
                    amount INTEGER,
                    fee INTEGER,
                    nonce INTEGER,
                    timestamp REAL,
                    public_key TEXT,
                    signature TEXT
                )
            ''')
            c.execute("SELECT * FROM transactions")
            rows = c.fetchall()
            conn.close()
            for row in rows:
                tx_data = {
                    'tx_id': row[0],
                    'from_address': row[1],
                    'to_address': row[2],
                    'amount': row[3],
                    'fee': row[4],
                    'nonce': row[5],
                    'timestamp': row[6],
                    'public_key': row[7],
                    'signature': row[8]
                }
                tx = Transaction.from_dict(tx_data)
                unconfirmed_transactions.append(tx)
        except sqlite3.Error as e:
            print(f"Chyba pøi naèítání mempoolu: {e}")
    return unconfirmed_transactions

# Inicializace blockchainu
droid_chain = load_data()

app = Flask(__name__)

# --- WEB UI (FLASK) ---

CSS = """
<style>
    body {
        background-color: black;
        color: white;
        font-family: Arial, sans-serif;
        margin: 0;
        padding: 10px;
        word-wrap: break-word;
    }
    input[type="text"] {
        width: 100%;
        padding: 10px;
        margin-bottom: 10px;
        box-sizing: border-box;
    }
    button {
        padding: 10px;
        width: 100%;
        background-color: #333;
        color: white;
        border: none;
        cursor: pointer;
    }
    .section {
        margin-bottom: 20px;
    }
    .block, .tx, .history {
        border: 1px solid #444;
        padding: 10px;
        margin-bottom: 10px;
    }
    a { color: cyan; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .pagination { margin-top: 10px; }
    .pagination a { margin: 0 5px; }
    .pagination .current { font-weight: bold; }
    @media (max-width: 600px) {
        body { padding: 5px; font-size: 14px; }
    }
</style>
"""

def format_amount(amount):
    return f"{format(amount / (10 ** DECIMALS), f'.{DECIMALS}f')} {TICKER}"

def generate_pagination(current_page, total_pages, base_url):
    html = "<div class='pagination'>"
    separator = '&' if '?' in base_url else '?'
    if current_page > 1:
        html += f"<a href='{base_url}{separator}page={current_page-1}'>&lt;&lt; Pøedchozí</a>"
    
    pages_to_show = set([1, current_page-1, current_page, current_page+1, total_pages])
    last_p = 0
    for p in sorted(pages_to_show):
        if p < 1 or p > total_pages:
            continue
        if p - last_p > 1:
            html += " ... "
        if p == current_page:
            html += f"<span class='current'>{p}</span>"
        else:
            html += f"<a href='{base_url}{separator}page={p}'>{p}</a>"
        last_p = p
        
    if current_page < total_pages:
        html += f"<a href='{base_url}{separator}page={current_page+1}'>Další &gt;&gt;</a>"
    html += "</div>"
    return html

def get_mempool_html(page=1):
    ITEMS_PER_PAGE = 10
    html = "<div class='section'><h2>Nepotvrzené transakce</h2>"
    mempool_size = sum(tx.get_size() for tx in droid_chain.unconfirmed_transactions)
    html += f"Velikost mempoolu: {mempool_size / 1024:.2f} KB / {MAX_MEMPOOL_SIZE_BYTES / 1024 / 1024:.2f} MB<br>"
    total_txs = len(droid_chain.unconfirmed_transactions)
    total_pages = max(1, (total_txs + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
    start = (page - 1) * ITEMS_PER_PAGE
    end = start + ITEMS_PER_PAGE
    
    sorted_mempool = sorted(droid_chain.unconfirmed_transactions, key=lambda x: x.fee, reverse=True)
    transactions = sorted_mempool[start:end]
    
    if not transactions:
        html += "Mempool je prázdný."
    else:
        for tx in transactions:
            html += f"<div class='tx'>TX ID: <a href='/tx/{tx.tx_id}'>{tx.tx_id}</a><br>"
            html += f"Od: <a href='/address/{tx.from_address}'>{tx.from_address}</a><br>"
            html += f"Komu: <a href='/address/{tx.to_address}'>{tx.to_address}</a><br>"
            html += f"Èástka: {format_amount(tx.amount)}<br>"
            html += f"Poplatek: {format_amount(tx.fee)}<br>"
            html += f"Nonce: {tx.nonce}<br>"
            html += f"Èas: {time.strftime('%d.%m.%Y %H:%M:%S UTC+00:00', time.gmtime(tx.timestamp))}<br>"
            html += f"Podpis: {tx.signature}</div>"
            
    if total_pages > 1:
        html += generate_pagination(page, total_pages, "/?section=mempool")
    html += "</div>"
    return html

def get_chain_status_html():
    total_size = 0
    total_blocks = 0
    total_size_bytes = 0
    if os.path.exists(BLOCKCHAIN_DB):
        try:
            conn = sqlite3.connect(BLOCKCHAIN_DB)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM blocks")
            total_blocks = c.fetchone()[0]
            c.execute("SELECT SUM(LENGTH(transactions)) FROM blocks")
            total_size_bytes = c.fetchone()[0] or 0
            conn.close()
        except sqlite3.Error:
            pass
            
    total_size_kb = total_size_bytes / 1024
    total_size_mb = total_size_kb / 1024
    total_txs = len(droid_chain.all_tx_ids)
    
    html = f"<div class='section'>Velikost blockchainu: {total_size_kb:.2f} KB / {total_size_mb:.2f} MB<br>"
    html += f"Celkový poèet blokù: {total_blocks}<br>"
    html += f"Celkový poèet transakcí: {total_txs}</div>"
    return html

def get_chain_html(page=1):
    ITEMS_PER_PAGE = 10
    html = "<div class='section'><h2>Blockchain</h2>"
    total_blocks = 0
    
    if os.path.exists(BLOCKCHAIN_DB):
        try:
            conn = sqlite3.connect(BLOCKCHAIN_DB)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM blocks")
            total_blocks = c.fetchone()[0]
            total_pages = max(1, (total_blocks + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
            offset = (page - 1) * ITEMS_PER_PAGE
            c.execute("SELECT * FROM blocks ORDER BY block_index DESC LIMIT ? OFFSET ?", (ITEMS_PER_PAGE, offset))
            rows = c.fetchall()
            conn.close()
            
            for row in rows:
                block_data = {
                    'index': row[0],
                    'timestamp': row[1],
                    'transactions': json.loads(row[2]),
                    'previous_hash': row[3],
                    'target': row[4],     # Nový sloupec
                    'nonce': row[5],
                    'hash': row[6],
                    'merkle_root': row[7] # Nový sloupec
                }
                block = Block.from_dict(block_data)
                
                html += f"<div class='block'>Blok <a href='/block/{block.index}'>#{block.index}</a><br>"
                html += f"Hash: {block.hash}<br>"
                html += f"Merkle Root: {block.merkle_root}<br>"
                html += f"Target (hex): {hex(block.target)[2:]}<br>"
                html += f"Pøedchozí hash: {block.previous_hash}<br>"
                html += f"PoW Nonce: {block.nonce}<br>"
                html += f"Èas: {time.strftime('%d.%m.%Y %H:%M:%S UTC+00:00', time.gmtime(block.timestamp))}<br>"
                html += f"Velikost bloku: {block.get_size() / 1024:.2f} KB<br>"
                html += f"Poèet potvrzení: {droid_chain.get_confirmations(block.hash)}<br>"
                html += f"Poèet transakcí: {len(block.transactions)}<br>"
                html += "Transakce:<br>"
                for tx in block.transactions:
                    html += f"- TX ID: <a href='/tx/{tx.tx_id}'>{tx.tx_id}</a><br>"
                    html += f"  Od: <a href='/address/{tx.from_address}'>{tx.from_address}</a><br>"
                    html += f"  Komu: <a href='/address/{tx.to_address}'>{tx.to_address}</a><br>"
                    html += f"  Èástka: {format_amount(tx.amount)}<br>"
                    if tx.from_address != "COINBASE":
                        html += f"  Poplatek: {format_amount(tx.fee)}<br>"
                        html += f"  TX Nonce: {tx.nonce}<br>"
                    html += f"  Podpis: {tx.signature}<br>"
                html += "</div>"
            
            if total_pages > 1:
                html += generate_pagination(page, total_pages, "/?section=chain")
        except sqlite3.Error as e:
            html += f"Chyba pøi ètení databáze: {e}"
            
    if total_blocks == 0:
        html += "Žádné bloky k zobrazení. (Databáze 'blockchain.db' nebyla nalezena nebo je prázdná)"
    html += "</div>"
    return html

@app.route('/')
def index():
    global droid_chain
    droid_chain = load_data()
    section = request.args.get('section')
    page = int(request.args.get('page', 1))
    
    mempool_page = 1
    chain_page = 1
    if section == 'mempool':
        mempool_page = page
    elif section == 'chain':
        chain_page = page

    html = "<html><head><title>Droid Blockchain Explorer</title><meta name='viewport' content='width=device-width, initial-scale=1.0'>" + CSS + "</head><body>"
    html += "<h1>Droid (DRX) Blockchain Explorer</h1>"
    html += "<form action='/search' method='GET'>"
    html += "<input type='text' name='query' placeholder='Vyhledat blok, TXID nebo adresu'>"
    html += "<button type='submit'>Vyhledat</button></form>"
    html += get_chain_status_html()
    html += get_mempool_html(mempool_page)
    html += get_chain_html(chain_page)
    html += "<div class='section'><form action='/export' method='GET' style='margin-top: 20px;'><button type='submit'>Exportovat Blockchain jako txt</button></form></div>"
    html += "</body></html>"
    return html

@app.route('/search')
def search():
    global droid_chain
    droid_chain = load_data()
    query = request.args.get('query', '').strip()
    page = int(request.args.get('page', 1))
    if not query:
        return index()

    html = "<html><head><title>Výsledky vyhledávání</title><meta name='viewport' content='width=device-width, initial-scale=1.0'>" + CSS + "</head><body>"
    html += f"<h2>Výsledky pro: {query}</h2>"
    html += "<a href='/'>Zpìt na hlavní stránku</a><br><br>"

    # 1. Zkusit Blok Index
    try:
        block_index = int(query)
        block = droid_chain.get_block_from_db(block_index)
        if block:
            html += f"<div class='block'>Blok #{block.index}<br>"
            html += f"Hash: {block.hash}<br>"
            html += f"Merkle Root: {block.merkle_root}<br>"
            html += f"Target (hex): {hex(block.target)[2:]}<br>"
            html += f"Pøedchozí hash: {block.previous_hash}<br>"
            html += f"PoW Nonce: {block.nonce}<br>"
            html += f"Èas: {time.strftime('%d.%m.%Y %H:%M:%S UTC+00:00', time.gmtime(block.timestamp))}<br>"
            html += f"Velikost bloku: {block.get_size() / 1024:.2f} KB<br>"
            html += f"Poèet potvrzení: {droid_chain.get_confirmations(block.hash)}<br>"
            html += f"Poèet transakcí: {len(block.transactions)}<br>"
            html += "Transakce:<br>"
            for tx in block.transactions:
                html += f"- TX ID: <a href='/tx/{tx.tx_id}'>{tx.tx_id}</a><br>"
                html += f"  Od: <a href='/address/{tx.from_address}'>{tx.from_address}</a><br>"
                html += f"  Komu: <a href='/address/{tx.to_address}'>{tx.to_address}</a><br>"
                html += f"  Èástka: {format_amount(tx.amount)}<br>"
                if tx.from_address != "COINBASE":
                    html += f"  Poplatek: {format_amount(tx.fee)}<br>"
                    html += f"  TX Nonce: {tx.nonce}<br>"
                html += f"  Podpis: {tx.signature}<br>"
            html += "</div>"
            return html + "</body></html>"
    except ValueError:
        pass

    # 2. Zkusit TXID
    tx, location = droid_chain.find_transaction_by_id(query)
    if tx:
        html += f"<div class='tx'>--- Detaily transakce (TX ID: {tx.tx_id}) ---<br>"
        html += f"Stav: Nalezena v {location}<br>"
        html += f"Od: <a href='/address/{tx.from_address}'>{tx.from_address}</a><br>"
        html += f"Komu: <a href='/address/{tx.to_address}'>{tx.to_address}</a><br>"
        html += f"Èástka: {format_amount(tx.amount)}<br>"
        html += f"Poplatek: {format_amount(tx.fee)}<br>"
        html += f"Nonce: {tx.nonce}<br>"
        html += f"Èas: {time.strftime('%d.%m.%Y %H:%M:%S UTC+00:00', time.gmtime(tx.timestamp))}<br>"
        html += f"Veøejný klíè: {tx.public_key}<br>"
        html += f"Podpis: {tx.signature}</div>"
        return html + "</body></html>"

    # 3. Zkusit Adresu
    ITEMS_PER_PAGE = 10
    confirmed_balance = droid_chain.get_confirmed_balance(query)
    html += f"<div class='history'>--- Historie transakcí pro adresu '{query}' ---<br>"
    html += f'<span style="color: purple;">Potvrzený zùstatek</span>: {format_amount(confirmed_balance)}<br>'
    tx_history = []
    
    if os.path.exists(BLOCKCHAIN_DB):
        try:
            conn = sqlite3.connect(BLOCKCHAIN_DB)
            c = conn.cursor()
            c.execute("SELECT block_index, transactions FROM blocks ORDER BY block_index DESC")
            for row in c:
                transactions = json.loads(row[1])
                for tx_data in transactions:
                    tx = Transaction.from_dict(tx_data)
                    if tx.from_address == query or tx.to_address == query:
                        tx_history.append((row[0], tx))
            conn.close()
        except sqlite3.Error as e:
            html += f"Chyba pøi ètení DB: {e}<br>"
            
    total_txs = len(tx_history)
    total_pages = max(1, (total_txs + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
    start = (page - 1) * ITEMS_PER_PAGE
    end = start + ITEMS_PER_PAGE
    tx_found = False
    
    for block_index, tx in tx_history[start:end]:
        tx_found = True
        if tx.from_address == query:
            direction_html = '<span style="color: red;">Odesláno</span>'
        else:
            direction_html = '<span style="color: green;">Pøijato</span>'
        html += f"TX ID: <a href='/tx/{tx.tx_id}'>{tx.tx_id}</a><br>"
        html += f"Blok: <a href='/block/{block_index}'>#{block_index}</a><br>"
        html += f"Smìr: {direction_html}<br>"
        html += f"Od: {tx.from_address}<br>"
        html += f"Komu: {tx.to_address}<br>"
        html += f"Èástka: {format_amount(tx.amount)}<br>"
        if tx.from_address != "COINBASE":
            html += f"Poplatek: {format_amount(tx.fee)}<br>"
        html += f"Èas: {time.strftime('%d.%m.%Y %H:%M:%S UTC+00:00', time.gmtime(tx.timestamp))}<br>"
        html += f"Podpis: {tx.signature}<br>--------------------<br>"
        
    if not tx_found:
        html += "Žádné transakce nebyly nalezeny."
    if total_pages > 1:
        html += generate_pagination(page, total_pages, f"/search?query={query}")
    html += "</div>"
    return html + "</body></html>"

@app.route('/address/<address>')
def address_history(address):
    global droid_chain
    droid_chain = load_data()
    page = int(request.args.get('page', 1))
    
    html = "<html><head><title>Historie adresy</title><meta name='viewport' content='width=device-width, initial-scale=1.0'>" + CSS + "</head><body>"
    ITEMS_PER_PAGE = 10
    confirmed_balance = droid_chain.get_confirmed_balance(address)
    html += f"<div class='history'>--- Historie transakcí pro adresu '{address}' ---<br>"
    html += f'<span style="color: purple;">Potvrzený zùstatek</span>: {format_amount(confirmed_balance)}<br>'
    
    tx_history = []
    if os.path.exists(BLOCKCHAIN_DB):
        try:
            conn = sqlite3.connect(BLOCKCHAIN_DB)
            c = conn.cursor()
            c.execute("SELECT block_index, transactions FROM blocks ORDER BY block_index DESC")
            for row in c:
                transactions = json.loads(row[1])
                for tx_data in transactions:
                    tx = Transaction.from_dict(tx_data)
                    if tx.from_address == address or tx.to_address == address:
                        tx_history.append((row[0], tx))
            conn.close()
        except sqlite3.Error as e:
            html += f"Chyba pøi ètení DB: {e}<br>"

    total_txs = len(tx_history)
    total_pages = max(1, (total_txs + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
    start = (page - 1) * ITEMS_PER_PAGE
    end = start + ITEMS_PER_PAGE
    tx_found = False
    for block_index, tx in tx_history[start:end]:
        tx_found = True
        if tx.from_address == address:
            direction_html = '<span style="color: red;">Odesláno</span>'
        else:
            direction_html = '<span style="color: green;">Pøijato</span>'
        html += f"TX ID: <a href='/tx/{tx.tx_id}'>{tx.tx_id}</a><br>"
        html += f"Blok: <a href='/block/{block_index}'>#{block_index}</a><br>"
        html += f"Smìr: {direction_html}<br>"
        html += f"Od: {tx.from_address}<br>"
        html += f"Komu: {tx.to_address}<br>"
        html += f"Èástka: {format_amount(tx.amount)}<br>"
        if tx.from_address != "COINBASE":
            html += f"Poplatek: {format_amount(tx.fee)}<br>"
        html += f"Èas: {time.strftime('%d.%m.%Y %H:%M:%S UTC+00:00', time.gmtime(tx.timestamp))}<br>"
        html += f"Podpis: {tx.signature}<br>--------------------<br>"
    if not tx_found:
        html += "Žádné transakce nebyly nalezeny."
        
    if total_pages > 1:
        html += generate_pagination(page, total_pages, f"/address/{address}")
    html += "</div><a href='/'>Zpìt</a></body></html>"
    return html

@app.route('/tx/<tx_id>')
def tx_details(tx_id):
    global droid_chain
    droid_chain = load_data()
    tx, location = droid_chain.find_transaction_by_id(tx_id)
    html = "<html><head><title>Detaily TX</title><meta name='viewport' content='width=device-width, initial-scale=1.0'>" + CSS + "</head><body>"
    if tx:
        html += f"<div class='tx'>--- Detaily transakce (TX ID: {tx.tx_id}) ---<br>"
        html += f"Stav: Nalezena v {location}<br>"
        html += f"Od: <a href='/address/{tx.from_address}'>{tx.from_address}</a><br>"
        html += f"Komu: <a href='/address/{tx.to_address}'>{tx.to_address}</a><br>"
        html += f"Èástka: {format_amount(tx.amount)}<br>"
        html += f"Poplatek: {format_amount(tx.fee)}<br>"
        html += f"Nonce: {tx.nonce}<br>"
        html += f"Èas: {time.strftime('%d.%m.%Y %H:%M:%S UTC+00:00', time.gmtime(tx.timestamp))}<br>"
        html += f"Veøejný klíè: {tx.public_key}<br>"
        html += f"Podpis: {tx.signature}</div>"
    else:
        html += "Transakce nebyla nalezena."
    html += "<a href='/'>Zpìt</a></body></html>"
    return html

@app.route('/block/<int:index>')
def block_details(index):
    global droid_chain
    droid_chain = load_data()
    block = droid_chain.get_block_from_db(index)
    html = "<html><head><title>Detaily bloku</title><meta name='viewport' content='width=device-width, initial-scale=1.0'>" + CSS + "</head><body>"
    if block:
        html += f"<div class='block'>Blok #{block.index}<br>"
        html += f"Hash: {block.hash}<br>"
        html += f"Merkle Root: {block.merkle_root}<br>"
        html += f"Target (hex): {hex(block.target)[2:]}<br>"
        html += f"Pøedchozí hash: {block.previous_hash}<br>"
        html += f"PoW Nonce: {block.nonce}<br>"
        html += f"Èas: {time.strftime('%d.%m.%Y %H:%M:%S UTC+00:00', time.gmtime(block.timestamp))}<br>"
        html += f"Velikost bloku: {block.get_size() / 1024:.2f} KB<br>"
        html += f"Poèet potvrzení: {droid_chain.get_confirmations(block.hash)}<br>"
        html += f"Poèet transakcí: {len(block.transactions)}<br>"
        html += "Transakce:<br>"
        for tx in block.transactions:
            html += f"- TX ID: <a href='/tx/{tx.tx_id}'>{tx.tx_id}</a><br>"
            html += f"  Od: <a href='/address/{tx.from_address}'>{tx.from_address}</a><br>"
            html += f"  Komu: <a href='/address/{tx.to_address}'>{tx.to_address}</a><br>"
            html += f"  Èástka: {format_amount(tx.amount)}<br>"
            if tx.from_address != "COINBASE":
                html += f"  Poplatek: {format_amount(tx.fee)}<br>"
                html += f"  TX Nonce: {tx.nonce}<br>"
            html += f"  Podpis: {tx.signature}<br>"
        html += "</div>"
    else:
        html += "Blok nebyl nalezen."
    html += "<a href='/'>Zpìt</a></body></html>"
    return html

@app.route('/export')
def export_blockchain():
    export_data = f"--- Export Blockchainu {PROJECT_NAME} ({TICKER}) ---\n"
    export_data += f"Export vygenerován: {time.strftime('%d.%m.%Y %H:%M:%S UTC+00:00', time.gmtime())}\n\n"
    
    try:
        conn = sqlite3.connect(BLOCKCHAIN_DB)
        c = conn.cursor()
        c.execute("SELECT * FROM blocks ORDER BY block_index ASC")
        rows = c.fetchall()
        conn.close()
        
        if not rows:
            export_data += "Blockchain je prázdný."
            
        for row in rows:
            block_data = {
                'index': row[0],
                'timestamp': row[1],
                'transactions': json.loads(row[2]),
                'previous_hash': row[3],
                'target': row[4],     # Nový sloupec
                'nonce': row[5],
                'hash': row[6],
                'merkle_root': row[7] # Nový sloupec
            }
            block = Block.from_dict(block_data)
            
            export_data += f"================ BLOK #{block.index} ================\n"
            export_data += f"Hash: {block.hash}\n"
            export_data += f"Merkle Root: {block.merkle_root}\n"
            export_data += f"Target (hex): {hex(block.target)[2:]}\n"
            export_data += f"Pøedchozí hash: {block.previous_hash}\n"
            export_data += f"Èas: {time.strftime('%d.%m.%Y %H:%M:%S UTC+00:00', time.gmtime(block.timestamp))}\n"
            export_data += f"Nonce: {block.nonce}\n"
            export_data += f"Poèet transakcí: {len(block.transactions)}\n"
            export_data += "--- Transakce v bloku ---\n"
            
            for tx in block.transactions:
                export_data += f"  TX ID: {tx.tx_id}\n"
                export_data += f"  Od: {tx.from_address}\n"
                export_data += f"  Komu: {tx.to_address}\n"
                export_data += f"  Èástka: {format_amount(tx.amount)}\n"
                if tx.from_address != "COINBASE":
                    export_data += f"  Poplatek: {format_amount(tx.fee)}\n"
                    export_data += f"  Nonce: {tx.nonce}\n"
                export_data += f"  --------------------\n"
            
            export_data += "========================================\n\n"
            
    except sqlite3.Error as e:
        export_data = f"CHYBA PØI EXPORTU: {e}"
    
    return Response(
        export_data,
        mimetype="text/plain",
        headers={"Content-disposition": f"attachment; filename=droid_blockchain_export_{int(time.time())}.txt"}
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)