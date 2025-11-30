import hashlib
import json
import time
import ecdsa
import binascii
import os
import threading
import socket
import sys
import queue
from colorama import Fore, Style, init
import struct
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import getpass
import sqlite3
import heapq
import multiprocessing
from collections import defaultdict
import readline
import math
import signal

# Inicializace colorama
init(autoreset=True)

# Projektové konstanty
PROJECT_NAME = "Droid"
TICKER = "DRX"
DECIMALS = 8
MAX_SUPPLY = 100_000_000 * (10 ** DECIMALS)
BLOCK_REWARD = 100 * (10 ** DECIMALS)
HALVING_INTERVAL_BLOCKS = 495_000
BLOCK_TIME_SECONDS = 60
TX_FEE_MIN = int(0.00000001 * (10 ** DECIMALS))
TX_FEE_MAX = int(0.01 * (10 ** DECIMALS))
MIN_TX_AMOUNT = int(0.00000001 * (10 ** DECIMALS))  # Minimální èástka pro transakci
DIFFICULTY_ADJUSTMENT_INTERVAL = 10
TARGET_BLOCK_TIME = BLOCK_TIME_SECONDS * (DIFFICULTY_ADJUSTMENT_INTERVAL)
INITIAL_DIFFICULTY_BITS = 20
FIXED_TARGET = (1 << 256) >> INITIAL_DIFFICULTY_BITS
DYNAMIC_DIFFICULTY_ADJUSTMENT = True
BLOCKCHAIN_DB = 'blockchain.db'
WALLETS_FILE = 'wallets.json.enc'
MEMPOOL_DB = 'mempool.db'
PEERS_FILE = 'peers.json'
P2P_HOST = '0.0.0.0'
GENESIS_ADDRESS = "DRXfc3d428153b2c71be82e84a04c8b70b3d5153c75cc7c75edc3323d6e9c7cb8d5d063"
GENESIS_ADDRESS_EXPECTED_HASH = "be2a18b752ca5804e83a2e2b2fd182654284d95a1bb51893ce5bcdf8f199bd42"
GENESIS_TIMESTAMP = 1762819200
GENESIS_BLOCK_EXPECTED_HASH = "00000ef4c1724b44915d3b0c77df25a2bae4f9ea5a649a6a677e2739a381c955"
GENESIS_AMOUNT = 100 * (10 ** DECIMALS)
MAX_BLOCK_SIZE_BYTES = 1 * 1024 * 1024  # 1 MB
MAX_MEMPOOL_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
CONFIRMATIONS_THRESHOLD = 6
NTP_SERVERS = ['pool.ntp.org', 'time.nist.gov', 'time.google.com']
LAST_BLOCKS_TO_KEEP = 100  # Držet jen posledních 100 blokù v RAM
MAX_PEERS = 20  # Maximální poèet peerù pro prevenci Sybil
RATE_LIMIT_REQUESTS = 10  # Maximálnì 10 požadavkù za sekundu od jedné IP
RATE_LIMIT_WINDOW = 1  # Okno v sekundách pro rate limiting
MAX_MESSAGE_SIZE = 10 * 1024 * 1024  # 10 MB maximální velikost pøijímané zprávy pro ochranu proti DoS
TX_RATE_LIMIT = 100  # Maximálnì 100 transakcí za minutu od jednoho uzlu
TX_RATE_WINDOW = 60  # 1 minuta
MEMPOOL_TX_EXPIRATION = 24 * 3600  # 24 hodin pro expiraci transakcí v mempoolu
ALLOW_EMPTY_BLOCKS = True

# Globální offset pro synchronizaci èasu
time_offset = 0

# Globální flag pro read-only režim (pokud NTP selže)
is_read_only = False

def get_ntp_time(server):
    TIME1970 = 2208988800  # Reference time (epoch 1970-01-01)
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client.settimeout(5)  # Timeout pro NTP požadavek
        data = b'\x1b' + 47 * b'\0'
        client.sendto(data, (server, 123))
        
        data, _ = client.recvfrom(1024)
        if data:
            t = struct.unpack('!12I', data)
            secs = t[10] - TIME1970
            frac = t[11] / 2**32
            return secs + frac
    except Exception:
        return None
    finally:
        client.close()  # Zajistìte uzavøení socketu
    return None

def sync_time_with_ntp():
    global time_offset
    global is_read_only
 
    available_servers = []
    for server in NTP_SERVERS:
        try:
            ntp_time = get_ntp_time(server)
            if ntp_time is not None:
                available_servers.append(server)
                print(f"{Fore.GREEN}NTP server {server} is available.{Style.RESET_ALL}")
                # Použijeme první dostupný server pro výpoèet offsetu
                if time_offset == 0:
                    local_time = time.time()
                    time_offset = ntp_time - local_time
                    print(f"{Fore.GREEN}Time synchronized with {server}. Offset: {time_offset:.6f} seconds.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}NTP server {server} is unavailable: {e}{Style.RESET_ALL}")
    if not available_servers:
        print(f"{Fore.RED}No NTP servers available. Switching to read-only mode.{Style.RESET_ALL}")
        is_read_only = True

def get_time():
    return time.time() + time_offset

class Wallet:
    def __init__(self, private_key=None):
        if private_key:
            self.private_key = ecdsa.SigningKey.from_string(binascii.unhexlify(private_key), curve=ecdsa.SECP256k1, hashfunc= hashlib.sha3_256)
        else:
            self.private_key = self.generate_private_key()
        self.public_key = self.private_key.get_verifying_key()
        self.address = self.generate_address()

    def generate_private_key(self):
        return ecdsa.SigningKey.generate(curve=ecdsa.SECP256k1, hashfunc=hashlib.sha3_256)

    def generate_address(self):
        return self.public_key_to_address(self.public_key.to_string())

    @staticmethod
    def public_key_to_address(public_key_bytes):
        address_hash = hashlib.sha3_256(public_key_bytes).hexdigest()
        base_address = f"{TICKER}{address_hash}"
        checksum = hashlib.sha3_256(base_address.encode()).hexdigest()[:4]
        return base_address + checksum

    def sign_transaction(self, transaction):
        message = json.dumps(transaction.to_dict_for_signing(), sort_keys=True).encode()
        return binascii.hexlify(self.private_key.sign(message)).decode()

class Transaction:
    def __init__(self, from_address, to_address, amount, fee=0, nonce=0, public_key=None, signature=None, timestamp=None, tx_id=None):
        self.from_address = from_address
        self.to_address = to_address
        self.amount = amount
        self.fee = fee
        self.nonce = nonce
        self.timestamp = timestamp or get_time()
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

    def is_valid_timestamp(self):
        return self.timestamp <= get_time() + 600

    def verify_signature(self):
        if self.from_address == "COINBASE":
            return True
        if not self.public_key or not self.signature:
            return False
        try:
            vk = ecdsa.VerifyingKey.from_string(binascii.unhexlify(self.public_key), curve=ecdsa.SECP256k1, hashfunc=hashlib.sha3_256)
            message = json.dumps(self.to_dict_for_signing(), sort_keys=True).encode()
            return vk.verify(binascii.unhexlify(self.signature), message)
        except (ecdsa.BadSignatureError, binascii.Error):
            return False

    def verify_sender_identity(self):
        if self.from_address == "COINBASE":
            return True
        if not self.public_key:
            return False
        try:
            public_key_bytes = binascii.unhexlify(self.public_key)
            generated_address = Wallet.public_key_to_address(public_key_bytes)
            if len(self.from_address) == 67:
                base_generated = generated_address[:-4]
                return self.from_address == base_generated
            elif len(self.from_address) == 71:
                return self.from_address == generated_address
            else:
                return False
        except binascii.Error:
            return False

def compute_merkle_root(transactions):
    if not transactions:
        return hashlib.sha3_256(b'').hexdigest()  # Prázdný root

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

class Block:
    def __init__(self, index, transactions, previous_hash, target, nonce=0, timestamp=None):
        self.index = index
        self.timestamp = timestamp or get_time()
        self.transactions = transactions
        self.merkle_root = compute_merkle_root(transactions)
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
            'target': hex(self.target)[2:],
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
        target = int(data['target'], 16)
        block = Block(data['index'], transactions, data['previous_hash'], target, data['nonce'], timestamp=data['timestamp'])
        block.hash = data['hash']
        block.merkle_root = data.get('merkle_root', compute_merkle_root(transactions))  # Zpìtná kompatibilita
        return block

    def is_valid_timestamp(self, previous_block_timestamp):
        return self.timestamp > previous_block_timestamp and self.timestamp <= get_time() + 600

class Blockchain:
    def __init__(self, create_genesis=True):
        self.lock = threading.Lock()
        self.chain = []  # Nyní drží jen posledních LAST_BLOCKS_TO_KEEP blokù
        self.max_block_index = 0
        self.unconfirmed_transactions = []
        self.mining_in_progress = False
        self.all_tx_ids = set()  # Set všech unikátních TX ID v chainu
        self.balance_map = {}  # address -> confirmed_balance
        self.nonce_map = {}  # address -> max_nonce
        self.orphan_pool = {}  # hash -> block for orphans
        self.orphan_parents = defaultdict(list)  # previous_hash -> list of child hashes
        if create_genesis:
            self.create_genesis_block()
        self.cleanup_thread = threading.Thread(target=self.cleanup_mempool_periodically)
        self.cleanup_thread.daemon = True
        self.cleanup_thread.start()

    def cleanup_mempool_periodically(self):
        while True:
            time.sleep(60)  # Každou minutu
            now = get_time()
            with self.lock:
                self.unconfirmed_transactions = [tx for tx in self.unconfirmed_transactions if now - tx.timestamp < MEMPOOL_TX_EXPIRATION]
            save_mempool(self.unconfirmed_transactions)

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
        genesis_block = Block(0, [genesis_tx], "0", FIXED_TARGET, nonce=2156424, timestamp=GENESIS_TIMESTAMP)
        self.chain.append(genesis_block)
        self.max_block_index = 0
        self.all_tx_ids.add(genesis_tx.tx_id)
        self.update_state_with_block(genesis_block)
        print(f"{Fore.GREEN}Genesis blok vytvoøen a pøidán do øetìzce!{Style.RESET_ALL}")

    def update_state_with_block(self, block):
        for tx in block.transactions:
            self.all_tx_ids.add(tx.tx_id)
            if tx.from_address == "COINBASE":
                self.balance_map[tx.to_address] = self.balance_map.get(tx.to_address, 0) + tx.amount
            else:
                self.balance_map[tx.from_address] = self.balance_map.get(tx.from_address, 0) - tx.amount - tx.fee
                self.balance_map[tx.to_address] = self.balance_map.get(tx.to_address, 0) + tx.amount
                self.nonce_map[tx.from_address] = max(self.nonce_map.get(tx.from_address, -1), tx.nonce)

    def rebuild_state(self):
        conn = sqlite3.connect(BLOCKCHAIN_DB, timeout=1.0)
        c = conn.cursor()
        c.execute("SELECT * FROM blocks ORDER BY block_index")
        self.balance_map = {}
        self.nonce_map = {}
        self.all_tx_ids = set()
        for row in c:
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
        conn = sqlite3.connect(BLOCKCHAIN_DB, timeout=1.0)
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

    def get_block(self, index):
        for b in self.chain:
            if b.index == index:
                return b
        return self.get_block_from_db(index)

    def get_last_block(self):
        if self.chain:
            return self.chain[-1]
        # Pokud není v cache, naèíst z DB
        return self.get_block_from_db(self.max_block_index)

    def get_previous_block(self, block):
        if block.index == 0:
            return None
        prev_index = block.index - 1
        # Zkusit v cache
        for b in reversed(self.chain):
            if b.index == prev_index:
                return b
        # Naèíst z DB
        return self.get_block_from_db(prev_index)

    def get_next_nonce(self, address):
        confirmed_nonce = self.nonce_map.get(address, -1)
        pending_count = sum(1 for tx in self.unconfirmed_transactions if tx.from_address == address)
        return confirmed_nonce + 1 + pending_count

    def get_pending_balance(self, wallet_address):
        balance_change = 0
        for tx in self.unconfirmed_transactions:
            if tx.from_address == wallet_address:
                balance_change -= tx.amount + tx.fee
            if tx.to_address == wallet_address:
                balance_change += tx.amount
        return balance_change

    def get_balance(self, wallet_address):
        return self.get_confirmed_balance(wallet_address) + self.get_pending_balance(wallet_address)

    def get_confirmed_balance(self, wallet_address):
        return self.balance_map.get(wallet_address, 0)

    def get_total_supply(self):
        total_supply = 0
        # Poèítat z max_block_index, protože nemáme celý chain
        for i in range(self.max_block_index + 1):
            halvings = i // HALVING_INTERVAL_BLOCKS
            if i == 0:
                subsidy = GENESIS_AMOUNT
            else:
                subsidy = BLOCK_REWARD // (2 ** halvings)
            total_supply += subsidy
        return total_supply

    def get_cumulative_work(self, up_to_index=None):
        if up_to_index is None:
            up_to_index = self.max_block_index
        total_work = 0
        conn = sqlite3.connect(BLOCKCHAIN_DB, timeout=1.0)
        c = conn.cursor()
        c.execute("SELECT target_hex FROM blocks WHERE block_index <= ?", (up_to_index,))
        rows = c.fetchall()
        for row in rows:
            target = int(row[0], 16)
            work = (1 << 256) // target if target > 0 else 0
            total_work += work
        conn.close()
        return total_work

    def add_transaction(self, transaction):
        if not self.lock.acquire(timeout=5):
            print(f"{Fore.RED}System is busy (lock timeout). Try again later.{Style.RESET_ALL}")
            return False
        try:
            # Kontrola expirace pøi pøidávání
            if get_time() - transaction.timestamp >= MEMPOOL_TX_EXPIRATION:
                print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Transakce je pøíliš stará a expirovala.")
                return False

            mempool_size = sum(tx.get_size() for tx in self.unconfirmed_transactions)
            if mempool_size + transaction.get_size() > MAX_MEMPOOL_SIZE_BYTES:
                print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Mempool je plný, nová transakce byla odmítnuta.")
                return False

            if not transaction.is_valid_timestamp():
                print(f"{Fore.RED}Chyba ovìøení transakce:{Style.RESET_ALL} Timestamp transakce je neplatný.")
                return False

            # Kontrola TX ID v all_tx_ids (nyní potøebujeme zkontrolovat v DB, protože nemáme celý chain)
            if self.is_tx_id_in_chain(transaction.tx_id):
                print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Duplicitní TX ID: {transaction.tx_id}. Transakce již existuje v blockchainu.")
                return False

            # Kontrola duplicitního TX ID v mempoolu
            if any(tx.tx_id == transaction.tx_id for tx in self.unconfirmed_transactions):
                print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Duplicitní TX ID v mempoolu: {transaction.tx_id}.")
                return False

            # Kontrola duplicitní nonce v mempoolu pro stejnou adresu
            if transaction.from_address != "COINBASE":
                nonce_set = {tx.nonce for tx in self.unconfirmed_transactions if tx.from_address == transaction.from_address}
                if transaction.nonce in nonce_set:
                    print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Duplicitní nonce {transaction.nonce} pro adresu {transaction.from_address} v mempoolu.")
                    return False

            if transaction.from_address == "COINBASE" and transaction.amount > 0:
                self.unconfirmed_transactions.append(transaction)
                return True

            if transaction.amount <= 0:
                print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Èástka transakce musí být vìtší než 0.")
                return False

            if transaction.amount < MIN_TX_AMOUNT:
                print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Èástka transakce je pøíliš malá. Minimální èástka je {format(MIN_TX_AMOUNT / (10 ** DECIMALS), f'.{DECIMALS}f')} {TICKER}.")
                return False

            if not is_valid_address(transaction.from_address) or not is_valid_address(transaction.to_address):
                print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Neplatný formát adresy odesílatele nebo pøíjemce.")
                return False

            if not transaction.verify_sender_identity():
                print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Veøejný klíè neodpovídá adrese odesílatele.")
                return False

            if transaction.from_address == transaction.to_address:
                print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Nelze posílat peníze na stejnou adresu.")
                return False

            if not (TX_FEE_MIN <= transaction.fee <= TX_FEE_MAX):
                print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Poplatek za transakci je mimo povolený rozsah ({TX_FEE_MIN/(10**DECIMALS)}-{TX_FEE_MAX/(10**DECIMALS)} {TICKER}).")
                return False

            if transaction.nonce != self.get_next_nonce(transaction.from_address):
                print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Neplatná nonce ({transaction.nonce}). Oèekávána: {self.get_next_nonce(transaction.from_address)}.")
                return False

            if not transaction.verify_signature():
                print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Neplatný podpis transakce od {transaction.from_address}")
                return False

            current_available_balance = self.get_confirmed_balance(transaction.from_address)
            
            for tx_in_mempool in self.unconfirmed_transactions:
                if tx_in_mempool.from_address == transaction.from_address:
                    current_available_balance -= (tx_in_mempool.amount + tx_in_mempool.fee)
                    
            if current_available_balance < transaction.amount + transaction.fee:
                print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Nedostateèný zùstatek. K dispozici: {format(current_available_balance / (10**DECIMALS), f'.{DECIMALS}f')} {TICKER}")
                return False

            self.unconfirmed_transactions.append(transaction)
            return True
        finally:
            self.lock.release()

    def is_tx_id_in_chain(self, tx_id):
        # Zkusit v all_tx_ids (z cache)
        if tx_id in self.all_tx_ids:
            return True
        # Zkontrolovat v DB (pro starší bloky)
        conn = sqlite3.connect(BLOCKCHAIN_DB, timeout=1.0)
        c = conn.cursor()
        # Escapovat speciální znaky pro LIKE: % -> \%, _ -> \_, \ -> \\
        escaped_tx_id = tx_id.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
        c.execute("SELECT 1 FROM blocks WHERE transactions LIKE ? ESCAPE '\\'", (f'%{escaped_tx_id}%',))
        result = c.fetchone()
        conn.close()
        return result is not None

    def get_target(self):
        if not DYNAMIC_DIFFICULTY_ADJUSTMENT:
            return FIXED_TARGET
        last_block = self.get_last_block()
        new_block_index = last_block.index + 1
        if new_block_index < DIFFICULTY_ADJUSTMENT_INTERVAL:
            return FIXED_TARGET

        if new_block_index % DIFFICULTY_ADJUSTMENT_INTERVAL != 0:
            return last_block.target

        # Naèíst první blok v intervalu z DB
        first_index = new_block_index - DIFFICULTY_ADJUSTMENT_INTERVAL
        first_block = self.get_block(first_index)
        if not first_block:
            return FIXED_TARGET  # Chyba, ale fallback
        time_elapsed = last_block.timestamp - first_block.timestamp

        if time_elapsed <= 0:
            time_elapsed = 1

        adjustment_factor = time_elapsed / TARGET_BLOCK_TIME
        old_target = last_block.target
        new_target = int(old_target * adjustment_factor)
        new_target = max(1, new_target)
        new_target = min(new_target, (1 << 256) - 1)

        p2p_node.add_log(f"{Fore.MAGENTA}ÚPRAVA TARGETU na bloku #{new_block_index}:{Style.RESET_ALL}")
        p2p_node.add_log(f"  Èas posledních {DIFFICULTY_ADJUSTMENT_INTERVAL} blokù: {time_elapsed:.2f}s (Cíl: {TARGET_BLOCK_TIME}s)")
        p2p_node.add_log(f"  Faktor úpravy: {adjustment_factor:.4f}")
        p2p_node.add_log(f"  Starý target (hex): {hex(old_target)[2:]} -> Nový target (hex): {hex(new_target)[2:]}")

        return new_target

    def calculate_expected_target(self, new_block_index, chain=None):
        if not DYNAMIC_DIFFICULTY_ADJUSTMENT:
            return FIXED_TARGET
        if new_block_index < DIFFICULTY_ADJUSTMENT_INTERVAL:
            return FIXED_TARGET

        if new_block_index % DIFFICULTY_ADJUSTMENT_INTERVAL != 0:
            if chain is None:
                last_block = self.get_block(new_block_index - 1)
            else:
                last_block = chain[new_block_index - 1]
            return last_block.target

        first_index = new_block_index - DIFFICULTY_ADJUSTMENT_INTERVAL
        if chain is None:
            first_block = self.get_block(first_index)
            last_block = self.get_block(new_block_index - 1)
        else:
            first_block = chain[first_index]
            last_block = chain[new_block_index - 1]
        if not first_block or not last_block:
            return FIXED_TARGET
        time_elapsed = last_block.timestamp - first_block.timestamp

        if time_elapsed <= 0:
            time_elapsed = 1

        adjustment_factor = time_elapsed / TARGET_BLOCK_TIME
        old_target = last_block.target
        new_target = int(old_target * adjustment_factor)
        new_target = max(1, new_target)
        new_target = min(new_target, (1 << 256) - 1)

        return new_target

    def mining_worker(self, block_data, start_nonce, step, result_queue, stop_event, update_interval=1.0, worker_id=0):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        index = block_data['index']
        timestamp = block_data['timestamp']
        merkle_root = block_data['merkle_root']  # Použij root místo transakcí
        previous_hash = block_data['previous_hash']
        target_hex = block_data['target']
        target = int(target_hex, 16)
        nonce = start_nonce
        hashes_calculated = 0
        last_update_time = time.time()

        while not stop_event.is_set():
            current_time = time.time()
            if current_time - timestamp > 60:
                timestamp = current_time
                nonce = start_nonce  # Reset nonce po zmìnì timestampu

            block_dict = {
                'index': index,
                'timestamp': timestamp,
                'merkle_root': merkle_root,
                'previous_hash': previous_hash,
                'target': hex(target)[2:],
                'nonce': nonce
            }
            block_string = json.dumps(block_dict, sort_keys=True)
            computed_hash = hashlib.sha3_256(block_string.encode()).hexdigest()
            hashes_calculated += 1

            if self.meets_difficulty(computed_hash, target):
                result_queue.put(('result', nonce, timestamp, computed_hash))
                return

            if current_time - last_update_time >= update_interval:
                result_queue.put(('update', worker_id, hashes_calculated, nonce, computed_hash))
                hashes_calculated = 0  # Reset po odeslání
                last_update_time = current_time

            nonce += step

        # Poslat finální update pøed ukonèením
        if hashes_calculated > 0:
            result_queue.put(('update', worker_id, hashes_calculated, nonce, computed_hash))

    def proof_of_work(self, block, num_cores):
        start_time = get_time()
        computed_hash = ""
        total_hashes_calculated = 0

        self.mining_in_progress = True
        sys.stdout.write(f"\n{Fore.YELLOW}Zaèínám tìžit blok...{Style.RESET_ALL}\n")

        # Pøipravit data bloku pro pøedání do procesù (serializovatelná)
        block_data = {
            'index': block.index,
            'timestamp': block.timestamp,
            'merkle_root': block.merkle_root,  # Použij root místo transakcí
            'previous_hash': block.previous_hash,
            'target': hex(block.target)[2:]
        }

        result_queue = multiprocessing.Queue()
        stop_event = multiprocessing.Event()
        processes = []

        core_hashes = [0] * num_cores
        core_nonces = [0] * num_cores
        core_last_hashes = [""] * num_cores

        for i in range(num_cores):
            p = multiprocessing.Process(target=self.mining_worker, args=(block_data, i, num_cores, result_queue, stop_event, 1.0, i))
            processes.append(p)
            p.start()

        last_hashrate_time = get_time()

        try:
            while self.mining_in_progress:
                try:
                    msg = result_queue.get(timeout=1)
                    if msg[0] == 'result':
                        nonce, new_timestamp, computed_hash = msg[1], msg[2], msg[3]
                        block.nonce = nonce
                        block.timestamp = new_timestamp
                        block.hash = computed_hash
                        stop_event.set()
                        break
                    elif msg[0] == 'update':
                        worker_id = msg[1]
                        core_hashes[worker_id] += msg[2]
                        core_nonces[worker_id] = msg[3]
                        core_last_hashes[worker_id] = msg[4][:10]
                except queue.Empty:
                    pass

                current_time = get_time()
                if current_time - last_hashrate_time >= 5:
                    elapsed_time = current_time - start_time
                    if elapsed_time > 0:
                        total_hashrate = sum(core_hashes) / elapsed_time
                        sys.stdout.write(f"\r{Fore.CYAN}Celkový Hashrate:{Style.RESET_ALL} {total_hashrate/1000:.2f} Kh/s {Fore.CYAN}Èas:{Style.RESET_ALL} {elapsed_time:.1f}s\n")
                        for i in range(num_cores):
                            core_hashrate = core_hashes[i] / elapsed_time if elapsed_time > 0 else 0
                            sys.stdout.write(f"Core {i+1}: {core_hashrate/1000:.2f} Kh/s | Nonce: {core_nonces[i]} | Hash: {core_last_hashes[i]}\n")
                            sys.stdout.flush()
                    last_hashrate_time = current_time
        except KeyboardInterrupt:
            sys.stdout.write(f"\r{Fore.YELLOW}Mining interrupted by user.{Style.RESET_ALL}\n")
            sys.stdout.flush()
            stop_event.set()
            for p in processes:
                p.join()
            self.mining_in_progress = False
            return None

        stop_event.set()
        for p in processes:
            p.join()

        self.mining_in_progress = False
        if computed_hash:
            elapsed_time = get_time() - start_time
            hashrate = sum(core_hashes) / elapsed_time if elapsed_time > 0 else 0
            sys.stdout.write(f"\r{Fore.GREEN}Blok nalezen!{Style.RESET_ALL} {Fore.CYAN}Celkový Hashrate:{Style.RESET_ALL} {hashrate/1000:.2f} Kh/s | {Fore.CYAN}Nonce:{Style.RESET_ALL} {block.nonce} | {Fore.CYAN}Hash:{Style.RESET_ALL} {computed_hash[:10]}... | {Fore.CYAN}Èas:{Style.RESET_ALL} {elapsed_time:.2f}s\n")
            sys.stdout.flush()
            return computed_hash
        else:
            sys.stdout.write(f"\r{Fore.YELLOW}Tìžba byla zastavena, pøijat nový blok od uzlu.{Style.RESET_ALL}\n")
            sys.stdout.flush()
            return None

    def meets_difficulty(self, hash_hex, target):
        hash_int = int(hash_hex, 16)
        return hash_int < target

    def add_block(self, block, proof):
        if not self.lock.acquire(timeout=5):
            print(f"{Fore.RED}System is busy (lock timeout). Try again later.{Style.RESET_ALL}")
            return False
        try:
            previous_block = self.get_last_block()
            previous_hash = previous_block.hash

            if previous_hash != block.previous_hash:
                return False

            if block.index != previous_block.index + 1:
                return False
                
            if not block.is_valid_timestamp(previous_block.timestamp):
                p2p_node.add_log(f"{Fore.RED}Chyba ovìøení bloku:{Style.RESET_ALL} Timestamp bloku je neplatný.")
                return False

            if block.target != self.get_target():
                return False

            if block.target != self.calculate_expected_target(block.index):
                p2p_node.add_log(f"{Fore.RED}Chyba ovìøení bloku: Nesprávný target bloku.{Style.RESET_ALL}")
                return False

            if block.merkle_root != compute_merkle_root(block.transactions):
                p2p_node.add_log(f"{Fore.RED}Chyba ovìøení bloku: Nesprávný Merkle root.{Style.RESET_ALL}")
                return False

            if not self.meets_difficulty(proof, block.target):
                return False

            # Nová kontrola: Žádný prázdný blok (musí obsahovat alespoò jednu uživatelskou transakci)
            if not ALLOW_EMPTY_BLOCKS and len(block.transactions) <= 1:
                p2p_node.add_log(f"{Fore.RED}Chyba ovìøení bloku: Blok je prázdný (obsahuje pouze coinbase transakci).{Style.RESET_ALL}")
                return False

            # Nová kontrola: Žádná TX v bloku nesmí mít duplicitní TX ID s existujícími v chainu
            for tx in block.transactions:
                if self.is_tx_id_in_chain(tx.tx_id):
                    p2p_node.add_log(f"{Fore.RED}Chyba ovìøení bloku:{Style.RESET_ALL} Duplicitní TX ID {tx.tx_id} v bloku.")
                    return False

            # Kontrola duplicitních nonce v bloku pro stejného odesílatele
            nonce_map = {}
            tx_id_set = set()
            for tx in block.transactions:
                if tx.tx_id in tx_id_set:
                    p2p_node.add_log(f"{Fore.RED}Chyba ovìøení bloku:{Style.RESET_ALL} Duplicitní TX ID {tx.tx_id} v bloku.")
                    return False
                tx_id_set.add(tx.tx_id)
                if tx.from_address != "COINBASE":
                    if tx.from_address in nonce_map:
                        if tx.nonce in nonce_map[tx.from_address]:
                            p2p_node.add_log(f"{Fore.RED}Chyba ovìøení bloku:{Style.RESET_ALL} Duplicitní nonce {tx.nonce} pro adresu {tx.from_address} v bloku.")
                            return False
                        nonce_map[tx.from_address].add(tx.nonce)
                    else:
                        nonce_map[tx.from_address] = {tx.nonce}

            # Globální kontrola nonce pro transakce v bloku
            for tx in block.transactions:
                if tx.from_address != "COINBASE":
                    max_nonce = self.nonce_map.get(tx.from_address, -1)
                    if tx.nonce <= max_nonce:
                        p2p_node.add_log(f"{Fore.RED}Chyba ovìøení bloku: Neplatná nebo duplicitní nonce {tx.nonce} pro adresu {tx.from_address} (oèekáváno vyšší než {max_nonce}).{Style.RESET_ALL}")
                        return False

            for tx in block.transactions:
                if not tx.verify_sender_identity() and tx.from_address != "COINBASE":
                    p2p_node.add_log(f"{Fore.RED}Chyba ovìøení bloku:{Style.RESET_ALL} Veøejný klíè neodpovídá adrese odesílatele.")
                    return False
                if not tx.verify_signature() and tx.from_address != "COINBASE":
                    p2p_node.add_log(f"{Fore.RED}Chyba ovìøení bloku:{Style.RESET_ALL} Neplatný podpis transakce.")
                    return False

                if tx.from_address != "COINBASE":
                    if tx.amount <= 0:
                        p2p_node.add_log(f"{Fore.RED}Chyba ovìøení bloku: Èástka transakce musí být vìtší než 0.{Style.RESET_ALL}")
                        return False
                    if tx.amount < MIN_TX_AMOUNT:
                        p2p_node.add_log(f"{Fore.RED}Chyba ovìøení bloku: Èástka transakce je pøíliš malá.{Style.RESET_ALL}")
                        return False
                    if not (TX_FEE_MIN <= tx.fee <= TX_FEE_MAX):
                        p2p_node.add_log(f"{Fore.RED}Chyba ovìøení bloku: Poplatek transakce mimo rozsah.{Style.RESET_ALL}")
                        return False

            # Kontrola coinbase: jen jedna, první, správná odmìna
            coinbase_txs = [tx for tx in block.transactions if tx.from_address == "COINBASE"]
            if len(coinbase_txs) != 1:
                p2p_node.add_log(f"{Fore.RED}Chyba ovìøení bloku: Nesprávný poèet coinbase transakcí (oèekávána 1).{Style.RESET_ALL}")
                return False
            if block.transactions[0].from_address != "COINBASE":
                p2p_node.add_log(f"{Fore.RED}Chyba ovìøení bloku: Coinbase transakce musí být první v bloku.{Style.RESET_ALL}")
                return False
            coinbase_tx = coinbase_txs[0]
            halvings = block.index // HALVING_INTERVAL_BLOCKS
            expected_reward = BLOCK_REWARD // (2 ** halvings) if block.index > 0 else GENESIS_AMOUNT
            total_fees = sum(tx.fee for tx in block.transactions if tx.from_address != "COINBASE")
            if coinbase_tx.amount != expected_reward + total_fees:
                p2p_node.add_log(f"{Fore.RED}Chyba ovìøení bloku: Nesprávná coinbase odmìna.{Style.RESET_ALL}")
                return False

            # Kontrola celkové nabídky
            if self.get_total_supply() + coinbase_tx.amount > MAX_SUPPLY:
                p2p_node.add_log(f"{Fore.RED}Chyba ovìøení bloku: Pøekroèení maximální nabídky mincí.{Style.RESET_ALL}")
                return False

            # Nová kontrola: Validace zùstatkù s kumulativní útratou v bloku
            temp_balance_changes = defaultdict(int)
            for tx in block.transactions:
                if tx.from_address != "COINBASE":
                    current_balance = self.balance_map.get(tx.from_address, 0) + temp_balance_changes[tx.from_address]
                    if current_balance < tx.amount + tx.fee:
                        p2p_node.add_log(f"{Fore.RED}Chyba ovìøení bloku: Nedostateèný zùstatek pro transakci {tx.tx_id} od {tx.from_address}.{Style.RESET_ALL}")
                        return False
                    temp_balance_changes[tx.from_address] -= (tx.amount + tx.fee)
                    temp_balance_changes[tx.to_address] += tx.amount
                else:
                    # Pøidat coinbase do temp_changes (pro pøípad, že by to ovlivnilo následující tx, ale normálnì ne, protože coinbase je první)
                    temp_balance_changes[tx.to_address] += tx.amount

            block.hash = proof
            self.chain.append(block)
            self.max_block_index = block.index
            # Omezit chain na posledních LAST_BLOCKS_TO_KEEP blokù
            if len(self.chain) > LAST_BLOCKS_TO_KEEP:
                self.chain = self.chain[-LAST_BLOCKS_TO_KEEP:]
            self.update_state_with_block(block)
            self.resolve_orphans(block.hash)
            return True
        finally:
            self.lock.release()

    def add_orphan_block(self, block):
        if block.hash in self.orphan_pool:
            return
        self.orphan_pool[block.hash] = block
        self.orphan_parents[block.previous_hash].append(block.hash)

    def resolve_orphans(self, parent_hash):
        if parent_hash not in self.orphan_parents:
            return
        for child_hash in list(self.orphan_parents[parent_hash]):
            if child_hash in self.orphan_pool:
                child_block = self.orphan_pool.pop(child_hash)
                self.orphan_parents[parent_hash].remove(child_hash)
                # Zkusit pøidat
                if child_block.previous_hash == parent_hash:
                    if self.add_block(child_block, child_block.hash):
                        p2p_node.add_log(f"{Fore.GREEN}Orphan blok {child_block.index} pøidán do chainu.{Style.RESET_ALL}")
                        self.resolve_orphans(child_block.hash)  # Rekurze pro dìti
                    else:
                        # Pokud nevalidní, zahodit a recyklovat tx
                        self.recycle_orphan_transactions([child_block])
                        p2p_node.add_log(f"{Fore.RED}Orphan blok {child_block.index} nevalidní, zahazuji a recykluji tx.{Style.RESET_ALL}")

    def recycle_orphan_transactions(self, orphaned_blocks):
        orphaned_transactions = []
        for block in orphaned_blocks:
            if self.get_confirmations(block.hash) >= CONFIRMATIONS_THRESHOLD:
                continue  # Nepøepisovat staré
            for tx in block.transactions:
                if tx.from_address != "COINBASE" and not self.is_tx_id_in_chain(tx.tx_id):
                    orphaned_transactions.append(tx)
        for tx in orphaned_transactions:
            if self.add_transaction(tx):
                p2p_node.add_log(f"{Fore.GREEN}Osiøelá transakce {tx.tx_id} pøidána zpìt do mempoolu.{Style.RESET_ALL}")
            else:
                p2p_node.add_log(f"{Fore.RED}Osiøelá transakce {tx.tx_id} nemohla být pøidána do mempoolu.{Style.RESET_ALL}")

    def mine(self, miner_address):
        # Vypoèítat odmìnu a poplatky
        halvings = self.max_block_index // HALVING_INTERVAL_BLOCKS
        current_reward = BLOCK_REWARD // (2 ** halvings)
        
        if self.get_total_supply() + current_reward > MAX_SUPPLY:
            print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Maximální nabídka dosažena, nelze vytìžit další blok.")
            return False

        # Zde pøidat volbu poètu jader
        num_cores = multiprocessing.cpu_count()
        print(f"Detekováno {num_cores} CPU jader.")
        try:
            user_cores = int(input(f"Vyberte poèet dostupných CPU jader: "))
            if 1 <= user_cores <= num_cores:
                num_cores = user_cores
            else:
                print(f"{Fore.RED}Neplatný poèet. Používám všechna {num_cores} jádra.{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}Neplatný vstup. Používám všechna {num_cores} jádra.{Style.RESET_ALL}")

        # Kontrola 2: Zda bìhem konfigurace (volba CPU) nepøišel nový blok
        if self.get_last_block().hash != self.get_last_block().hash:
            print(f"{Fore.YELLOW}Tìžba zrušena: Mezitím pøišel nový blok od jiného uzlu.{Style.RESET_ALL}")
            return False

        while True:
            # Vytìøit coinbase transakci jako první
            mining_reward = Transaction("COINBASE", miner_address, 0, nonce=0, public_key="COINBASE", signature="COINBASE")
            new_block_transactions = [mining_reward] # Coinbase je nyní první

            current_block_size = 0
            dummy_block = Block(self.max_block_index + 1, [], self.get_last_block().hash, self.get_target())
            current_block_size += dummy_block.get_size()
            current_block_size += mining_reward.get_size() # Pøipoèítat velikost coinbase transakce

            total_fees = 0

            # Grupování transakcí podle from_address
            tx_by_sender = {}
            for tx in self.unconfirmed_transactions:
                if tx.from_address not in tx_by_sender:
                    tx_by_sender[tx.from_address] = []
                tx_by_sender[tx.from_address].append(tx)

            # Pro každou adresu seøadit podle nonce vzestupnì
            for sender in tx_by_sender:
                tx_by_sender[sender].sort(key=lambda tx: tx.nonce)

            # Priority queue (max-heap) pro výbìr transakcí: (-fee, sender) pro max fee
            pq = []
            for sender, txs in tx_by_sender.items():
                if txs:
                    first_tx = txs[0]
                    heapq.heappush(pq, (-first_tx.fee, sender))

            selected_txs = []
            while pq:
                _, sender = heapq.heappop(pq)
                if sender not in tx_by_sender or not tx_by_sender[sender]:
                    continue
                tx = tx_by_sender[sender].pop(0)
                tx_size = tx.get_size()
                if current_block_size + tx_size > MAX_BLOCK_SIZE_BYTES:
                    # Vrátit zpìt, pokud se nevejde
                    tx_by_sender[sender].insert(0, tx)
                    continue
                selected_txs.append(tx)
                current_block_size += tx_size
                total_fees += tx.fee
                # Pokud jsou další tx pro sender, pøidat zpìt do pq s fee další tx
                if tx_by_sender[sender]:
                    next_tx = tx_by_sender[sender][0]
                    heapq.heappush(pq, (-next_tx.fee, sender))

            # Kontrola duplicit v selected_txs: duplicitní TX ID a duplicitní nonce pro stejnou adresu
            tx_id_set = set()
            nonce_map = {}
            for tx in selected_txs:
                if tx.tx_id in tx_id_set:
                    print(f"{Fore.RED}Chyba v mineru:{Style.RESET_ALL} Duplicitní TX ID {tx.tx_id} v vybraných transakcích. Tìžba nebyla spuštìna.")
                    return False
                tx_id_set.add(tx.tx_id)
                if tx.from_address != "COINBASE":
                    if tx.from_address in nonce_map:
                        if tx.nonce in nonce_map[tx.from_address]:
                            print(f"{Fore.RED}Chyba v mineru:{Style.RESET_ALL} Duplicitní nonce {tx.nonce} pro adresu {tx.from_address} v vybraných transakcích. Tìžba nebyla spuštìna.")
                            return False
                        nonce_map[tx.from_address].add(tx.nonce)
                    else:
                        nonce_map[tx.from_address] = {tx.nonce}

            new_block_transactions += selected_txs

            if not ALLOW_EMPTY_BLOCKS and len(selected_txs) == 0:
                print(f"{Fore.YELLOW}Upozornìní:{Style.RESET_ALL} V mempoolu nejsou žádné transakce k vytìžení. Tìžba byla zastavena.")
                return False

            final_reward = current_reward + total_fees
            if final_reward == 0:
                print(f"{Fore.YELLOW}Upozornìní:{Style.RESET_ALL} Maximální nabídka byla dosažena a nejsou k dispozici žádné transakce k vytìžení.")
                return False
            
            # Aktualizovat èástku v coinbase transakci, která je již v seznamu
            new_block_transactions[0].amount = final_reward

            last_block = self.get_last_block()
            target = self.get_target()
            new_block = Block(
                index=last_block.index + 1,
                transactions=new_block_transactions,
                previous_hash=last_block.hash,
                target=target
            )

            # Kontrola 1: Zda bìhem pøípravy transakcí nepøišel nový blok
            if self.get_last_block().hash != last_block.hash:
                print(f"{Fore.YELLOW}Tìžba zrušena: Mezitím pøišel nový blok od jiného uzlu.{Style.RESET_ALL}")
                return False

            proof = self.proof_of_work(new_block, num_cores)
            if proof is None:
                return False

            if self.add_block(new_block, proof):
                print(f"{Fore.GREEN}Blok {new_block.index} byl vytìžen a pøidán do øetìzce!{Style.RESET_ALL} (Target (hex): {hex(target)[2:]})")
                print(f"  Datum: {time.strftime('%d.%m.%Y %H:%M:%S UTC+00:00', time.gmtime(new_block.timestamp))}")
                print(f"  Velikost bloku: {Fore.CYAN}{new_block.get_size() / 1024:.2f} KB{Style.RESET_ALL}")
                print(f"  Odmìna za blok: {Fore.CYAN}{format(current_reward / (10 ** DECIMALS), f'.{DECIMALS}f')} {TICKER}{Style.RESET_ALL}")
                print(f"  Poplatky z transakcí: {Fore.CYAN}{format(total_fees / (10 ** DECIMALS), f'.{DECIMALS}f')} {TICKER}{Style.RESET_ALL}")
                print(f"  Celková odmìna pro tìžaøe: {Fore.CYAN}{format(final_reward / (10 ** DECIMALS), f'.{DECIMALS}f')} {TICKER}{Style.RESET_ALL}")
                
                confirmed_tx_ids = {tx.tx_id for tx in new_block_transactions if tx.from_address != "COINBASE"}
                self.unconfirmed_transactions = [
                    tx for tx in self.unconfirmed_transactions
                    if tx.tx_id not in confirmed_tx_ids
                ]
                save_mempool(self.unconfirmed_transactions)
                p2p_node.send_data_to_peers({'type': 'new_block', 'data': new_block.to_dict()})
                save_data(self, wallets, password, p2p_node.peers)

            else:
                return False

    def is_valid_chain(self, chain=None):
        # Pokud chain není poskytnut, validovat celý chain z DB
        if chain is None:
            conn = sqlite3.connect(BLOCKCHAIN_DB, timeout=1.0)
            c = conn.cursor()
            c.execute("SELECT * FROM blocks ORDER BY block_index")
            rows = c.fetchall()
            conn.close()
            chain = [Block.from_dict({
                'index': row[0],
                'timestamp': row[1],
                'transactions': json.loads(row[2]),
                'previous_hash': row[3],
                'target': row[4],
                'nonce': row[5],
                'hash': row[6],
                'merkle_root': row[7]
            }) for row in rows]

        seen_tx_ids = set()
        nonce_maps = {}
        total_supply = 0

        # Kontrola genesis bloku
        genesis_block = chain[0]
        if genesis_block.index != 0 or genesis_block.previous_hash != "0" or genesis_block.target != FIXED_TARGET:
            p2p_node.add_log(f"{Fore.RED}Chyba ovìøení øetìzce: Nesprávné vlastnosti genesis bloku.{Style.RESET_ALL}")
            return False
        if len(genesis_block.transactions) != 1 or genesis_block.transactions[0].from_address != "COINBASE":
            p2p_node.add_log(f"{Fore.RED}Chyba ovìøení øetìzce: Nesprávná coinbase v genesis bloku.{Style.RESET_ALL}")
            return False
        genesis_tx = genesis_block.transactions[0]
        if genesis_tx.to_address != GENESIS_ADDRESS or genesis_tx.amount != GENESIS_AMOUNT or genesis_tx.timestamp != GENESIS_TIMESTAMP:
            p2p_node.add_log(f"{Fore.RED}Chyba ovìøení øetìzce: Nesprávné konstanty v genesis transakci.{Style.RESET_ALL}")
            return False
        if genesis_block.merkle_root != compute_merkle_root(genesis_block.transactions):
            p2p_node.add_log(f"{Fore.RED}Chyba ovìøení øetìzce: Nesprávný Merkle root v genesis bloku.{Style.RESET_ALL}")
            return False
        total_supply += GENESIS_AMOUNT

        for i in range(1, len(chain)):
            current_block = chain[i]
            previous_block = chain[i-1]

            if not current_block.is_valid_timestamp(previous_block.timestamp):
                p2p_node.add_log(f"{Fore.RED}Chyba ovìøení bloku: Timestamp bloku #{current_block.index} v øetìzci je neplatný.{Style.RESET_ALL}")
                return False

            if current_block.hash != current_block.compute_hash():
                return False

            if current_block.previous_hash != previous_block.hash:
                return False

            if current_block.target != self.calculate_expected_target(current_block.index, chain=chain):
                p2p_node.add_log(f"{Fore.RED}Chyba ovìøení øetìzce: Nesprávný target bloku #{current_block.index}.{Style.RESET_ALL}")
                return False

            if current_block.merkle_root != compute_merkle_root(current_block.transactions):
                p2p_node.add_log(f"{Fore.RED}Chyba ovìøení øetìzce: Nesprávný Merkle root v bloku #{current_block.index}.{Style.RESET_ALL}")
                return False

            if not self.meets_difficulty(current_block.hash, current_block.target):
                return False

            # Kontrola coinbase: jen jedna, první, správná odmìna
            coinbase_txs = [tx for tx in current_block.transactions if tx.from_address == "COINBASE"]
            if len(coinbase_txs) != 1:
                p2p_node.add_log(f"{Fore.RED}Chyba ovìøení øetìzce: Nesprávný poèet coinbase transakcí v bloku #{current_block.index} (oèekávána 1).{Style.RESET_ALL}")
                return False
            if current_block.transactions[0].from_address != "COINBASE":
                p2p_node.add_log(f"{Fore.RED}Chyba ovìøení øetìzce: Coinbase transakce musí být první v bloku #{current_block.index}.{Style.RESET_ALL}")
                return False
            coinbase_tx = coinbase_txs[0]
            halvings = current_block.index // HALVING_INTERVAL_BLOCKS
            expected_reward = BLOCK_REWARD // (2 ** halvings)
            total_fees = sum(tx.fee for tx in current_block.transactions if tx.from_address != "COINBASE")
            if coinbase_tx.amount != expected_reward + total_fees:
                p2p_node.add_log(f"{Fore.RED}Chyba ovìøení øetìzce: Nesprávná coinbase odmìna v bloku #{current_block.index}.{Style.RESET_ALL}")
                return False
            total_supply += coinbase_tx.amount
            if total_supply > MAX_SUPPLY:
                p2p_node.add_log(f"{Fore.RED}Chyba ovìøení øetìzce: Pøekroèení maximální nabídky mincí po bloku #{current_block.index}.{Style.RESET_ALL}")
                return False

            # Kontrola duplicitních TX ID v bloku a v celém øetìzci
            block_tx_ids = set()
            block_nonce_map = {}
            for tx in current_block.transactions:
                if tx.tx_id in seen_tx_ids:
                    p2p_node.add_log(f"{Fore.RED}Chyba ovìøení øetìzce: Duplicitní TX ID {tx.tx_id} v øetìzci.{Style.RESET_ALL}")
                    return False
                if tx.tx_id in block_tx_ids:
                    p2p_node.add_log(f"{Fore.RED}Chyba ovìøení bloku: Duplicitní TX ID {tx.tx_id} v bloku #{current_block.index}.{Style.RESET_ALL}")
                    return False
                block_tx_ids.add(tx.tx_id)
                seen_tx_ids.add(tx.tx_id)

                if tx.from_address != "COINBASE":
                    if tx.amount <= 0:
                        p2p_node.add_log(f"{Fore.RED}Chyba ovìøení øetìzce: Èástka transakce musí být vìtší než 0.{Style.RESET_ALL}")
                        return False
                    if tx.amount < MIN_TX_AMOUNT:
                        p2p_node.add_log(f"{Fore.RED}Chyba ovìøení øetìzce: Èástka transakce je pøíliš malá.{Style.RESET_ALL}")
                        return False
                    if not (TX_FEE_MIN <= tx.fee <= TX_FEE_MAX):
                        p2p_node.add_log(f"{Fore.RED}Chyba ovìøení øetìzce: Poplatek transakce mimo rozsah.{Style.RESET_ALL}")
                        return False

                    # Kontrola nonce v bloku
                    if tx.from_address in block_nonce_map:
                        if tx.nonce in block_nonce_map[tx.from_address]:
                            p2p_node.add_log(f"{Fore.RED}Chyba ovìøení bloku: Duplicitní nonce {tx.nonce} pro adresu {tx.from_address} v bloku #{current_block.index}.{Style.RESET_ALL}")
                            return False
                        block_nonce_map[tx.from_address].add(tx.nonce)
                    else:
                        block_nonce_map[tx.from_address] = {tx.nonce}

                    # Globální kontrola nonce sekvence v øetìzci (musí být vzestupné a bez duplicit)
                    if tx.from_address in nonce_maps:
                        if tx.nonce <= max(nonce_maps[tx.from_address]):
                            p2p_node.add_log(f"{Fore.RED}Chyba ovìøení øetìzce: Neplatná nebo duplicitní nonce {tx.nonce} pro adresu {tx.from_address} (oèekáváno vyšší než {max(nonce_maps[tx.from_address])}).{Style.RESET_ALL}")
                            return False
                        nonce_maps[tx.from_address].add(tx.nonce)
                    else:
                        nonce_maps[tx.from_address] = {tx.nonce}

                if not tx.verify_sender_identity() and tx.from_address != "COINBASE":
                    return False
                if not tx.verify_signature() and tx.from_address != "COINBASE":
                    return False

            # Kontrola prázdného bloku v øetìzci
            if not ALLOW_EMPTY_BLOCKS and len(current_block.transactions) <= 1:
                p2p_node.add_log(f"{Fore.RED}Chyba ovìøení øetìzce: Blok #{current_block.index} je prázdný (obsahuje pouze coinbase transakci).{Style.RESET_ALL}")
                return False

        return True

    def get_confirmations(self, block_hash):
        # Najít index bloku v DB
        conn = sqlite3.connect(BLOCKCHAIN_DB, timeout=1.0)
        c = conn.cursor()
        c.execute("SELECT block_index FROM blocks WHERE block_hash = ?", (block_hash,))
        row = c.fetchone()
        conn.close()
        if row:
            block_index = row[0]
            return self.max_block_index - block_index
        return 0

    def replace_chain(self, new_chain_data):
        if not self.lock.acquire(timeout=5):
            print(f"{Fore.RED}System is busy (lock timeout). Try again later.{Style.RESET_ALL}")
            return False
        try:
            new_chain = [Block.from_dict(block_data) for block_data in new_chain_data]
            current_cum_work = self.get_cumulative_work()
            new_cum_work = sum(((1 << 256) // b.target if b.target > 0 else 0) for b in new_chain)
            current_length = self.max_block_index + 1
            new_length = len(new_chain)
            if not self.is_valid_chain(new_chain):
                return False

            if new_cum_work > current_cum_work:
                pass  # Pokraèovat k replace
            elif new_cum_work == current_cum_work:
                if new_length > current_length:
                    pass  # Pokraèovat k replace
                elif new_length == current_length:
                    # Tie-breaker: Porovnat hash posledního bloku (menší hash vyhrává jako "lepší" chain)
                    if new_chain[-1].hash >= self.get_last_block().hash:
                        return False
                else:
                    return False
            else:
                return False

            # Najít fork point
            fork_index = -1
            min_length = min(self.max_block_index + 1, len(new_chain))
            for i in range(min_length):
                local_block = self.get_block_from_db(i)
                if local_block.hash != new_chain[i].hash:
                    break
                fork_index = i

            # Kontrola pevných blokù
            if fork_index >= 0:
                fork_confirmations = self.get_confirmations(self.get_block_from_db(fork_index).hash)
                if fork_confirmations >= CONFIRMATIONS_THRESHOLD:
                    p2p_node.add_log(f"{Fore.RED}Odmítnutí nového øetìzce: Pokus o pøepsání bloku s {fork_confirmations} potvrzeními.{Style.RESET_ALL}")
                    return False

            # Sbírat všechny tx_ids v novém øetìzci
            new_tx_ids = set()
            for block in new_chain:
                for tx in block.transactions:
                    new_tx_ids.add(tx.tx_id)

            # Znovuzaøadit osiøelé transakce z opuštìné vìtve
            orphaned_transactions = []
            for i in range(fork_index + 1, self.max_block_index + 1):
                local_block = self.get_block_from_db(i)
                for tx in local_block.transactions:
                    if tx.from_address != "COINBASE" and tx.tx_id not in new_tx_ids:
                        orphaned_transactions.append(tx)

            p2p_node.add_log(f"{Fore.YELLOW}Nalezen lepší øetìzec (kumulativní práce). Nahrazuji svùj øetìzec...{Style.RESET_ALL}")

            # Aktualizovat DB s novým chainem
            conn = sqlite3.connect(BLOCKCHAIN_DB, timeout=1.0)
            c = conn.cursor()
            c.execute("DELETE FROM blocks")
            for block in new_chain:
                transactions_json = json.dumps([tx.to_dict() for tx in block.transactions])
                target_hex = hex(block.target)[2:]
                c.execute("INSERT INTO blocks (block_index, timestamp, transactions, previous_hash, target_hex, nonce, block_hash, merkle_root) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                          (block.index, block.timestamp, transactions_json, block.previous_hash, target_hex, block.nonce, block.hash, block.merkle_root))
            conn.commit()
            conn.close()

            # Rebuild state
            self.rebuild_state()
            self.max_block_index = new_chain[-1].index
            self.chain = new_chain[-LAST_BLOCKS_TO_KEEP:] if len(new_chain) > LAST_BLOCKS_TO_KEEP else new_chain

            # Pøidat osiøelé transakce zpìt do mempoolu
            self.unconfirmed_transactions = []
            for tx in orphaned_transactions:
                if self.add_transaction(tx):
                    p2p_node.add_log(f"{Fore.GREEN}Osiøelá transakce {tx.tx_id} pøidána zpìt do mempoolu.{Style.RESET_ALL}")
                else:
                    p2p_node.add_log(f"{Fore.RED}Osiøelá transakce {tx.tx_id} nemohla být pøidána do mempoolu.{Style.RESET_ALL}")

            save_mempool(self.unconfirmed_transactions)
            return True
        finally:
            self.lock.release()

    def find_transaction_by_id(self, tx_id):
        for tx in self.unconfirmed_transactions:
            if tx.tx_id == tx_id:
                return tx, "Mempool"
        # Hledat v cache blokù
        for block in self.chain:
            for tx in block.transactions:
                if tx.tx_id == tx_id:
                    return tx, f"Blok #{block.index}"
        # Hledat v DB pro starší bloky
        conn = sqlite3.connect(BLOCKCHAIN_DB, timeout=1.0)
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

def save_data(droid_chain, wallets, password, peers):
    try:
        conn = sqlite3.connect(BLOCKCHAIN_DB, timeout=1.0)
        c = conn.cursor()
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
        for block in droid_chain.chain:
            transactions_json = json.dumps([tx.to_dict() for tx in block.transactions])
            target_hex = hex(block.target)[2:]
            c.execute('''
                INSERT OR REPLACE INTO blocks (block_index, timestamp, transactions, previous_hash, target_hex, nonce, block_hash, merkle_root)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (block.index, block.timestamp, transactions_json, block.previous_hash, target_hex, block.nonce, block.hash, block.merkle_root))
        conn.commit()
        conn.close()

        save_wallets_enc(wallets, password)

        save_mempool(droid_chain.unconfirmed_transactions)
        save_peers(peers)
        print(f"{Fore.GREEN}Data byla úspìšnì uložena.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Chyba pøi ukládání dat:{Style.RESET_ALL} {e}")

def save_wallets_enc(wallets, password):
    wallet_data = {
        address: binascii.hexlify(wallet.private_key.to_string()).decode()
        for address, wallet in wallets.items()
    }
    data_json = json.dumps(wallet_data).encode()
    
    salt = os.urandom(16)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = kdf.derive(password.encode())
    
    nonce = os.urandom(12)
    aesgcm = AESGCM(key)
    ciphertext_and_tag = aesgcm.encrypt(nonce, data_json, None)
    
    with open(WALLETS_FILE, 'wb') as f:
        f.write(salt + nonce + ciphertext_and_tag)

def save_mempool(unconfirmed_transactions):
    try:
        conn = sqlite3.connect(MEMPOOL_DB, timeout=1.0)
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
        # Smazat existující záznamy
        c.execute('DELETE FROM transactions')
        # Vložit nové transakce
        for tx in unconfirmed_transactions:
            c.execute('''
                INSERT INTO transactions (tx_id, from_address, to_address, amount, fee, nonce, timestamp, public_key, signature)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (tx.tx_id, tx.from_address, tx.to_address, tx.amount, tx.fee, tx.nonce, tx.timestamp, tx.public_key, tx.signature))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"{Fore.RED}Chyba pøi ukládání mempoolu:{Style.RESET_ALL} {e}")

def save_peers(peers):
    try:
        with open(PEERS_FILE, 'w') as f:
            json.dump(peers, f, indent=4)
    except Exception as e:
        print(f"{Fore.RED}Chyba pøi ukládání peers:{Style.RESET_ALL} {e}")

def load_data():
    wallets = {}
    droid_chain = None
    peers = []
    password = None

    # 1. ZPRACOVÁNÍ PENÌŽENEK A HESLA
    if os.path.exists(WALLETS_FILE):
        password = getpass.getpass(f"{Fore.BLUE}Zadejte heslo: {Style.RESET_ALL}")
        try:
            with open(WALLETS_FILE, 'rb') as f:
                data = f.read()
            salt = data[:16]
            nonce = data[16:28]
            ciphertext_and_tag = data[28:]
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            key = kdf.derive(password.encode())
            
            aesgcm = AESGCM(key)
            decrypted = aesgcm.decrypt(nonce, ciphertext_and_tag, None)
            wallet_data = json.loads(decrypted.decode())
            wallets = {
                address: Wallet(private_key)
                for address, private_key in wallet_data.items()
            }
            print(f"{Fore.GREEN}Penìženky byly naèteny ze souboru.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Chyba pøi dešifrování penìženek: Špatné heslo nebo poškozený soubor.{Style.RESET_ALL}")
            sys.exit(1)
    else:
        print(f"{Fore.YELLOW}Žádný šifrovaný soubor penìženek nenalezen. Vytváøím nový.{Style.RESET_ALL}")
        while True:
            pwd1 = getpass.getpass(f"{Fore.BLUE}Vytvoøte heslo (8-20 znakù): {Style.RESET_ALL}")
            if not 8 <= len(pwd1) <= 20:
                print(f"{Fore.RED}Délka hesla musí být mezi 8 a 20 znaky.{Style.RESET_ALL}")
                continue
            pwd2 = getpass.getpass(f"{Fore.BLUE}Potvrïte heslo: {Style.RESET_ALL}")
            if pwd1 == pwd2:
                password = pwd1
                break
            else:
                print(f"{Fore.RED}Hesla se neshodují.{Style.RESET_ALL}")
        wallets = {}
        save_wallets_enc(wallets, password)
        print(f"{Fore.GREEN}Nový šifrovaný soubor penìženek vytvoøen. Zálohujte si své privátní klíèe oddìlenì pro pøípad obnovy.{Style.RESET_ALL}")
        print() # <-- POŽADOVANÁ MEZERA

    # 2. NAÈTENÍ / VYTVOØENÍ BLOCKCHAINU AŽ PO ZADÁNÍ HESLA
    droid_chain = Blockchain(create_genesis=False)
    if os.path.exists(BLOCKCHAIN_DB):
        try:
            conn = sqlite3.connect(BLOCKCHAIN_DB, timeout=1.0)
            c = conn.cursor()
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
            droid_chain.max_block_index = c.fetchone()[0] or 0
            # Naèíst posledních LAST_BLOCKS_TO_KEEP blokù
            c.execute("SELECT * FROM blocks WHERE block_index > ? ORDER BY block_index", (droid_chain.max_block_index - LAST_BLOCKS_TO_KEEP,))
            rows = c.fetchall()
            droid_chain.chain = [Block.from_dict({
                'index': row[0],
                'timestamp': row[1],
                'transactions': json.loads(row[2]),
                'previous_hash': row[3],
                'target': row[4],
                'nonce': row[5],
                'hash': row[6],
                'merkle_root': row[7]
            }) for row in rows]
            # Rebuild state z DB
            droid_chain.rebuild_state()
            conn.close()
            print(f"{Fore.GREEN}Blockchain byl naèten z databáze (jen posledních {LAST_BLOCKS_TO_KEEP} blokù v RAM).{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Chyba pøi naèítání blockchainu:{Style.RESET_ALL} {e}")
            print(f"{Fore.YELLOW}Vytváøím nový blockchain s genesis blokem.{Style.RESET_ALL}")
            droid_chain.create_genesis_block()
            save_data(droid_chain, wallets, password, peers)
    else:
        print(f"{Fore.YELLOW}Databáze blockchainu nenalezena. Vytváøím nový blockchain s genesis blokem.{Style.RESET_ALL}")
        droid_chain.create_genesis_block()
        save_data(droid_chain, wallets, password, peers)
        
    # 3. NAÈTENÍ MEMPOOLU
    droid_chain.unconfirmed_transactions = load_mempool(droid_chain)

    # 4. NAÈTENÍ PEERS
    if os.path.exists(PEERS_FILE):
        try:
            with open(PEERS_FILE, 'r') as f:
                peers_data = json.load(f)
                peers = [tuple(p) for p in peers_data]
                print(f"{Fore.GREEN}Peers byly naèteny ze souboru.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Chyba pøi naèítání peers:{Style.RESET_ALL} {e}")

    return droid_chain, wallets, peers, password

def load_mempool(droid_chain):
    unconfirmed_transactions = []
    if os.path.exists(MEMPOOL_DB):
        try:
            conn = sqlite3.connect(MEMPOOL_DB, timeout=1.0)
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
            now = get_time()
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
                # Ne pøímo append, ale ovìøit pøes add_transaction (kontroluje duplicitu, nonce, atd.)
                if now - tx.timestamp < MEMPOOL_TX_EXPIRATION and droid_chain.add_transaction(tx):
                    unconfirmed_transactions.append(tx)
                else:
                    print(f"{Fore.RED}Transakce z mempoolu DB zamítnuta (duplicitní, neplatná nebo expirovaná): {tx.tx_id}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Chyba pøi naèítání mempoolu:{Style.RESET_ALL} {e}")
    return unconfirmed_transactions

class P2PNode:
    def __init__(self, blockchain, host, port, initial_peers):
        self.blockchain = blockchain
        self.host = host
        self.port = port
        self.peers = initial_peers
        self.server_thread = threading.Thread(target=self.start_server)
        self.running = True
        self.sync_thread = threading.Thread(target=self.sync_chain_periodically)
        self.sync_thread.daemon = True
        self.p2p_log = queue.Queue()
        self.rate_limit = defaultdict(list)  # IP -> list èasù požadavkù pro rate limiting
        self.tx_rate = defaultdict(list)  # addr (tuple) -> list timestamps transakcí pro tx rate limiting
        self.blacklist = set()  # Blacklist addr (IP, PORT) tuples pro DoS

    def add_log(self, message):
        self.p2p_log.put(message)

    def is_rate_limited(self, addr):
        now = time.time()
        self.rate_limit[addr[0]] = [t for t in self.rate_limit[addr[0]] if now - t < RATE_LIMIT_WINDOW]
        if len(self.rate_limit[addr[0]]) >= RATE_LIMIT_REQUESTS:
            self.blacklist.add(addr)
            return True
        self.rate_limit[addr[0]].append(now)
        return False

    def is_tx_rate_limited(self, addr):
        now = time.time()
        self.tx_rate[addr] = [t for t in self.tx_rate[addr] if now - t < TX_RATE_WINDOW]
        if len(self.tx_rate[addr]) >= TX_RATE_LIMIT:
            self.blacklist.add(addr)
            self.add_log(f"{Fore.RED}Uzol {addr} pøekroèil limit transakcí ({TX_RATE_LIMIT}/min), pøidán do blacklistu.{Style.RESET_ALL}")
            return True
        return False

    def is_blacklisted(self, addr):
        return addr in self.blacklist

    def is_peer_valid(self, peer_addr):
        # Jednoduchá validace pro Sybil - napø. omezit na MAX_PEERS, nebo vyžadovat PoW (zde jen limit)
        if len(self.peers) >= MAX_PEERS:
            return False
        return True

    def start_server(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen()
        self.add_log(f"{Fore.CYAN}Poslouchám na {self.host}:{self.port}...{Style.RESET_ALL}")
        server_socket.settimeout(1)
        while self.running:
            try:
                conn, addr = server_socket.accept()
                if self.is_blacklisted(addr):
                    conn.close()
                    continue
                if self.is_rate_limited(addr):
                    conn.close()
                    continue
                with conn:
                    conn.settimeout(10)
                    
                    # KROK 1: Pøeèteme 4 bajty pro délku zprávy
                    raw_msglen = conn.recv(4)
                    if not raw_msglen:
                        continue # Uzel se odpojil

                    msglen = struct.unpack('!I', raw_msglen)[0]
                    
                    # Ochrana proti DoS: Kontrola maximální velikosti zprávy
                    if msglen > MAX_MESSAGE_SIZE:
                        self.add_log(f"{Fore.RED}Pøijatá zpráva pøíliš velká od {addr}: {msglen} bajtù. Odmítnuto.{Style.RESET_ALL}")
                        self.blacklist.add(addr)
                        continue
                    
                    # KROK 2: Pøeèteme celou zprávu o dané délce
                    data_buffer = b''
                    while len(data_buffer) < msglen:
                        part = conn.recv(msglen - len(data_buffer))
                        if not part:
                            # Spojení bylo ztraceno døíve, než pøišla celá zpráva
                            data_buffer = None
                            break
                        data_buffer += part
                    
                    if not data_buffer:
                        self.add_log(f"{Fore.RED}Spojení s {addr} pøerušeno pøi pøijímání dat.{Style.RESET_ALL}")
                        continue
                    
                    if len(data_buffer) == msglen:
                        message = json.loads(data_buffer.decode('utf-8'))
                        self.handle_message(message, addr)
                    else:
                        self.add_log(f"{Fore.RED}Pøijata neúplná zpráva od {addr}. Oèekáváno {msglen}, pøijato {len(data_buffer)}.{Style.RESET_ALL}")

            except socket.timeout:
                pass
            except json.JSONDecodeError as e:
                self.add_log(f"{Fore.RED}Chyba dekódování JSON: {e}{Style.RESET_ALL}")
            except Exception as e:
                self.add_log(f"{Fore.RED}Chyba serveru: {e}{Style.RESET_ALL}")

    def handle_message(self, message, addr):
        self.add_log(f"\n{Fore.CYAN}Pøijata zpráva typu: {message['type']} od {addr}{Style.RESET_ALL}")

        if message['type'] == 'transaction':
            # Kontrola TX rate limitu pøed zpracováním
            if self.is_tx_rate_limited(addr):
                return
            # Pøidat timestamp transakce do tx_rate
            now = time.time()
            self.tx_rate[addr].append(now)
            tx_data = message['data']
            tx = Transaction.from_dict(tx_data)
            if self.blockchain.add_transaction(tx):
                self.add_log(f"{Fore.GREEN}Pøijata a ovìøena nová transakce.{Style.RESET_ALL}")
                save_mempool(self.blockchain.unconfirmed_transactions)
            else:
                self.add_log(f"{Fore.RED}Pøijatá transakce je neplatná, odmítnuta.{Style.RESET_ALL}")

        elif message['type'] == 'request_chain_info':
            self.add_log(f"{Fore.YELLOW}Pøijat požadavek na info o øetìzci, odesílám...{Style.RESET_ALL}")
            local_length = self.blockchain.max_block_index + 1
            local_last_hash = self.blockchain.get_last_block().hash
            self.send_data_to_peers({'type': 'response_chain_info', 'data': {'length': local_length, 'last_hash': local_last_hash}})

        elif message['type'] == 'response_chain_info':
            data = message['data']
            remote_length = data['length']
            remote_last_hash = data['last_hash']
            local_length = self.blockchain.max_block_index + 1
            local_last_hash = self.blockchain.get_last_block().hash
            if remote_length > local_length:
                self.add_log(f"{Fore.YELLOW}Detekován delší øetìzec, žádám o chybìjící bloky...{Style.RESET_ALL}")
                self.send_data_to_peers({'type': 'request_blocks', 'data': {'start_index': local_length}})
            elif remote_length == local_length:
                if remote_last_hash < local_last_hash:
                    self.add_log(f"{Fore.YELLOW}Stejná délka, ale lepší hash (tie-breaker), žádám o celý øetìzec...{Style.RESET_ALL}")
                    self.send_data_to_peers({'type': 'request_full_chain'})
                else:
                    self.add_log(f"{Fore.GREEN}Øetìzce synchronizovány (stejná délka, náš lepší nebo stejný).{Style.RESET_ALL}")

        elif message['type'] == 'request_blocks':
            start_index = message['data']['start_index']
            self.add_log(f"{Fore.YELLOW}Pøijat požadavek na bloky od indexu {start_index}, odesílám...{Style.RESET_ALL}")
            blocks_data = []
            local_length = self.blockchain.max_block_index + 1
            conn = sqlite3.connect(BLOCKCHAIN_DB, timeout=1.0)
            c = conn.cursor()
            c.execute("SELECT * FROM blocks WHERE block_index >= ? ORDER BY block_index", (start_index,))
            rows = c.fetchall()
            conn.close()
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
                blocks_data.append(block_data)
            self.send_data_to_peers({'type': 'response_blocks', 'data': blocks_data})

        elif message['type'] == 'response_blocks':
            blocks_data = message['data']
            self.add_log(f"{Fore.YELLOW}Pøijaty chybìjící bloky ({len(blocks_data)}), pøidávám...{Style.RESET_ALL}")
            current_index = self.blockchain.max_block_index + 1
            last_hash = self.blockchain.get_last_block().hash
            added = False
            for block_data in blocks_data:
                block = Block.from_dict(block_data)
                if block.index != current_index or block.previous_hash != last_hash:
                    self.add_log(f"{Fore.RED}Blok nenesleduje (fork detekován), zahajuji full sync...{Style.RESET_ALL}")
                    self.send_data_to_peers({'type': 'request_full_chain'})
                    break
                if self.blockchain.add_block(block, block.hash):
                    current_index += 1
                    last_hash = block.hash
                    added =True
                else:
                    self.add_log(f"{Fore.RED}Neplatný blok, pøerušuji pøidávání.{Style.RESET_ALL}")
                    break
            if added:
                save_data(self.blockchain, wallets, password, self.peers)
                self.add_log(f"{Fore.GREEN}Chybìjící bloky pøidány a uloženy.{Style.RESET_ALL}")

        elif message['type'] == 'request_full_chain':
            self.add_log(f"{Fore.YELLOW}Pøijat požadavek na celý øetìzec, odesílám...{Style.RESET_ALL}")
            conn = sqlite3.connect(BLOCKCHAIN_DB, timeout=1.0)
            c = conn.cursor()
            c.execute("SELECT * FROM blocks ORDER BY block_index")
            rows = c.fetchall()
            chain_data = [Block.from_dict({
                'index': row[0],
                'timestamp': row[1],
                'transactions': json.loads(row[2]),
                'previous_hash': row[3],
                'target': row[4],
                'nonce': row[5],
                'hash': row[6],
                'merkle_root': row[7]
            }).to_dict() for row in rows]
            conn.close()
            self.send_data_to_peers({'type': 'response_full_chain', 'data': chain_data})

        elif message['type'] == 'response_full_chain':
            new_chain_data = message['data']
            if self.blockchain.replace_chain(new_chain_data):
                save_data(self.blockchain, wallets, password, self.peers)
                self.add_log(f"{Fore.GREEN}Øetìzec byl úspìšnì synchronizován a uložen.{Style.RESET_ALL}")
            else:
                self.add_log(f"{Fore.YELLOW}Pøijatý øetìzec není delší nebo platný, odmítám ho.{Style.RESET_ALL}")

        elif message['type'] == 'new_block':
            new_block_data = message['data']
            new_block = Block.from_dict(new_block_data)

            if self.blockchain.mining_in_progress:
                self.blockchain.mining_in_progress = False
                self.add_log(f"{Fore.YELLOW}Tìžba zastavena, pøijat nový blok.{Style.RESET_ALL}")

            last_block_hash = self.blockchain.get_last_block().hash
            if new_block.previous_hash == last_block_hash:
                if self.blockchain.add_block(new_block, new_block.hash):
                    self.add_log(f"{Fore.GREEN}Pøijat a pøidán nový blok {new_block.index} od jiného uzlu.{Style.RESET_ALL}")
                    confirmed_tx_ids = {tx.tx_id for tx in new_block.transactions if tx.from_address != "COINBASE"}
                    self.blockchain.unconfirmed_transactions = [
                        tx for tx in self.blockchain.unconfirmed_transactions
                        if tx.tx_id not in confirmed_tx_ids
                    ]
                    self.add_log(f"{Fore.YELLOW}Mempool byl aktualizován, odstranìno {len(confirmed_tx_ids)} potvrzených transakcí.{Style.RESET_ALL}")
                    save_data(self.blockchain, wallets, password, self.peers)
            else:
                # Kontrola, zda previous existuje (fork nebo orphan)
                conn = sqlite3.connect(BLOCKCHAIN_DB, timeout=1.0)
                c = conn.cursor()
                c.execute("SELECT 1 FROM blocks WHERE block_hash = ?", (new_block.previous_hash,))
                exists = c.fetchone() is not None
                conn.close()
                if exists:
                    self.add_log(f"{Fore.YELLOW}Pøijatý blok není navázaný na náš poslední blok. Zahajuji synchronizaci...{Style.RESET_ALL}")
                    self.send_data_to_peers({'type': 'request_chain_info'})
                else:
                    self.blockchain.add_orphan_block(new_block)
                    self.add_log(f"{Fore.YELLOW}Pøijat orphan blok {new_block.index}, uložen do poolu.{Style.RESET_ALL}")

        elif message['type'] == 'request_mempool':
            self.add_log(f"{Fore.YELLOW}Pøijat požadavek na mempool, odesílám...{Style.RESET_ALL}")
            self.send_data_to_peers({'type': 'response_mempool', 'data': [tx.to_dict() for tx in self.blockchain.unconfirmed_transactions]})

        elif message['type'] == 'response_mempool':
            tx_data_list = message['data']
            self.add_log(f"{Fore.YELLOW}Pøijat mempool s {len(tx_data_list)} transakcemi.{Style.RESET_ALL}")
            for tx_data in tx_data_list:
                tx = Transaction.from_dict(tx_data)
                if tx.tx_id not in {t.tx_id for t in self.blockchain.unconfirmed_transactions}:
                    tx_nonce = self.blockchain.get_next_nonce(tx.from_address)
                    if tx.nonce == tx_nonce and self.blockchain.add_transaction(tx):
                        self.add_log(f"{Fore.GREEN}Pøidána nová transakce z mempoolu od uzlu: {tx.tx_id}{Style.RESET_ALL}")
                    else:
                        self.add_log(f"{Fore.RED}Transakce z mempoolu zamítnuta (neplatná nonce nebo jiná chyba): {tx.tx_id}{Style.RESET_ALL}")
            save_mempool(self.blockchain.unconfirmed_transactions)
            
        elif message['type'] == 'new_peer':
            new_peer_addr = tuple(message['data'])
            if new_peer_addr not in self.peers and new_peer_addr != (self.host, self.port) and self.is_peer_valid(new_peer_addr):
                self.connect_to_peer(new_peer_addr)

    def connect_to_peer(self, peer_addr):
        if peer_addr not in self.peers:
            try:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.settimeout(3)
                client_socket.connect(peer_addr)
                self.peers.append(peer_addr)
                self.add_log(f"{Fore.GREEN}Úspìšnì pøipojeno k uzlu {peer_addr}{Style.RESET_ALL}")
                self.send_data_to_peers({'type': 'request_mempool'})
                self.send_data_to_peers({'type': 'request_chain_info'})
                save_peers(self.peers)
                client_socket.close()
                return True
            except ConnectionRefusedError:
                print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Nelze se pøipojit k uzlu {peer_addr}")
            except Exception as e:
                print(f"{Fore.RED}Chyba pøi pøipojování:{Style.RESET_ALL} {e}")
        return False
    
    def connect_to_all_peers(self):
        for peer in list(self.peers):
            self.connect_to_peer(peer)

    def _send_to_single_peer(self, peer, data):
        """Pomocná funkce pro odeslání dat jednomu uzlu (pro použití ve vláknì)."""
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(5)  # Nastavíme timeout pro pøipojení
            client_socket.connect(peer)

            # Zabalíme data do JSON a zakódujeme do UTF-8
            message = json.dumps(data).encode('utf-8')
            # Vytvoøíme 4bajtový prefix s délkou zprávy
            message_length = struct.pack('!I', len(message))
            
            # Odešleme prefix a poté samotnou zprávu
            client_socket.sendall(message_length + message)
            
            client_socket.close()
        except Exception as e:
            self.add_log(f"{Fore.RED}Chyba pøi odesílání dat uzlu {peer}: {e}{Style.RESET_ALL}")


    def send_data_to_peers(self, data):
        """Odešle data všem pøipojeným uzlùm, každý v samostatném vláknì."""
        for peer in self.peers:
            thread = threading.Thread(target=self._send_to_single_peer, args=(peer, data))
            thread.daemon = True
            thread.start()
    
    def sync_chain_periodically(self):
        while self.running:
            time.sleep(10)
            if self.peers:
                self.add_log(f"{Fore.YELLOW}Synchronizuji blockchain a mempool se sousedními uzly...{Style.RESET_ALL}")
                self.send_data_to_peers({'type': 'request_chain_info'})
                self.send_data_to_peers({'type': 'request_mempool'})

    # --- NOVÁ METODA PRO KONTROLU ONLINE STAVU PEERÙ ---
    def check_peer_connectivity(self, peer, online_peers_list, lock):
        """Pomocná funkce pro ovìøení konektivity jednoho peeru ve vláknì."""
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(2)  # Krátký timeout pro pokus o pøipojení
            client_socket.connect(peer)
            client_socket.close()
            with lock:
                online_peers_list.append(peer)
        except (socket.timeout, ConnectionRefusedError, OSError):
            # Peer je offline, nic se nedìje
            pass

    # --- NOVÁ METODA PRO ZÍSKÁNÍ SEZNAMU ONLINE PEERÙ ---
    def get_online_peers(self):
        """
        Ovìøí konektivitu všech známých peerù a vrátí seznam tìch, které jsou online.
        Používá vlákna pro paralelní a efektivní kontrolu.
        """
        online_peers = []
        threads = []
        lock = threading.Lock()  # Zámek pro bezpeèný zápis do seznamu z více vláken

        # Vytvoøení a spuštìní vlákna pro každý peer
        for peer in self.peers:
            thread = threading.Thread(target=self.check_peer_connectivity, args=(peer, online_peers, lock))
            thread.daemon = True  # Umožní ukonèení programu, i když vlákna bìží
            threads.append(thread)
            thread.start()

        # Poèká na dokonèení všech vláken
        for thread in threads:
            thread.join()

        return online_peers

def is_valid_address(address):
    if not isinstance(address, str) or not address.startswith(TICKER):
        return False
    if len(address) == 67:
        return all(c in '0123456789abcdef' for c in address[3:])
    elif len(address) == 71:
        base = address[:-4]
        expected_checksum = hashlib.sha3_256(base.encode()).hexdigest()[:4]
        return address[-4:] == expected_checksum and all(c in '0123456789abcdef' for c in address[3:])
    else:
        return False

def is_valid_private_key(key_hex):
    if not isinstance(key_hex, str):
        return False
    if len(key_hex) != 64:
        return False
    try:
        ecdsa.SigningKey.from_string(binascii.unhexlify(key_hex), curve=ecdsa.SECP256k1)
        return True
    except (binascii.Error, ecdsa.BadSignatureError):
        return False

def show_p2p_log():
    print(f"\n{Fore.YELLOW}--- Log P2P sítì (stisknìte Enter pro návrat) ---{Style.RESET_ALL}")
    while not p2p_node.p2p_log.empty():
        print(p2p_node.p2p_log.get())
    input()

def get_mempool_size_bytes(unconfirmed_transactions):
    return sum(tx.get_size() for tx in unconfirmed_transactions)

def print_menu():
    print(f"\n{Fore.YELLOW}--- Menu ---{Style.RESET_ALL}")
    print(f"{Fore.GREEN}1{Style.RESET_ALL} - Vytìžit nový blok")
    print(f"{Fore.GREEN}2{Style.RESET_ALL} - Vytvoøit transakci")
    print(f"{Fore.GREEN}3{Style.RESET_ALL} - Zobrazit nepotvrzené transakce")
    print(f"{Fore.GREEN}4{Style.RESET_ALL} - Zobrazit blockchain")
    print(f"{Fore.GREEN}5{Style.RESET_ALL} - Vytvoøit novou penìženku")
    print(f"{Fore.GREEN}6{Style.RESET_ALL} - Zobrazit penìženky a zùstatky")
    print(f"{Fore.GREEN}7{Style.RESET_ALL} - Smazat penìženku")
    print(f"{Fore.GREEN}8{Style.RESET_ALL} - Importovat privátní klíè")
    print(f"{Fore.GREEN}9{Style.RESET_ALL} - Exportovat privátní klíè")
    print(f"{Fore.GREEN}10{Style.RESET_ALL} - Pøipojit se k uzlu")
    print(f"{Fore.GREEN}11{Style.RESET_ALL} - Zobrazit stav uzlù")
    print(f"{Fore.GREEN}12{Style.RESET_ALL} - Uložit a uložit")
    print(f"{Fore.GREEN}13{Style.RESET_ALL} - Zobrazit historii transakcí pro adresu")
    print(f"{Fore.GREEN}14{Style.RESET_ALL} - Zobrazit log P2P sítì")
    print(f"{Fore.GREEN}15{Style.RESET_ALL} - Zobrazit detaily transakce podle TX ID")
    print(f"{Fore.GREEN}16{Style.RESET_ALL} - Zobrazit celkovou nabídku mincí")
    print(f"{Fore.GREEN}17{Style.RESET_ALL} - Automaticky se pøipojit ke známým uzlùm")
    print(f"{Fore.GREEN}18{Style.RESET_ALL} - Manuálnì pøidat nový uzel")
    print(f"{Fore.GREEN}19{Style.RESET_ALL} - Smazat uzel")
    print(f"{Fore.GREEN}20{Style.RESET_ALL} - Zobrazit blok")

def verify_genesis_address():
    current_hash = hashlib.sha256(GENESIS_ADDRESS.encode()).hexdigest()
    if current_hash != GENESIS_ADDRESS_EXPECTED_HASH:
        print(f"{Fore.RED}Chyba: Genesis adresa byla zmìnìna! Program se ukonèuje.{Style.RESET_ALL}")
        sys.exit(1)

def verify_genesis_block(chain):
    genesis_block = chain.get_block_from_db(0)
    if genesis_block.timestamp != GENESIS_TIMESTAMP or genesis_block.hash != GENESIS_BLOCK_EXPECTED_HASH:
        print(f"{Fore.RED}Chyba: Genesis blok byl zmìnìn (timestamp nebo hash nesouhlasí)! Program se ukonèuje.{Style.RESET_ALL}")
        sys.exit(1)

def main():
    global wallets
    global p2p_node
    global password

    # Kontrola integrity genesis adresy
    verify_genesis_address()

    # Synchronizace èasu s NTP na zaèátku
    sync_time_with_ntp()

    if len(sys.argv) > 1:
        p2p_port = int(sys.argv[1])
    else:
        p2p_port = 5001

    droid_chain, wallets, peers, password = load_data()

    # Kontrola integrity genesis bloku po naètení
    verify_genesis_block(droid_chain)

    p2p_node = P2PNode(droid_chain, P2P_HOST, p2p_port, peers)
    p2p_node.server_thread.daemon = True
    p2p_node.server_thread.start()
    p2p_node.sync_thread.start()

    print(f"\n{Fore.GREEN} Vítejte v {PROJECT_NAME} ({TICKER}) {Style.RESET_ALL}")
    print(f"{Fore.CYAN}Tento uzel bìží na portu {p2p_port}.{Style.RESET_ALL}")
    while True:
        try:
            print_menu()
            try:
                choice = input(f"\n{Fore.BLUE}Zadejte èíslo volby: {Style.RESET_ALL}").strip()
            except (EOFError, KeyboardInterrupt):
                # Handle Ctrl+C or Ctrl+D gracefully to exit the program
                choice = "12"

            if choice == "1":
                # --- ZMÌNA: KONTROLA PØIPOJENÍ PØED TÌŽBOU ---
                if is_read_only:
                    print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Tìžba není možné. Program bìží v read-only režimu kvùli nedostupnosti NTP serverù.")
                    continue
                if not p2p_node.get_online_peers():
                    print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Tìžba není možné. Musíte být pøipojen k alespoò jednomu dalšímu uzlu.")
                    continue

                if not wallets:
                    print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Žádné penìženky nejsou dostupné. Nejdøíve vytvoøte nebo importujte penìženku.")
                    continue
                print(f"{Fore.YELLOW}Dostupné penìženky pro tìžbu:{Style.RESET_ALL}")
                wallet_list = list(wallets.keys())
                for i, addr in enumerate(wallet_list, 1):
                    print(f" {i}. {Fore.CYAN}{addr}{Style.RESET_ALL}")
                try:
                    selected = int(input(f"{Fore.BLUE}Vyberte èíslo penìženky tìžaøe: {Style.RESET_ALL}"))
                    if 1 <= selected <= len(wallet_list):
                        miner_address = wallet_list[selected - 1]
                        print(f"{Fore.GREEN}Tìžím na adresu: {Fore.CYAN}{miner_address}{Style.RESET_ALL}")
                        droid_chain.mine(miner_address)
                    else:
                        print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Neplatná volba.")
                except ValueError:
                    print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Neplatný vstup. Zadejte èíslo.")
            
            elif choice == "2":
                # --- ZMÌNA: KONTROLA PØIPOJENÍ PØED VYTVOØENÍM TRANSAKCE ---
                if is_read_only:
                    print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Vytvoøení transakce není možné. Program bìží v read-only režimu kvùli nedostupnosti NTP serverù.")
                    continue
                if not p2p_node.get_online_peers():
                    print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Vytvoøení transakce není možné. Musíte být pøipojen k alespoò jednomu dalšímu uzlu.")
                    continue

                from_address = input(f"Zadejte ADRESU penìženky odesílatele: ")
                if from_address not in wallets:
                    print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Penìženka s adresou '{from_address}' neexistuje. Nejdøíve ji vytvoøte nebo importujte.")
                    continue
                to_address = input(f"Zadejte ADRESU pøíjemce: ")
                if not is_valid_address(to_address):
                    print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Neplatný formát adresy pøíjemce.")
                    continue
                if from_address == to_address:
                    print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Nelze posílat peníze na stejnou adresu.")
                    continue
                try:
                    amount = float(input(f"Zadejte èástku: "))
                    amount_in_decimal = int(amount * (10 ** DECIMALS))
                    if amount_in_decimal < MIN_TX_AMOUNT:
                        print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Èástka transakce je pøíliš malá. Minimální èástka je {format(MIN_TX_AMOUNT / (10 ** DECIMALS), f'.{DECIMALS}f')} {TICKER}.")
                        continue
                except ValueError:
                    print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Èástka musí být èíslo.")
                    continue

                # Pøedbìžná kontrola zùstatku ihned po zadání èástky (pro amount + min fee)
                current_available_balance = droid_chain.get_confirmed_balance(from_address)
                for tx_in_mempool in droid_chain.unconfirmed_transactions:
                    if tx_in_mempool.from_address == from_address:
                        current_available_balance -= (tx_in_mempool.amount + tx_in_mempool.fee)
                if current_available_balance < amount_in_decimal + TX_FEE_MIN:
                    print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Nedostateèný zùstatek pro tuto èástku (vèetnì min. poplatku). K dispozici: {format(current_available_balance / (10**DECIMALS), f'.{DECIMALS}f')} {TICKER}")
                    continue

                try:
                    fee_input = input(f"Zadejte poplatek ({format(TX_FEE_MIN/(10**DECIMALS), f'.{DECIMALS}f')}-{format(TX_FEE_MAX/(10**DECIMALS), f'.{DECIMALS}f')} {TICKER}, prázdné pro {format(TX_FEE_MIN/(10**DECIMALS), f'.{DECIMALS}f')}): ")
                    if fee_input == "":
                        fee = TX_FEE_MIN
                    else:
                        fee_value = float(fee_input)
                        if TX_FEE_MIN / (10**DECIMALS) <= fee_value <= TX_FEE_MAX / (10**DECIMALS):
                            fee = int(fee_value * (10 ** DECIMALS))
                        else:
                            print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Poplatek za transakci je mimo povolený rozsah.")
                            continue
                except ValueError:
                    print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Neplatný formát poplatku.")
                    continue

                from_wallet = wallets[from_address]
                nonce = droid_chain.get_next_nonce(from_address)

                tx = Transaction(
                    from_wallet.address,
                    to_address,
                    amount_in_decimal,
                    fee,
                    nonce=nonce
                )

                tx.public_key = binascii.hexlify(from_wallet.public_key.to_string()).decode()
                tx.signature = from_wallet.sign_transaction(tx)

                # Kontrola duplicit po vytvoøení transakce (duplicitní TX ID a nonce v mempoolu)
                if any(t.tx_id == tx.tx_id for t in droid_chain.unconfirmed_transactions):
                    print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Duplicitní TX ID {tx.tx_id} po vytvoøení. Transakce odmítnuta.")
                    continue
                if tx.from_address != "COINBASE" and any(t.nonce == tx.nonce and t.from_address == tx.from_address for t in droid_chain.unconfirmed_transactions):
                    print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Duplicitní nonce {tx.nonce} pro adresu {tx.from_address} po vytvoøení. Transakce odmítnuta.")
                    continue

                if droid_chain.add_transaction(tx):
                    print(f"{Fore.GREEN}Transakce byla úspìšnì pøidána do fronty.{Style.RESET_ALL}")
                    print(f" TX ID: {Fore.MAGENTA}{tx.tx_id}{Style.RESET_ALL}")
                    print(f" Nonce: {Fore.MAGENTA}{tx.nonce}{Style.RESET_ALL}")
                    print(f"{Fore.GREEN}Transakce byla podepsána privátním klíèem.{Style.RESET_ALL}")
                    save_mempool(droid_chain.unconfirmed_transactions)
                    p2p_node.send_data_to_peers({'type': 'transaction', 'data': tx.to_dict()})
            
            elif choice == "3":
                print(f"\n{Fore.YELLOW}--- Mempool (nepotvrzené transakce) ---{Style.RESET_ALL}")
                mempool_size = get_mempool_size_bytes(droid_chain.unconfirmed_transactions)
                print(f"Velikost mempoolu: {Fore.CYAN}{mempool_size / 1024:.2f} KB / {MAX_MEMPOOL_SIZE_BYTES / 1024 / 1024:.2f} MB{Style.RESET_ALL}")
                if not droid_chain.unconfirmed_transactions:
                    print(f"{Fore.CYAN}Mempool je prázdný.{Style.RESET_ALL}")
                else:
                    for tx in droid_chain.unconfirmed_transactions:
                        print(f"TX ID: {Fore.CYAN}{tx.tx_id}{Style.RESET_ALL}")
                        print(f" Od: {tx.from_address}")
                        print(f" Komu: {tx.to_address}")
                        print(f" Èástka: {Fore.MAGENTA}{format(tx.amount / (10 ** DECIMALS), f'.{DECIMALS}f')} {TICKER}{Style.RESET_ALL}")
                        print(f" Poplatek: {Fore.MAGENTA}{format(tx.fee / (10 ** DECIMALS), f'.{DECIMALS}f')} {TICKER}{Style.RESET_ALL}")
                        print(f" Nonce: {Fore.MAGENTA}{tx.nonce}{Style.RESET_ALL}")
                        print(f" Èas: {time.strftime('%d.%m.%Y %H:%M:%S UTC+00:00', time.gmtime(tx.timestamp))}")
                        if tx.signature:
                            print(f" Podpis: {Fore.BLUE}{tx.signature}{Style.RESET_ALL}")
                        else:
                            print(f" Podpis: {Fore.RED}žádný (chybí){Style.RESET_ALL}")
                        print("-" * 20)
            
            elif choice == "4":
                conn = sqlite3.connect(BLOCKCHAIN_DB, timeout=1.0)
                c = conn.cursor()
                c.execute("SELECT * FROM blocks ORDER BY block_index")
                rows = c.fetchall()
                conn.close()
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
                    block = Block.from_dict(block_data)
                    target_hex = hex(block.target)[2:]
                    print(f"Blok #{block.index}")
                    print(f" Hash: {Fore.MAGENTA}{block.hash}{Style.RESET_ALL}")
                    print(f" Merkle root: {Fore.CYAN}{block.merkle_root}{Style.RESET_ALL}")
                    print(f" Cílový target (hex): {Fore.CYAN}{target_hex}{Style.RESET_ALL}")
                    print(f" Pøedchozí hash: {Fore.MAGENTA}{block.previous_hash}{Style.RESET_ALL}")
                    print(f" PoW Nonce: {Fore.CYAN}{block.nonce}{Style.RESET_ALL}")
                    print(f" Èas: {Fore.CYAN}{time.strftime('%d.%m.%Y %H:%M:%S UTC+00:00', time.gmtime(block.timestamp))}{Style.RESET_ALL}")
                    print(f" Velikost bloku: {Fore.CYAN}{block.get_size() / 1024:.2f} KB{Style.RESET_ALL}")
                    print(f" Poèet potvrzení: {Fore.CYAN}{droid_chain.get_confirmations(block.hash)}{Style.RESET_ALL}")
                    print(f" Poèet transakcí: {len(block.transactions)}")
                    if block.transactions:
                        print(f" {Fore.YELLOW}Transakce:{Style.RESET_ALL}")
                        
                        for tx in block.transactions:
                            print(f" - TX ID: {Fore.CYAN}{tx.tx_id}{Style.RESET_ALL}")
                            print(f"   Od: {tx.from_address}")
                            print(f"   Komu: {tx.to_address}")
                            print(f"   Èástka: {Fore.CYAN}{format(tx.amount / (10 ** DECIMALS), f'.{DECIMALS}f')} {TICKER}{Style.RESET_ALL}")
                            if tx.from_address != "COINBASE":
                                print(f"   Poplatek: {format(tx.fee / (10 ** DECIMALS), f'.{DECIMALS}f')} {TICKER}")
                                print(f"   TX Nonce: {Fore.MAGENTA}{tx.nonce}{Style.RESET_ALL}")
                            if tx.signature:
                                print(f"   Podpis: {Fore.BLUE}{tx.signature}{Style.RESET_ALL}")
                            else:
                                print(f"   Podpis: {Fore.RED}žádný (chybí){Style.RESET_ALL}")
                    print("=" * 40)
                
                total_size = sum(block.get_size() for block in droid_chain.chain)  # Pøibližné, pro celou DB by to bylo nutné poèítat z DB
                total_size_kb = total_size / 1024
                total_size_mb = total_size_kb / 1024
                print(f"Velikost blockchainu: {Fore.CYAN}{total_size_kb:.2f} KB / {total_size_mb:.2f} MB{Style.RESET_ALL} (pøibližné)")
                print(f"Celkový poèet blokù: {Fore.CYAN}{droid_chain.max_block_index + 1}{Style.RESET_ALL}")
                print(f"Celkový poèet transakcí: {Fore.CYAN}{len(droid_chain.all_tx_ids)}{Style.RESET_ALL}")
            
            elif choice == "5":
                new_wallet = Wallet()
                wallets[new_wallet.address] = new_wallet
                print(f"{Fore.GREEN}Nová penìženka byla vytvoøena!{Style.RESET_ALL}")
                print(f" Adresa: {Fore.CYAN}{new_wallet.address}{Style.RESET_ALL}")
                print(f" Privátní klíè (hex): {Fore.RED}{binascii.hexlify(new_wallet.private_key.to_string()).decode()}{Style.RESET_ALL}")
                save_data(droid_chain, wallets, password, p2p_node.peers)
            
            elif choice == "6":
                print(f"\n{Fore.YELLOW}--- Penìženky a zùstatky ---{Style.RESET_ALL}")
                if not wallets:
                    print(f"{Fore.CYAN}Žádné penìženky nebyly nalezeny.{Style.RESET_ALL}")
                else:
                    for address, wallet in wallets.items():
                        confirmed_balance = droid_chain.get_confirmed_balance(address)
                        
                        pending_outgoing = []
                        pending_incoming = []
                        
                        pending_outgoing_sum = 0
                        for tx in droid_chain.unconfirmed_transactions:
                            if tx.from_address == address:
                                pending_outgoing.append(tx)
                                pending_outgoing_sum += tx.amount + tx.fee
                            if tx.to_address == address:
                                pending_incoming.append(tx)

                        total_balance = confirmed_balance - pending_outgoing_sum
                        pending_incoming_sum = sum(tx.amount for tx in pending_incoming)

                        print(f"Adresa: {Fore.CYAN}{address}{Style.RESET_ALL}")
                        print(f" Celkový zùstatek: {Fore.MAGENTA}{format(total_balance / (10 ** DECIMALS), f'.{DECIMALS}f')} {TICKER}{Style.RESET_ALL}")
                        
                        
                        if pending_outgoing:
                            print(f" Pending (-): {Fore.RED} -{format(pending_outgoing_sum / (10 ** DECIMALS), f'.{DECIMALS}f')} {TICKER}{Style.RESET_ALL}")
                            for tx in pending_outgoing:
                                print(f"  Pøíjemce: {tx.to_address}")
                                print(f"  Èástka: {Fore.RED}-{format(tx.amount / (10 ** DECIMALS), f'.{DECIMALS}f')} {TICKER}{Style.RESET_ALL}")
                                print(f"  Poplatek: {Fore.RED}-{format(tx.fee / (10 ** DECIMALS), f'.{DECIMALS}f')} {TICKER}{Style.RESET_ALL}")
                        
                        if pending_incoming:
                            print(f" Pending (+): {Fore.GREEN}+{format(pending_incoming_sum / (10 ** DECIMALS), f'.{DECIMALS}f')} {TICKER}{Style.RESET_ALL}")
                            for tx in pending_incoming:
                                print(f"  Odesílatel: {tx.from_address}")
                                print(f"  Èástka: {Fore.GREEN}+{format(tx.amount / (10 ** DECIMALS), f'.{DECIMALS}f')} {TICKER}{Style.RESET_ALL}")

                        print("-" * 20)

            elif choice == "7":
                address_to_delete = input(f"Zadejte ADRESU penìženky, kterou chcete smazat: ")
                if address_to_delete in wallets:
                    del wallets[address_to_delete]
                    print(f"{Fore.GREEN}Penìženka '{address_to_delete}' byla úspìšnì smazána.{Style.RESET_ALL}")
                    save_data(droid_chain, wallets, password, p2p_node.peers)
                else:
                    print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Penìženka s adresou '{address_to_delete}' neexistuje.")
            
            elif choice == "8":
                key_hex = input(f"Zadejte privátní klíè (hex): ")
                if not is_valid_private_key(key_hex):
                    print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Neplatný formát privátního klíèe.")
                    continue
                imported_wallet = Wallet(private_key=key_hex)
                wallets[imported_wallet.address] = imported_wallet
                print(f"{Fore.GREEN}Penìženka byla úspìšnì importována!{Style.RESET_ALL}")
                print(f" Adresa: {Fore.CYAN}{imported_wallet.address}{Style.RESET_ALL}")
                save_data(droid_chain, wallets, password, p2p_node.peers)
            
            elif choice == "9":
                address = input(f"Zadejte ADRESU penìženky, jejíž klíè chcete exportovat: ")
                if address in wallets:
                    private_key_hex = binascii.hexlify(wallets[address].private_key.to_string()).decode()
                    print(f"{Fore.GREEN}Privátní klíè pro adresu '{address}':{Style.RESET_ALL} {Fore.RED}{private_key_hex}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}Chyba:{Style.RESET_ALL} Penìženka s adresou '{address}' neexistuje.")
            
            elif choice == "10":
                peer_ip = input("Zadejte IP adresu uzlu k pøipojení: ")
                peer_port = int(input("Zadejte port uzlu: "))
                p2p_node.connect_to_peer((peer_ip, peer_port))
            
            # --- VYLEPŠENÉ ZOBRAZENÍ STAVU UZLÙ ---
            elif choice == "11":
                if not p2p_node.peers:
                    print(f"\n{Fore.YELLOW}Žádné uzly nejsou uloženy.{Style.RESET_ALL}")
                else:
                    print(f"\n{Fore.YELLOW}--- Stav známých uzlù ---{Style.RESET_ALL}")
                    online_peers = p2p_node.get_online_peers()
                    for peer in p2p_node.peers:
                        if peer in online_peers:
                            status = f"{Fore.GREEN}[ONLINE]{Style.RESET_ALL}"
                        else:
                            status = f"{Fore.RED}[OFFLINE]{Style.RESET_ALL}"
                        print(f"  {Fore.CYAN}{peer[0]}:{peer[1]}{Style.RESET_ALL} {status}")
            
            elif choice == "12":
                p2p_node.running = False
                print(f"\n{Fore.YELLOW}Ukládám a vypínám...{Style.RESET_ALL}")
                save_data(droid_chain, wallets, password, p2p_node.peers)
                if os.path.exists(MEMPOOL_DB):
                    os.remove(MEMPOOL_DB)
                    print(f"{Fore.GREEN}Mempool databáze byla smazána.{Style.RESET_ALL}")
                break
            
            elif choice == "13":
                address = input(f"Zadejte ADRESU pro zobrazení historie transakcí: ")
                print(f"\n{Fore.YELLOW}--- Historie transakcí pro adresu '{address}' ---{Style.RESET_ALL}")
                confirmed_balance = droid_chain.get_confirmed_balance(address)
                print(f" Potvrzený zùstatek: {Fore.MAGENTA}{format(confirmed_balance / (10 ** DECIMALS), f'.{DECIMALS}f')} {TICKER}{Style.RESET_ALL}")
                tx_found = False
                conn = sqlite3.connect(BLOCKCHAIN_DB, timeout=1.0)
                c = conn.cursor()
                c.execute("SELECT block_index, transactions FROM blocks ORDER BY block_index")
                for row in c:
                    transactions = json.loads(row[1])
                    for tx_data in transactions:
                        tx = Transaction.from_dict(tx_data)
                        if tx.from_address == address or tx.to_address == address:
                            tx_found = True
                            if tx.from_address == address:
                                direction = f"{Fore.RED}Odesláno{Style.RESET_ALL}"
                            else:
                                direction = f"{Fore.GREEN}Pøijato{Style.RESET_ALL}"
                            print(f"TX ID: {Fore.CYAN}{tx.tx_id}{Style.RESET_ALL}")
                            print(f" Blok: #{row[0]}")
                            print(f" Smìr: {direction}")
                            print(f" Od: {tx.from_address}")
                            print(f" Komu: {tx.to_address}")
                            print(f" Èástka: {format(tx.amount / (10 ** DECIMALS), f'.{DECIMALS}f')} {TICKER}")
                            if tx.from_address != "COINBASE":
                                print(f" Poplatek: {format(tx.fee / (10 ** DECIMALS), f'.{DECIMALS}f')} {TICKER}")
                            print(f" Èas: {time.strftime('%d.%m.%Y %H:%M:%S UTC+00:00', time.gmtime(tx.timestamp))}")
                            print(f" Podpis: {Fore.BLUE}{tx.signature}{Style.RESET_ALL}")
                            print("-" * 20)
                conn.close()
                if not tx_found:
                    print(f"{Fore.CYAN}Žádné transakce nebyly nalezeny.{Style.RESET_ALL}")
            
            elif choice == "14":
                show_p2p_log()
            
            elif choice == "15":
                tx_id = input("Zadejte TX ID transakce: ")
                tx, location = droid_chain.find_transaction_by_id(tx_id)
                if tx:
                    print(f"\n{Fore.YELLOW}--- Detaily transakce (TX ID: {tx.tx_id}) ---{Style.RESET_ALL}")
                    print(f" Stav: {Fore.GREEN}Nalezena v {location}{Style.RESET_ALL}")
                    print(f" Od: {tx.from_address}")
                    print(f" Komu: {tx.to_address}")
                    print(f" Èástka: {format(tx.amount / (10 ** DECIMALS), f'.{DECIMALS}f')} {TICKER}")
                    print(f" Poplatek: {format(tx.fee / (10 ** DECIMALS), f'.{DECIMALS}f')} {TICKER}")
                    print(f" Nonce: {tx.nonce}")
                    print(f" Èas: {time.strftime('%d.%m.%Y %H:%M:%S UTC+00:00', time.gmtime(tx.timestamp))}")
                    print(f" Veøejný klíè: {tx.public_key}")
                    print(f" Podpis: {tx.signature}")
                    print("-" * 20)
                else:
                    print(f"{Fore.RED}Transakce s TX ID '{tx_id}' nebyla nalezena.{Style.RESET_ALL}")
            
            elif choice == "16":
                total_supply = droid_chain.get_total_supply()
                max_supply = MAX_SUPPLY
                print(f"\n{Fore.YELLOW}--- Celková nabídka mincí ---{Style.RESET_ALL}")
                print(f" Celková nabídka: {Fore.CYAN}{format(total_supply / (10 ** DECIMALS), f'.{DECIMALS}f')} {TICKER}{Style.RESET_ALL}")
                print(f" Maximální nabídka: {Fore.CYAN}{format(max_supply / (10 ** DECIMALS), f'.{DECIMALS}f')} {TICKER}{Style.RESET_ALL}")
            
            elif choice == "17":
                p2p_node.connect_to_all_peers()
            
            elif choice == "18":
                peer_ip = input("Zadejte IP adresu uzlu k pøidání: ")
                peer_port = int(input("Zadejte port uzlu: "))
                new_peer = (peer_ip, peer_port)
                if new_peer not in p2p_node.peers:
                    p2p_node.peers.append(new_peer)
                    save_peers(p2p_node.peers)
                    print(f"{Fore.GREEN}Uzel {new_peer} byl pøidán do seznamu.{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}Uzel {new_peer} již v seznamu existuje.{Style.RESET_ALL}")
            
            elif choice == "19":
                if p2p_node.peers:
                    print(f"\n{Fore.YELLOW}--- Uložené uzly ---{Style.RESET_ALL}")
                    for i, peer in enumerate(p2p_node.peers, 1):
                        print(f"{i}. {peer[0]}:{peer[1]}")
                else:
                    print(f"{Fore.YELLOW}Žádné uzly nejsou uloženy.{Style.RESET_ALL}")
                    continue
                peer_ip = input("Zadejte IP adresu uzlu k odstranìní: ")
                try:
                    peer_port = int(input("Zadejte port uzlu k odstranìní: "))
                except ValueError:
                    print(f"{Fore.RED}Neplatný port.{Style.RESET_ALL}")
                    continue
                
                peer_to_remove = (peer_ip, peer_port)
                if peer_to_remove in p2p_node.peers:
                    p2p_node.peers.remove(peer_to_remove)
                    save_peers(p2p_node.peers)
                    print(f"{Fore.GREEN}Uzel {peer_to_remove} byl odstranìn.{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}Uzel {peer_to_remove} není v seznamu.{Style.RESET_ALL}")
            
            elif choice == "20":
                try:
                    block_index = int(input("Zadejte èíslo bloku: "))
                    if 0 <= block_index <= droid_chain.max_block_index:
                        block = droid_chain.get_block(block_index)
                        target_hex = hex(block.target)[2:]
                        
                        print(f"Blok #{block.index}")
                        print(f" Hash: {Fore.MAGENTA}{block.hash}{Style.RESET_ALL}")
                        print(f" Merkle root: {Fore.CYAN}{block.merkle_root}{Style.RESET_ALL}")
                        print(f" Cílový target (hex): {Fore.CYAN}{target_hex}{Style.RESET_ALL}")
                        print(f" Pøedchozí hash: {Fore.MAGENTA}{block.previous_hash}{Style.RESET_ALL}")
                        print(f" PoW Nonce: {Fore.CYAN}{block.nonce}{Style.RESET_ALL}")
                        print(f" Èas: {Fore.CYAN}{time.strftime('%d.%m.%Y %H:%M:%S UTC+00:00', time.gmtime(block.timestamp))}{Style.RESET_ALL}")
                        print(f" Velikost bloku: {Fore.CYAN}{block.get_size() / 1024:.2f} KB{Style.RESET_ALL}")
                        print(f" Poèet potvrzení: {Fore.CYAN}{droid_chain.get_confirmations(block.hash)}{Style.RESET_ALL}")
                        print(f" Poèet transakcí: {len(block.transactions)}")
                        if block.transactions:
                            print(f" {Fore.YELLOW}Transakce:{Style.RESET_ALL}")
                            
                            for tx in block.transactions:
                                print(f" - TX ID: {Fore.CYAN}{tx.tx_id}{Style.RESET_ALL}")
                                print(f"   Od: {tx.from_address}")
                                print(f"   Komu: {tx.to_address}")
                                print(f"   Èástka: {Fore.CYAN}{format(tx.amount / (10 ** DECIMALS), f'.{DECIMALS}f')} {TICKER}{Style.RESET_ALL}")
                                if tx.from_address != "COINBASE":
                                    print(f"   Poplatek: {format(tx.fee / (10 ** DECIMALS), f'.{DECIMALS}f')} {TICKER}")
                                    print(f"   TX Nonce: {Fore.MAGENTA}{tx.nonce}{Style.RESET_ALL}")
                                if tx.signature:
                                    print(f"   Podpis: {Fore.BLUE}{tx.signature}{Style.RESET_ALL}")
                                else:
                                    print(f"   Podpis: {Fore.RED}žádný (chybí){Style.RESET_ALL}")
                        print("=" * 40)
                    else:
                        print(f"{Fore.RED}Blok s èíslem {block_index} neexistuje.{Style.RESET_ALL}")
                except ValueError:
                    print(f"{Fore.RED}Neplatné èíslo bloku.{Style.RESET_ALL}")
            
            else:
                print(f"{Fore.RED}Neplatná volba. Zkuste to prosím znovu.{Style.RESET_ALL}")

        except Exception as e:
            print(f"{Fore.RED}Došlo k chybì: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()