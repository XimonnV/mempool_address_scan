#!/usr/bin/env python3
"""
Bitcoin mempool address scanner - Python version
Looks up Bitcoin transactions from the mempool based on a given Bitcoin address
"""

import argparse
import configparser
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
from requests.auth import HTTPBasicAuth


@dataclass
class TransactionInput:
    txid: str
    vout: int


@dataclass
class TransactionOutput:
    value_sats: int
    script_pubkey: str
    address: str


@dataclass
class TransactionInfo:
    txid: str
    inputs: List[TransactionInput]
    outputs: List[TransactionOutput]
    incoming_sats: int
    total_out_sats: int
    receivers: Dict[str, int]  # address -> sats


class BitcoinRPC:
    """Bitcoin Core RPC client"""

    def __init__(self, rpc_user: str, rpc_password: str, rpc_port: int = 8332, host: str = "127.0.0.1"):
        self.url = f"http://{host}:{rpc_port}/"
        self.auth = HTTPBasicAuth(rpc_user, rpc_password)
        self.session = requests.Session()
        self.session.auth = self.auth

    def call(self, method: str, params: List = None) -> dict:
        """Make a single RPC call"""
        if params is None:
            params = []

        payload = {
            "jsonrpc": "1.0",
            "id": "1",
            "method": method,
            "params": params
        }

        try:
            response = self.session.post(self.url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()

            if result.get("error"):
                raise Exception(f"RPC error: {result['error']}")

            return result.get("result")
        except requests.exceptions.RequestException as e:
            raise Exception(f"RPC connection error: {e}")

    def batch_call(self, calls: List[Tuple[str, List]], chunk_size: int = 300, progress_bar: 'ProgressBar' = None) -> List:
        """Make batch RPC calls with chunking and optional progress tracking"""
        all_results = []
        total_processed = 0

        for i in range(0, len(calls), chunk_size):
            chunk = calls[i:i + chunk_size]
            batch_payload = []

            for idx, (method, params) in enumerate(chunk):
                batch_payload.append({
                    "jsonrpc": "1.0",
                    "id": idx,
                    "method": method,
                    "params": params or []
                })

            try:
                response = self.session.post(self.url, json=batch_payload, timeout=60)
                response.raise_for_status()
                results = response.json()

                # Handle both single result and batch results
                if not isinstance(results, list):
                    results = [results]

                # Sort by id and extract results
                sorted_results = sorted(results, key=lambda x: x.get("id", 0))
                chunk_results = [r.get("result") if not r.get("error") else None for r in sorted_results]
                all_results.extend(chunk_results)

            except requests.exceptions.RequestException as e:
                # Fill with None for failed chunk
                all_results.extend([None] * len(chunk))

            # Update progress if progress bar provided
            total_processed += len(chunk)
            if progress_bar:
                progress_bar.update(total_processed)

        return all_results


class ProgressBar:
    """Enhanced progress bar with context manager support"""

    def __init__(self, total: int, prefix: str = "Progress", width: int = 50):
        self.total = total
        self.prefix = prefix
        self.width = width
        self.current = 0
        self.completed = False

    def update(self, current: int):
        """Update progress to current value"""
        self.current = current
        if self.total == 0:
            print(f"\r{self.prefix}: 0%", end="", flush=True)
            return

        percent = min(100, int(current * 100 / self.total))
        filled = int(current * self.width / self.total)
        bar = "#" * filled + "-" * (self.width - filled)

        print(f"\r{self.prefix}: {percent:3d}% [{bar}]", end="", flush=True)

        if current >= self.total and not self.completed:
            print()  # New line when complete
            self.completed = True

    def increment(self, amount: int = 1):
        """Increment progress by amount"""
        self.update(self.current + amount)

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure completion"""
        if not self.completed:
            self.update(self.total)


def parse_duration(duration_str: str) -> int:
    """Parse duration string to seconds"""
    if not duration_str:
        return 0

    # Match pattern like 1h, 24h, 2d, 90m, 3600s
    match = re.match(r'^(\d+)([smhd])$', duration_str)
    if match:
        num, unit = match.groups()
        num = int(num)
        multipliers = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}
        return num * multipliers[unit]

    # Try parsing as plain number (seconds)
    if duration_str.isdigit():
        return int(duration_str)

    raise ValueError(f"Invalid duration format: {duration_str}")


def sat_to_btc(sats: int) -> str:
    """Convert satoshis to BTC string"""
    return f"{sats / 100_000_000:.8f}"


def load_bitcoin_config(conf_dir: str = None) -> Tuple[str, str, int]:
    """Load Bitcoin Core configuration"""
    if conf_dir is None:
        conf_dir = os.path.expanduser("~/.bitcoin")

    conf_path = Path(conf_dir) / "bitcoin.conf"
    if not conf_path.exists():
        raise FileNotFoundError(f"bitcoin.conf not found at {conf_path}")

    # Try to read directly from file to handle flat config files
    section = {}
    with open(conf_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                section[key.strip()] = value.strip()

    # If we didn't get anything, try configparser
    if not section:
        config = configparser.ConfigParser()
        config.read(conf_path)

        # Handle both section-based and flat config files
        if config.sections():
            section = dict(config.items(config.sections()[0]))
        else:
            section = dict(config.items('DEFAULT')) if config.has_section('DEFAULT') else {}

    rpc_user = section.get('rpcuser', '')
    rpc_password = section.get('rpcpassword', '')
    rpc_port = int(section.get('rpcport', 8332))

    if not rpc_user or not rpc_password:
        raise ValueError("RPC credentials not found in bitcoin.conf")

    return rpc_user, rpc_password, rpc_port


class MempoolScanner:
    """Main mempool scanner class"""

    def __init__(self, rpc: BitcoinRPC):
        self.rpc = rpc
        self.target_spk = ""
        self.target_address = ""

    def get_address_scriptpubkey(self, address: str) -> str:
        """Get scriptPubKey for address"""
        try:
            # Try getaddressinfo first (newer method)
            result = self.rpc.call("getaddressinfo", [address])
            return result.get("scriptPubKey", "")
        except:
            try:
                # Fallback to validateaddress
                result = self.rpc.call("validateaddress", [address])
                if result.get("isvalid"):
                    return result.get("scriptPubKey", "")
            except:
                pass

        raise ValueError(f"Invalid Bitcoin address: {address}")

    def get_mempool_txids(self, since_seconds: int = 0) -> List[str]:
        """Get mempool transaction IDs with optional time filtering"""
        print("Fetching mempool transaction IDs...")
        
        mempool_info = self.rpc.call("getrawmempool", [True])

        if not mempool_info:
            return []

        if since_seconds > 0:
            cutoff_time = int(time.time()) - since_seconds
            filtered_txids = [txid for txid, info in mempool_info.items()
                             if info.get("time", 0) >= cutoff_time]
            print(f"Found {len(filtered_txids)} transactions in the last {since_seconds} seconds")
            return filtered_txids

        print(f"Found {len(mempool_info)} total mempool transactions")
        return list(mempool_info.keys())

    def decode_transactions(self, txids: List[str]) -> Dict[str, TransactionInfo]:
        """Decode transactions in batches with progress tracking"""
        print("Decoding mempool transactions...")
        
        with ProgressBar(len(txids), "Decode mempool") as progress:
            # Batch getrawtransaction calls
            calls = [("getrawtransaction", [txid, True]) for txid in txids]
            raw_txs = self.rpc.batch_call(calls, progress_bar=progress)

        transactions = {}

        print("Processing decoded transactions...")
        with ProgressBar(len(txids), "Process decoded") as progress:
            for i, (txid, raw_tx) in enumerate(zip(txids, raw_txs)):
                progress.update(i + 1)

                if not raw_tx:
                    continue

                # Extract transaction info
                inputs = []
                for vin in raw_tx.get("vin", []):
                    if "txid" in vin and "vout" in vin:
                        inputs.append(TransactionInput(vin["txid"], vin["vout"]))

                outputs = []
                receivers = {}
                incoming_sats = 0
                total_out_sats = 0

                for vout in raw_tx.get("vout", []):
                    value_sats = int(vout.get("value", 0) * 100_000_000)
                    spk = vout.get("scriptPubKey", {})
                    spk_hex = spk.get("hex", "")
                    address = spk.get("address", f"({spk.get('type', 'unknown')}) {spk.get('asm', '')}")

                    outputs.append(TransactionOutput(value_sats, spk_hex, address))
                    total_out_sats += value_sats

                    # Track receivers
                    receivers[address] = receivers.get(address, 0) + value_sats

                    # Check if this output goes to our target address
                    if spk_hex == self.target_spk:
                        incoming_sats += value_sats

                transactions[txid] = TransactionInfo(
                    txid=txid,
                    inputs=inputs,
                    outputs=outputs,
                    incoming_sats=incoming_sats,
                    total_out_sats=total_out_sats,
                    receivers=receivers
                )

        return transactions

    def fetch_parent_transactions(self, transactions: Dict[str, TransactionInfo]) -> Dict[str, Dict[int, TransactionOutput]]:
        """Fetch parent transaction outputs with progress tracking"""
        # Collect all unique parent txids
        parent_txids = set()
        for tx_info in transactions.values():
            for inp in tx_info.inputs:
                parent_txids.add(inp.txid)

        parent_txids = list(parent_txids)
        if not parent_txids:
            return {}

        print("Fetching parent transactions...")
        
        with ProgressBar(len(parent_txids), "Fetch parents") as progress:
            # Batch fetch parent transactions
            calls = [("getrawtransaction", [txid, True]) for txid in parent_txids]
            parent_raws = self.rpc.batch_call(calls, progress_bar=progress)

        parent_outputs = {}

        print("Processing parent transactions...")
        with ProgressBar(len(parent_txids), "Process parents") as progress:
            for i, (parent_txid, parent_raw) in enumerate(zip(parent_txids, parent_raws)):
                progress.update(i + 1)

                if not parent_raw:
                    continue

                outputs = {}
                for vout in parent_raw.get("vout", []):
                    n = vout.get("n")
                    if n is not None:
                        value_sats = int(vout.get("value", 0) * 100_000_000)
                        spk = vout.get("scriptPubKey", {})
                        spk_hex = spk.get("hex", "")
                        address = spk.get("address", f"({spk.get('type', 'unknown')}) {spk.get('asm', '')}")

                        outputs[n] = TransactionOutput(value_sats, spk_hex, address)

                parent_outputs[parent_txid] = outputs

        # Try to fetch missing parents from mempool
        missing_parents = []
        for tx_info in transactions.values():
            for inp in tx_info.inputs:
                if inp.txid not in parent_outputs:
                    missing_parents.append(inp.txid)

        if missing_parents:
            missing_parents = list(set(missing_parents))
            print(f"Fetching {len(missing_parents)} missing parents from mempool...")

            with ProgressBar(len(missing_parents), "Missing parents") as progress:
                calls = [("getrawtransaction", [txid, True]) for txid in missing_parents]
                missing_raws = self.rpc.batch_call(calls, progress_bar=progress)

            print("Processing missing parent transactions...")
            with ProgressBar(len(missing_parents), "Process missing") as progress:
                for i, (parent_txid, parent_raw) in enumerate(zip(missing_parents, missing_raws)):
                    progress.update(i + 1)
                    
                    if not parent_raw:
                        continue

                    outputs = {}
                    for vout in parent_raw.get("vout", []):
                        n = vout.get("n")
                        if n is not None:
                            value_sats = int(vout.get("value", 0) * 100_000_000)
                            spk = vout.get("scriptPubKey", {})
                            spk_hex = spk.get("hex", "")
                            address = spk.get("address", f"({spk.get('type', 'unknown')}) {spk.get('asm', '')}")

                            outputs[n] = TransactionOutput(value_sats, spk_hex, address)

                    parent_outputs[parent_txid] = outputs

        return parent_outputs

    def process_transactions(self, transactions: Dict[str, TransactionInfo],
                           parent_outputs: Dict[str, Dict[int, TransactionOutput]]) -> List[dict]:
        """Process transactions and find relevant ones with progress tracking"""
        print("Processing transactions...")
        
        relevant_txs = []

        with ProgressBar(len(transactions), "Analyze txs") as progress:
            for i, (txid, tx_info) in enumerate(transactions.items()):
                progress.update(i + 1)

                # Calculate input details
                total_in_sats = 0
                total_from_our_spk_sats = 0
                senders = {}

                for inp in tx_info.inputs:
                    parent_tx_outputs = parent_outputs.get(inp.txid, {})
                    parent_output = parent_tx_outputs.get(inp.vout)

                    if parent_output:
                        total_in_sats += parent_output.value_sats
                        senders[parent_output.address] = senders.get(parent_output.address, 0) + parent_output.value_sats

                        if parent_output.script_pubkey == self.target_spk:
                            total_from_our_spk_sats += parent_output.value_sats

                # Check if transaction is relevant
                is_incoming = tx_info.incoming_sats > 0
                is_outgoing = total_from_our_spk_sats > 0

                if is_incoming or is_outgoing:
                    fee_sats = total_in_sats - tx_info.total_out_sats if total_in_sats > 0 else None

                    relevant_txs.append({
                        "txid": txid,
                        "is_incoming": is_incoming,
                        "is_outgoing": is_outgoing,
                        "incoming_sats": tx_info.incoming_sats,
                        "outgoing_sats": total_from_our_spk_sats,
                        "fee_sats": fee_sats,
                        "senders": senders,
                        "receivers": tx_info.receivers,
                        "total_in_sats": total_in_sats
                    })

        return relevant_txs

    def scan_address(self, address: str, since_seconds: int = 0) -> List[dict]:
        """Main scanning function"""
        self.target_address = address
        self.target_spk = self.get_address_scriptpubkey(address)

        print(f"Scanning mempool for transactions related to address: {address}")
        print(f"ScriptPubKey: {self.target_spk}")
        if since_seconds > 0:
            print(f"Time filter: last {since_seconds} seconds")
        else:
            print("Time filter: none")
        print()

        # Get mempool transaction IDs
        txids = self.get_mempool_txids(since_seconds)
        if not txids:
            print("No mempool transactions found.")
            return []

        # Decode transactions
        transactions = self.decode_transactions(txids)

        # Fetch parent transactions
        parent_outputs = self.fetch_parent_transactions(transactions)

        # Process and find relevant transactions
        relevant_txs = self.process_transactions(transactions, parent_outputs)

        return relevant_txs


def print_results(relevant_txs: List[dict], target_address: str):
    """Print scan results"""
    if not relevant_txs:
        print(f"No related mempool transactions found for {target_address}")
        return

    for tx in relevant_txs:
        print(f"txid: {tx['txid']}")

        if tx['fee_sats'] is not None:
            print(f"  fee: {sat_to_btc(tx['fee_sats'])} BTC")
        else:
            print("  fee: (unknown)")

        if tx['is_incoming']:
            print("  incoming:")
            print(f"    amount to {target_address}: {sat_to_btc(tx['incoming_sats'])} BTC")
            if tx['senders']:
                print("    from addresses:")
                for addr, sats in tx['senders'].items():
                    print(f"      {addr}: {sat_to_btc(sats)} BTC")
            else:
                print("    from addresses: (unknown)")

        if tx['is_outgoing']:
            print("  outgoing:")
            print(f"    amount sent from {target_address}: {sat_to_btc(tx['outgoing_sats'])} BTC")
            if tx['receivers']:
                print("    receiving addresses:")
                for addr, sats in tx['receivers'].items():
                    print(f"      {addr}: {sat_to_btc(sats)} BTC")
            else:
                print("    receiving addresses: (none)")

        print()


def main():
    parser = argparse.ArgumentParser(
        description="Scan Bitcoin mempool for transactions related to an address",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh
  %(prog)s --since 1h bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh
  %(prog)s --since 24h bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh

Duration formats: 1h, 24h, 2d, 90m, 3600s
        """
    )

    parser.add_argument("address", help="Bitcoin address to scan for")
    parser.add_argument("--since", help="Only include transactions first seen within DURATION (e.g., 1h, 24h, 2d)")
    parser.add_argument("--conf-dir", help="Bitcoin configuration directory", default=None)

    args = parser.parse_args()

    try:
        # Parse duration
        since_seconds = 0
        if args.since:
            since_seconds = parse_duration(args.since)

        # Load Bitcoin configuration
        rpc_user, rpc_password, rpc_port = load_bitcoin_config(args.conf_dir)

        # Create RPC client
        rpc = BitcoinRPC(rpc_user, rpc_password, rpc_port)

        # Create scanner and scan
        scanner = MempoolScanner(rpc)
        relevant_txs = scanner.scan_address(args.address, since_seconds)

        # Print results
        print_results(relevant_txs, args.address)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
