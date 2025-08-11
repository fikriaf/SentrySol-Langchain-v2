import os
from dotenv import load_dotenv
from modules.helius_api import (
    fetch_transaction,
    fetch_address_history,
    fetch_token_metadata,
    fetch_nft_metadata,
    fetch_balance_changes,
    resolve_address_name,
    fetch_webhook_events,
    get_signatures_for_address,
)
from modules.metasleuth_api import fetch_wallet_score
from modules.preprocess import aggregate_context
from modules.analysis_chain import run_analysis

# Load ENV
load_dotenv()

# Config
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY")
METASLEUTH_API_KEY = os.getenv("METASLEUTH_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")

if not HELIUS_API_KEY:
    raise RuntimeError("HELIUS_API_KEY not found in .env")
if not METASLEUTH_API_KEY:
    raise RuntimeError("METASLEUTH_API_KEY not found in .env")
if not MISTRAL_API_KEY:
    raise RuntimeError("MISTRAL_API_KEY not found in .env")


def main():
    # Gunakan address/signature yang valid, bukan placeholder
    target_address = (
        "2YcwVbKx9L25Jpaj2vfWSXD5UKugZumWjzEe6suBUJi2"  # Contoh address Solana
    )

    address_history = fetch_address_history(target_address, limit=20, enriched=True)
    signatures = get_signatures_for_address(target_address, limit=10)
    token_meta = []

    # Handle RPC response format - check if result exists and is a list
    if address_history.get("result") and isinstance(address_history["result"], list):
        for tx in address_history["result"]:
            if isinstance(tx, dict) and tx.get("tokenTransfers"):
                for token in tx.get("tokenTransfers", []):
                    mint = token.get("mint")
                    if mint:
                        token_meta.append(fetch_token_metadata(mint))

    # Contoh pemanggilan fungsi baru
    tx_details = fetch_transaction(signatures["result"][0]["signature"])
    nft_meta = fetch_nft_metadata("ExampleMintAddressHere")
    balance_changes = fetch_balance_changes(target_address)
    address_name = resolve_address_name(target_address)
    webhook_events = fetch_webhook_events([target_address], limit=10)

    wallet_score = fetch_wallet_score(target_address)

    context = aggregate_context(
        helius_txs=[tx_details]
        + (
            address_history.get("result", [])
            if isinstance(address_history.get("result"), list)
            else []
        ),
        metasleuth_score=wallet_score,
        target_address=target_address,
        extra_notes="Initial scan",
    )

    result = run_analysis(context)

    print("\n=== LLM Analysis Result ===")
    print(result)

    # Print contoh hasil fungsi baru
    print("\nNFT Metadata:", nft_meta)
    print("\nBalance Changes:", balance_changes)
    print("\nAddress Name:", address_name)
    print("\nWebhook Events:", webhook_events)
    print("\nSignatures for address:", signatures)

    print("\nWallet Score:", wallet_score)

if __name__ == "__main__":
    main()
