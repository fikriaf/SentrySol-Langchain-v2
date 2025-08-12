from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
import os
from dotenv import load_dotenv
from modules.helius_api import (
    fetch_transaction,
    fetch_address_history,
    fetch_token_metadata,
    fetch_nft_metadata,
    fetch_balance_changes,
    resolve_owner,
    fetch_webhook_events,
    get_signatures_for_address,
)
from modules.metasleuth_api import fetch_wallet_score
from modules.preprocess import aggregate_context
from modules.analysis_chain import run_analysis

# Load ENV
load_dotenv()

app = FastAPI(title="Solana Analysis API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY")
METASLEUTH_API_KEY = os.getenv("METASLEUTH_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")


async def stream_analysis(target_address: str):
    """Generator untuk streaming analisis bertahap"""

    # Step 1: Fetch address history
    yield f"data: {json.dumps({'step': 1, 'status': 'Fetching address history...', 'progress': 10})}\n\n"
    await asyncio.sleep(0.1)

    address_history = fetch_address_history(target_address, limit=20, enriched=True)
    yield f"data: {json.dumps({'step': 1, 'status': 'Address history fetched', 'progress': 15, 'data': {'transactions_count': len(address_history.get('result', []))}})}\n\n"

    # Step 2: Get signatures
    yield f"data: {json.dumps({'step': 2, 'status': 'Getting transaction signatures...', 'progress': 25})}\n\n"
    await asyncio.sleep(0.1)

    signatures = get_signatures_for_address(target_address, limit=10)
    yield f"data: {json.dumps({'step': 2, 'status': 'Signatures retrieved', 'progress': 35, 'data': {'signatures_count': len(signatures.get('result', []))}})}\n\n"

    # Step 3: Fetch token metadata
    yield f"data: {json.dumps({'step': 3, 'status': 'Analyzing token transfers...', 'progress': 45})}\n\n"
    await asyncio.sleep(0.1)

    token_meta = []
    nft_meta = []

    if address_history.get("result") and isinstance(address_history["result"], list):
        for tx in address_history["result"]:
            if isinstance(tx, dict) and tx.get("tokenTransfers"):
                for token in tx.get("tokenTransfers", []):
                    mint = token.get("mint")
                    if mint:
                        # Fetch token metadata
                        token_metadata = fetch_token_metadata(mint)
                        token_meta.append(token_metadata)

                        # Check if it's an NFT and fetch NFT metadata
                        if token_metadata and token_metadata.get("decimals") == 0:
                            nft_metadata = fetch_nft_metadata(mint)
                            if nft_metadata:
                                nft_meta.append(nft_metadata)

    yield f"data: {json.dumps({'step': 3, 'status': 'Token and NFT metadata collected', 'progress': 55, 'data': {'tokens_analyzed': len(token_meta), 'nfts_found': len(nft_meta)}})}\n\n"

    # Step 4: Get wallet score
    yield f"data: {json.dumps({'step': 4, 'status': 'Calculating wallet risk score...', 'progress': 65})}\n\n"
    await asyncio.sleep(0.1)

    wallet_score = fetch_wallet_score(target_address)
    yield f"data: {json.dumps({'step': 4, 'status': 'Wallet score calculated', 'progress': 75})}\n\n"

    # Step 5: Additional data
    yield f"data: {json.dumps({'step': 5, 'status': 'Gathering additional data...', 'progress': 80})}\n\n"
    await asyncio.sleep(0.1)

    tx_details = (
        fetch_transaction(signatures["result"][0]["signature"])
        if signatures.get("result")
        else {}
    )
    balance_changes = fetch_balance_changes(target_address)
    owner = resolve_owner(target_address)
    webhook_events = fetch_webhook_events([target_address], limit=5)

    yield f"data: {json.dumps({'step': 5, 'status': 'Additional data gathered', 'progress': 85, 'data': {'balance_changes_count': len(balance_changes) if balance_changes else 0}})}\n\n"

    # Step 6: Aggregate context
    yield f"data: {json.dumps({'step': 6, 'status': 'Aggregating context for analysis...', 'progress': 90})}\n\n"
    await asyncio.sleep(0.1)

    context = aggregate_context(
        helius_txs=[tx_details]
        + (
            address_history.get("result", [])
            if isinstance(address_history.get("result"), list)
            else []
        ),
        metasleuth_score=wallet_score,
        target_address=target_address,
        extra_notes="API streaming analysis with NFT data",
    )

    # Step 7: Run LLM analysis
    yield f"data: {json.dumps({'step': 7, 'status': 'Running AI analysis...', 'progress': 95})}\n\n"
    await asyncio.sleep(0.1)

    result = run_analysis(context)

    # Parse result jika berupa string JSON
    parsed_result = result
    if isinstance(result, str):
        try:
            # Coba parse jika result adalah string JSON
            if result.strip().startswith("{") or result.strip().startswith("```"):
                # Remove markdown formatting if present
                clean_result = result
                if "```json\n" in result:
                    clean_result = result.split("```json\n")[1].split("\n```")[0]
                elif "```\n" in result:
                    clean_result = result.split("```\n")[1].split("\n```")[0]

                parsed_result = json.loads(clean_result)
        except (json.JSONDecodeError, IndexError) as e:
            print(f"Could not parse result as JSON: {e}")
            # Keep original result as string
            parsed_result = result

    # Final result with structured data
    final_data = {
        "step": 8,
        "status": "Analysis complete",
        "progress": 100,
        "analysis_result": parsed_result,  # Send parsed JSON object instead of string
        "detailed_data": {
            "wallet_info": {
                "address": target_address,
                "owner": owner["result"]["value"]["owner"] if owner["result"]["value"] else None,
                "risk_score": wallet_score,
            },
            "transaction_summary": {
                "total_transactions": len(address_history.get("result", [])),
                "recent_signatures": len(signatures.get("result", [])),
                "balance_changes": balance_changes["result"]["value"],
            },
            "token_analysis": {
                "tokens_found": len(token_meta),
                "token_metadata": token_meta[:5] if token_meta else [],
                "nfts_found": len(nft_meta),
                "nft_metadata": nft_meta[:3] if nft_meta else [],
            },
            "webhook_events": webhook_events,
        },
    }

    yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"
    yield f"data: [DONE]\n\n"


@app.get("/")
async def root():
    return {"message": "Solana Analysis API is running"}


@app.get("/analyze/{address}")
async def analyze_address_stream(address: str):
    """Stream analisis address secara bertahap"""

    # Validasi address format (basic)
    if len(address) < 32 or len(address) > 44:
        return {"error": "Invalid Solana address format"}

    return StreamingResponse(
        stream_analysis(address),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        },
    )


@app.get("/analyze-sync/{address}")
async def analyze_address_sync(address: str):
    """Analisis synchronous (non-streaming) untuk testing"""

    if len(address) < 32 or len(address) > 44:
        return {"error": "Invalid Solana address format"}

    # Run semua analisis sekaligus
    address_history = fetch_address_history(address, limit=20, enriched=True)
    signatures = get_signatures_for_address(address, limit=10)
    wallet_score = fetch_wallet_score(address)

    context = aggregate_context(
        helius_txs=address_history.get("result", []),
        metasleuth_score=wallet_score,
        target_address=address,
        extra_notes="Sync analysis",
    )

    result = run_analysis(context)

    return {
        "address": address,
        "analysis_result": result,
        "wallet_score": wallet_score,
        "transaction_count": len(address_history.get("result", [])),
    }


@app.get("/health")
async def health_check():
    """Health check endpoint dengan status sistem"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "apis_configured": {
            "blockchain_api": bool(HELIUS_API_KEY),
            "risk_analysis_api": bool(METASLEUTH_API_KEY),
            "ai_model": bool(MISTRAL_API_KEY),
        },
        "model": LLM_MODEL,
    }


@app.get("/address/{address}/transactions")
async def get_address_transactions(address: str, limit: int = 20):
    """Get transaction history for address"""
    if len(address) < 32 or len(address) > 44:
        return {"error": "Invalid Solana address format"}

    result = fetch_address_history(address, limit=limit, enriched=True)
    return result


@app.get("/address/{address}/signatures")
async def get_address_signatures(address: str, limit: int = 10):
    """Get signatures for address"""
    if len(address) < 32 or len(address) > 44:
        return {"error": "Invalid Solana address format"}

    result = get_signatures_for_address(address, limit=limit)
    return result


@app.get("/address/{address}/balance-changes")
async def get_balance_changes(address: str):
    """Get balance changes for address"""
    if len(address) < 32 or len(address) > 44:
        return {"error": "Invalid Solana address format"}

    result = fetch_balance_changes(address)
    return {"address": address, "balance_changes": result}


@app.get("/address/{address}/risk-score")
async def get_wallet_risk_score(address: str):
    """Get wallet risk score from risk analysis service"""
    if len(address) < 32 or len(address) > 44:
        return {"error": "Invalid Solana address format"}

    score = fetch_wallet_score(address)
    return {"address": address, "risk_score": score}


@app.get("/address/{address}/resolve-name")
async def resolve_wallet_name(address: str):
    """Resolve address to domain name"""
    if len(address) < 32 or len(address) > 44:
        return {"error": "Invalid Solana address format"}

    name = resolve_owner(address)
    return {"address": address, "name": name}


@app.get("/transaction/{signature}")
async def get_transaction_details(signature: str):
    """Get detailed transaction information"""
    result = fetch_transaction(signature)
    return result


@app.get("/token/{mint}/metadata")
async def get_token_metadata(mint: str):
    """Get token metadata"""
    result = fetch_token_metadata(mint)
    return result


@app.get("/nft/{mint}/metadata")
async def get_nft_metadata(mint: str):
    """Get NFT metadata"""
    result = fetch_nft_metadata(mint)
    return result


@app.get("/webhook/events")
async def get_webhook_events(addresses: str, limit: int = 5):
    """Get webhook events for addresses (comma-separated)"""
    address_list = [addr.strip() for addr in addresses.split(",")]
    result = fetch_webhook_events(address_list, limit=limit)
    return result


@app.get("/address/{address}/raw-data")
async def get_raw_address_data(address: str):
    """Get all raw data for address without AI analysis"""
    if len(address) < 32 or len(address) > 44:
        return {"error": "Invalid Solana address format"}

    # Fetch all data
    address_history = fetch_address_history(address, limit=20, enriched=True)
    signatures = get_signatures_for_address(address, limit=10)
    wallet_score = fetch_wallet_score(address)
    balance_changes = fetch_balance_changes(address)
    owner = resolve_owner(address)

    # Get token/NFT data
    token_meta = []
    nft_meta = []

    if address_history.get("result") and isinstance(address_history["result"], list):
        for tx in address_history["result"][:5]:  # Limit to first 5 transactions
            if isinstance(tx, dict) and tx.get("tokenTransfers"):
                for token in tx.get("tokenTransfers", []):
                    mint = token.get("mint")
                    if mint:
                        token_metadata = fetch_token_metadata(mint)
                        token_meta.append(token_metadata)

                        if token_metadata and token_metadata.get("decimals") == 0:
                            nft_metadata = fetch_nft_metadata(mint)
                            if nft_metadata:
                                nft_meta.append(nft_metadata)

    return {
        "address": address,
        "owner": owner,
        "risk_score": wallet_score,
        "transactions": address_history,
        "signatures": signatures,
        "balance_changes": balance_changes,
        "tokens": token_meta,
        "nfts": nft_meta,
        "summary": {
            "transaction_count": len(address_history.get("result", [])),
            "signature_count": len(signatures.get("result", [])),
            "token_count": len(token_meta),
            "nft_count": len(nft_meta),
        },
    }


@app.post("/analyze/batch")
async def analyze_batch_addresses(addresses: list[str], limit: int = 5):
    """Analyze multiple addresses (max 5 for performance)"""
    if len(addresses) > limit:
        return {"error": f"Maximum {limit} addresses allowed per batch"}

    results = {}

    for address in addresses:
        if len(address) < 32 or len(address) > 44:
            results[address] = {"error": "Invalid address format"}
            continue

        try:
            # Basic analysis for each address
            wallet_score = fetch_wallet_score(address)
            address_history = fetch_address_history(address, limit=5, enriched=True)

            results[address] = {
                "risk_score": wallet_score,
                "transaction_count": len(address_history.get("result", [])),
                "status": "success",
            }
        except Exception as e:
            results[address] = {"error": str(e), "status": "failed"}

    return {"batch_results": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
