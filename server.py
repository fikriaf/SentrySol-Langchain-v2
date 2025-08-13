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
    yield f"data: {json.dumps({'step': 7, 'status': 'Running SentrySol-Core analysis...', 'progress': 95})}\n\n"
    await asyncio.sleep(0.1)

    result = run_analysis(context)

    # Parse result jika berupa string JSON
    parsed_result = result
    if isinstance(result, str):
        try:
            # Log the full result for debugging
            print(f"Full LLM result length: {len(result)}")
            
            # Remove markdown formatting if present
            clean_result = result.strip()
            
            # Handle different markdown formats
            if "```json" in clean_result:
                # Extract content between ```json and ```
                start_marker = "```json"
                end_marker = "```"
                
                start_idx = clean_result.find(start_marker)
                if start_idx != -1:
                    # Skip past the marker and any newlines
                    start_idx = clean_result.find('\n', start_idx)
                    if start_idx == -1:
                        start_idx = clean_result.find(start_marker) + len(start_marker)
                    else:
                        start_idx += 1
                    
                    # Find the closing marker
                    end_idx = clean_result.rfind(end_marker)
                    if end_idx != -1 and end_idx > start_idx:
                        clean_result = clean_result[start_idx:end_idx].strip()
            
            elif clean_result.startswith("```") and clean_result.endswith("```"):
                # Remove generic code block markers
                lines = clean_result.split('\n')
                if len(lines) > 2:
                    clean_result = '\n'.join(lines[1:-1]).strip()
            
            # Clean up control characters but preserve JSON structure
            import re
            # Remove only problematic control characters, keep newlines and tabs for JSON
            clean_result = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', clean_result)
            
            # Find JSON boundaries more carefully
            if not clean_result.startswith('{'):
                start_brace = clean_result.find('{')
                if start_brace != -1:
                    clean_result = clean_result[start_brace:]
            
            # Enhanced bracket validation and fixing
            def fix_json_structure(json_str):
                """Try to fix common JSON structure issues"""
                lines = json_str.split('\n')
                fixed_lines = []
                
                for line in lines:
                    # Skip empty lines
                    if not line.strip():
                        continue
                    
                    # Add line to fixed structure
                    fixed_lines.append(line)
                
                # Rejoin and check structure
                fixed_json = '\n'.join(fixed_lines)
                
                # Count brackets
                open_braces = fixed_json.count('{')
                close_braces = fixed_json.count('}')
                open_brackets = fixed_json.count('[')
                close_brackets = fixed_json.count(']')
                
                # Fix missing closing braces
                if open_braces > close_braces:
                    fixed_json += '}' * (open_braces - close_braces)
                
                # Fix missing closing brackets  
                if open_brackets > close_brackets:
                    fixed_json += ']' * (open_brackets - close_brackets)
                
                return fixed_json
            
            # Try to fix the JSON structure
            clean_result = fix_json_structure(clean_result)
            
            print(f"Final cleaned result length: {len(clean_result)}")
            
            # Write the cleaned result to file for debugging
            with open("debug_json.txt", "w", encoding="utf-8") as f:
                f.write(clean_result)
            print("Wrote cleaned JSON to debug_json.txt for inspection")
            
            # Try to parse the cleaned result
            parsed_result = json.loads(clean_result)
            print("Successfully parsed JSON result")
            
        except (json.JSONDecodeError, IndexError, AttributeError) as e:
            print(f"Could not parse result as JSON: {e}")
            print(f"Error at position: {getattr(e, 'pos', 'unknown')}")
            print(f"Raw result length: {len(result)}")
            print(f"Cleaned result length: {len(clean_result) if 'clean_result' in locals() else 'N/A'}")
            
            # Show context around the error position if available
            if hasattr(e, 'pos') and 'clean_result' in locals():
                error_pos = e.pos
                start = max(0, error_pos - 50)
                end = min(len(clean_result), error_pos + 50)
                print(f"Context around error: {repr(clean_result[start:end])}")
            
            # Try more aggressive parsing methods
            alternative_result = None
            
            # Method 1: Fix common JSON syntax issues before parsing
            if 'clean_result' in locals():
                try:
                    import re
                    # Fix common issues
                    fixed_result = clean_result
                    
                    # Fix invalid number formats like "+2679600" -> "2679600"
                    fixed_result = re.sub(r'"delta":\s*\+(\d+)', r'"delta": \1', fixed_result)
                    fixed_result = re.sub(r'"delta":\s*-(\d+)', r'"delta": -\1', fixed_result)
                    
                    # Fix any other "+number" patterns that aren't valid JSON
                    fixed_result = re.sub(r':\s*\+(\d+)', r': \1', fixed_result)
                    
                    # Fix trailing commas in objects/arrays
                    fixed_result = re.sub(r',(\s*[}\]])', r'\1', fixed_result)
                    
                    # Fix missing quotes around unquoted keys
                    fixed_result = re.sub(r'(\w+):', r'"\1":', fixed_result)
                    
                    # Try to parse the fixed result
                    alternative_result = json.loads(fixed_result)
                    print(f"Successfully parsed with JSON fixes: {len(fixed_result)} chars")
                    
                except Exception as fix_error:
                    print(f"JSON fixing failed: {fix_error}")
            
            # Method 2: Try to find and parse the largest complete JSON object using regex
            if not alternative_result and '{' in result:
                try:
                    # More sophisticated regex to find complete JSON objects
                    import re
                    # This pattern tries to match balanced braces
                    json_pattern = r'\{(?:[^{}]|{[^{}]*})*\}'
                    matches = re.findall(json_pattern, result, re.DOTALL)
                    
                    # Try each match, starting with the longest
                    for match in sorted(matches, key=len, reverse=True):
                        try:
                            # Clean the match before parsing
                            clean_match = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', match)
                            # Fix common JSON issues
                            clean_match = re.sub(r':\s*\+(\d+)', r': \1', clean_match)
                            clean_match = re.sub(r',(\s*[}\]])', r'\1', clean_match)
                            
                            alternative_result = json.loads(clean_match)
                            print(f"Successfully parsed JSON match: {len(match)} chars")
                            break
                        except:
                            continue
                            
                except Exception as regex_error:
                    print(f"Regex parsing failed: {regex_error}")
            
            # Method 3: Try bracket counting approach with JSON fixes
            if not alternative_result and '{' in result and '}' in result:
                try:
                    first_brace = result.find('{')
                    brace_count = 0
                    end_pos = first_brace
                    
                    for i, char in enumerate(result[first_brace:], first_brace):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_pos = i + 1
                                break
                    
                    if brace_count == 0:
                        json_str = result[first_brace:end_pos]
                        # Clean and fix this extracted JSON
                        json_str = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', json_str)
                        # Fix number format issues
                        json_str = re.sub(r':\s*\+(\d+)', r': \1', json_str)
                        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                        
                        alternative_result = json.loads(json_str)
                        print("Successfully parsed using bracket counting method with fixes")
                except Exception as bracket_error:
                    print(f"Bracket counting failed: {bracket_error}")
            
            # Method 4: Try to extract just a simple structure if all else fails
            if not alternative_result:
                try:
                    # Look for key information in the text
                    threat_match = re.search(r'"threat_type":\s*"([^"]+)"', result)
                    risk_match = re.search(r'"risk_level":\s*"([^"]+)"', result) 
                    score_match = re.search(r'"risk_score":\s*(\d+(?:\.\d+)?)', result)
                    
                    if threat_match or risk_match or score_match:
                        alternative_result = {
                            "analysis_status": "partial_extraction",
                            "threat_analysis": {
                                "metadata": {
                                    "target_address": target_address,
                                    "analysis_timestamp": "2025-08-13 07:11:59",
                                    "extraction_method": "regex_fallback"
                                },
                                "potential_threats": [{
                                    "threat_type": threat_match.group(1) if threat_match else "Unknown",
                                    "risk_level": risk_match.group(1) if risk_match else "Unknown",
                                    "risk_score": float(score_match.group(1)) if score_match else 0.0
                                }]
                            }
                        }
                        print("Successfully extracted partial analysis using regex")
                except Exception as extract_error:
                    print(f"Regex extraction failed: {extract_error}")
            
            # Use the alternative result if it worked, otherwise create fallback
            if alternative_result:
                parsed_result = alternative_result
            else:
                # Create a comprehensive fallback response
                parsed_result = {
                    "analysis_status": "parsing_error",
                    "error_details": {
                        "error_message": str(e),
                        "error_position": getattr(e, 'pos', None),
                        "result_length": len(result),
                        "cleaned_length": len(clean_result) if 'clean_result' in locals() else None,
                        "error_context": clean_result[max(0, getattr(e, 'pos', 0) - 50):getattr(e, 'pos', 0) + 50] if hasattr(e, 'pos') and 'clean_result' in locals() else None
                    },
                    "partial_analysis": {
                        "note": "Analysis completed but JSON parsing failed",
                        "raw_excerpt": result[:500] + "..." if len(result) > 500 else result,
                        "possible_causes": [
                            "Invalid JSON syntax (e.g., +numbers, trailing commas)",
                            "Incomplete JSON response from AI model",
                            "Special characters in response",
                            "Truncated response due to length limits"
                        ]
                    }
                }

    try:
        # Safely extract data with better error handling
        def safe_get(obj, *keys, default=None):
            """Safely get nested dictionary values"""
            try:
                if obj is None:
                    return default
                for key in keys:
                    obj = obj[key]
                return obj
            except (KeyError, TypeError, AttributeError):
                return default

        # Final result with structured data
        final_data = {
            "step": 8,
            "status": "Analysis complete",
            "progress": 100,
            "analysis_result": parsed_result,  # Send parsed JSON object instead of string
            "detailed_data": {
                "wallet_info": {
                    "address": target_address,
                    "owner": safe_get(owner, "result", "value", "owner") if safe_get(owner, "result", "value", "owner") else target_address,
                    "risk_score": wallet_score,
                },
                "transaction_summary": {
                    "total_transactions": len(address_history.get("result", [])),
                    "recent_signatures": len(signatures.get("result", [])),
                    "balance_changes": safe_get(balance_changes, "result", "value"),
                },
                "token_analysis": {
                    "tokens_found": len(token_meta),
                    "token_metadata": token_meta[:5] if token_meta else [],
                    "nfts_found": len(nft_meta),
                    "nft_metadata": nft_meta[:3] if nft_meta else [],
                },
                "webhook_events": webhook_events if webhook_events else [],
            },
        }

        # Safe JSON serialization
        json_str = json.dumps(final_data, ensure_ascii=False, default=str, separators=(',', ':'))
        yield f"data: {json_str}\n\n"
        
    except Exception as final_error:
        print(f"Error creating final response: {final_error}")
        # Ultra-safe fallback
        fallback_data = {
            "step": 8,
            "status": "Analysis complete with errors",
            "progress": 100,
            "error": str(final_error),
            "analysis_result": parsed_result if parsed_result else "No result available"
        }
        yield f"data: {json.dumps(fallback_data, ensure_ascii=False, default=str)}\n\n"

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
    import os

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ["PORT"])  # Ambil langsung dari Railway
    )