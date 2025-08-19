
# SentrySol-Langchain Backend

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green?logo=fastapi)
![License](https://img.shields.io/github/license/fikriaf/SentrySol-Langchain-v2)
![Status](https://img.shields.io/badge/status-active-brightgreen)

SentrySol-Langchain is a professional backend service for Solana blockchain analysis, wallet risk scoring, and AI-powered chat, built with FastAPI. It provides robust RESTful endpoints for transaction analysis, token/NFT metadata, risk scoring, and interactive chat with an AI assistant specialized in Solana security and analytics.

---

## Features

- **Solana Address Analysis**: Analyze wallet history, risk score, token/NFT holdings, and more.
- **Streaming & Synchronous Analysis**: Real-time streaming (SSE) or synchronous analysis endpoints.
- **Transaction & Token Endpoints**: Fetch transaction details, token and NFT metadata, balance changes, and webhook events.
- **AI Chat**: Interact with an AI assistant for security, compliance, and educational queries.
- **Batch Analysis**: Analyze multiple addresses in one request.
- **CORS Enabled**: Ready for integration with web frontends.

---

## Requirements

- Python 3.10 or higher
- See `requirements.txt` for dependencies
- Solana (Helius), Metasleuth, and Mistral API keys (set in environment variables)

---

## Setup & Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/fikriaf/SentrySol-Langchain-v2.git
    cd SentrySol-Langchain-v2
    ```
2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
3. **Set environment variables**
    Create a `.env` file or set these variables in your environment:
    ```env
    HELIUS_API_KEY=your_helius_api_key
    METASLEUTH_API_KEY=your_metasleuth_api_key
    MISTRAL_API_KEY=your_mistral_api_key
    LLM_MODEL=your_model_name
    PORT=8000
    ```
4. **Run the server**
    ```bash
    python server.py
    # or
    uvicorn server:app --host 0.0.0.0 --port 8000
    ```

---

## API Endpoints & Payloads

### 1. Health Check
**GET** `/health`

**Response:**
```json
{
   "status": "healthy",
   "version": "1.0.0",
   "apis_configured": {
      "blockchain_api": true,
      "risk_analysis_api": true,
      "ai_model": true
   },
   "model": "mistral-xyz"
}
```

### 2. Streaming Address Analysis
**GET** `/analyze/{address}`

**Response:**
SSE (Server-Sent Events) streaming JSON per step. Example chunk:
```json
data: {"step": 1, "status": "Fetching address history...", "progress": 10}
```

### 3. Synchronous Address Analysis
**GET** `/analyze-sync/{address}`

**Response:**
```json
{
   "address": "...",
   "analysis_result": {...},
   "wallet_score": 80,
   "transaction_count": 12
}
```

### 4. Transaction History
**GET** `/address/{address}/transactions?limit=20`

**Response:**
```json
{
   "result": [ {"signature": "...", ...}, ... ]
}
```

### 5. Transaction Signatures
**GET** `/address/{address}/signatures?limit=10`

**Response:**
```json
{
   "result": [ {"signature": "..."}, ... ]
}
```

### 6. Balance Changes
**GET** `/address/{address}/balance-changes`

**Response:**
```json
{
   "address": "...",
   "balance_changes": {...}
}
```

### 7. Wallet Risk Score
**GET** `/address/{address}/risk-score`

**Response:**
```json
{
   "address": "...",
   "risk_score": 80
}
```

### 8. Resolve Address Name
**GET** `/address/{address}/resolve-name`

**Response:**
```json
{
   "address": "...",
   "name": "domain.sol"
}
```

### 9. Transaction Details
**GET** `/transaction/{signature}`

**Response:**
```json
{
   "result": { ...transaction details... }
}
```

### 10. Token Metadata
**GET** `/token/{mint}/metadata`

**Response:**
```json
{
   "result": { ...token metadata... }
}
```

### 11. NFT Metadata
**GET** `/nft/{mint}/metadata`

**Response:**
```json
{
   "result": { ...nft metadata... }
}
```

### 12. Webhook Events
**GET** `/webhook/events?addresses=addr1,addr2&limit=5`

**Response:**
```json
{
   "result": [ ...events... ]
}
```

### 13. Raw Data for Address
**GET** `/address/{address}/raw-data`

**Response:**
```json
{
   "address": "...",
   "owner": {...},
   "risk_score": 80,
   "transactions": {...},
   "signatures": {...},
   "balance_changes": {...},
   "tokens": [...],
   "nfts": [...],
   "summary": {
      "transaction_count": 12,
      "signature_count": 10,
      "token_count": 2,
      "nft_count": 1
   }
}
```

### 14. Batch Address Analysis
**POST** `/analyze/batch`

**Request Body:**
```json
[
   "address1...",
   "address2..."
]
```
**Response:**
```json
{
   "batch_results": {
      "address1...": {"risk_score": 80, "transaction_count": 5, "status": "success"},
      "address2...": {"error": "Invalid address format", "status": "failed"}
   }
}
```

### 15. AI Chat
**POST** `/chat-sentrysol`

**Request Body:**
```json
{
   "message": "Explain this transaction...",
   "system_prompt": "security_analysis",
   "temperature": 0.7,
   "max_tokens": 2000,
   "conversation_history": [ {"role": "user", "content": "..."} ]
}
```
**Response:**
```json
{
   "response": "...AI answer..."
}
```

### 16. AI Chat Streaming
**POST** `/chat-sentrysol-stream`

**Request Body:**
```json
{
   "message": "Explain this transaction...",
   "system_prompt": "security_analysis",
   "temperature": 0.7,
   "max_tokens": 2000
}
```
**Response:**
SSE streaming JSON chunks.

### 17. System Prompts
**GET** `/chat-sentrysol/system-prompts`

**Response:**
```json
{
   "available_prompts": {
      "default": "...",
      "security_analysis": "...",
      "transaction_analysis": "...",
      "educational": "...",
      "compliance": "..."
   },
   "usage": "Select a prompt key and use it in the system_prompt field of your chat request"
}
```

---

## Example Usage

**Analyze an address (streaming):**
```bash
curl http://localhost:8000/analyze/YourSolanaAddressHere
```

**Chat with SentrySol AI:**
```bash
curl -X POST http://localhost:8000/chat-sentrysol \
   -H "Content-Type: application/json" \
   -d '{"message": "Explain this transaction...", "system_prompt": "security_analysis"}'
```

---

## License

This project is licensed under the MIT License.

## Authors

- fikriaf

---
For more details, see the code in `server.py` and the `modules/` directory.
