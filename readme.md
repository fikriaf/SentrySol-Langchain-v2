run on terminal:
python -m venv .venv

--------------------------

On macOS/Linux: 
source .venv/bin/activate

On Windows (PowerShell): 
.venv\Scripts\Activate.ps1

--------------------------

Create .env :

MISTRAL_API_KEY=YOUR_API_KEY
MISTRAL_MODEL=ft:mistral-medium-latest:b319469f:20250807:b80c0dce

HELIUS_API_KEY=YOUR_API_KEY

CHAINABUSE_API_KEY=YOUR_API_KEY

METASLEUTH_API_KEY=YOUR_API_KEY

--------------------------

pip install -r requirements.txt

--------------------------

Start the server:
uvicorn server:app --reload

----------------------------

then run HTML file to browser.