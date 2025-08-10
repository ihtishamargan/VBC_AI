"""Quick script to examine VBC contract structure."""
import os
from langchain_community.document_loaders import PyPDFLoader

def examine_contract(contract_path):
    """Examine a single contract PDF."""
    try:
        print(f"\n=== Examining: {os.path.basename(contract_path)} ===")
        loader = PyPDFLoader(contract_path)
        pages = loader.load()
        
        print(f"Pages: {len(pages)}")
        print(f"First 800 characters:")
        print("-" * 50)
        print(pages[0].page_content[:800])
        print("-" * 50)
        
        return True
    except Exception as e:
        print(f"Error loading {contract_path}: {e}")
        return False

# Examine first 3 contracts
contract_dir = "./backend/data/Contract-1"
contracts_to_check = ["Contract-1.pdf", "Contract-2.pdf", "Contract-3.pdf"]

for contract_file in contracts_to_check:
    contract_path = os.path.join(contract_dir, contract_file)
    if os.path.exists(contract_path):
        examine_contract(contract_path)
        print("\n" + "="*80)
    else:
        print(f"File not found: {contract_path}")
