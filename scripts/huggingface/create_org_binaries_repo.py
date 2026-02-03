#!/usr/bin/env python3
"""
Create llamatelemetry/binaries repository under the CORRECT organization
Run this AFTER getting admin access to llamatelemetry org
"""

from huggingface_hub import HfApi, create_repo
import os
import sys

HF_TOKEN = os.environ.get("HF_TOKEN", "")
REPO_ID = "llamatelemetry/binaries"  # Correct org name

def main():
    print("=" * 70)
    print("Creating llamatelemetry/binaries Repository")
    print("=" * 70)
    print()
    
    api = HfApi(token=HF_TOKEN)
    
    # Check if user has access
    print("🔍 Checking organization access...")
    try:
        user = api.whoami(token=HF_TOKEN)
        orgs = [org['name'] if isinstance(org, dict) else org for org in user.get('orgs', [])]
        
        if 'llamatelemetry' not in orgs:
            print("❌ ERROR: You don't have access to 'llamatelemetry' organization")
            print()
            print("Please follow these steps:")
            print("1. Go to https://huggingface.co/organizations/llamatelemetry/settings/members")
            print("2. Add yourself (waqasm86) as admin")
            print("3. Run this script again")
            print()
            sys.exit(1)
        
        print(f"✅ Access confirmed to llamatelemetry organization")
        
    except Exception as e:
        print(f"❌ Error checking access: {e}")
        sys.exit(1)
    
    # Create repository
    print(f"\n🔨 Creating repository: {REPO_ID}")
    try:
        create_repo(
            repo_id=REPO_ID,
            token=HF_TOKEN,
            repo_type="model",
            exist_ok=True,
            private=False,
        )
        print(f"✅ Repository created successfully!")
        print(f"🔗 URL: https://huggingface.co/{REPO_ID}")
        
    except Exception as e:
        print(f"❌ Failed to create repository: {e}")
        print()
        print("Try creating it manually:")
        print("1. Go to https://huggingface.co/new")
        print("2. Owner: llamatelemetry")
        print("3. Model name: binaries")
        print("4. License: MIT")
        print("5. Create model")
        sys.exit(1)
    
    print()
    print("=" * 70)
    print("✅ SUCCESS! Repository is ready")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Run: python3 scripts/huggingface/upload_to_llamatelemetry_org.py")
    print("2. Update bootstrap.py to use llamatelemetry/binaries")
    print("3. Commit and push changes")

if __name__ == "__main__":
    main()
