# How to Add Yourself as Admin to llamatelemetry Organization

## Problem
You're currently a member of `llamatelemtery` (typo) but not `llamatelemetry` (correct org).

## Solution: Add Yourself as Admin

### Step 1: Identify Who Created the Organization

The `llamatelemetry` organization was created by someone. You need to:

1. Go to https://huggingface.co/llamatelemetry
2. Look for "Team members" or organization info
3. Identify the organization owner

### Step 2: Add Admin Access

**If YOU created the organization with a different account:**

1. Log into HuggingFace with the account that created `llamatelemetry`
2. Go to https://huggingface.co/organizations/llamatelemetry/settings/members
3. Click "Add member"
4. Enter username: `waqasm86`
5. Select role: **Admin** or **Write**
6. Send invitation

Then:
7. Log in as `waqasm86`
8. Go to your notifications/invitations
9. Accept the invitation

**If someone ELSE created the organization:**

Contact them and ask them to add `waqasm86` as admin.

### Step 3: Verify Access

Run this to confirm:

```bash
python3 -c "
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get('HF_TOKEN'))
user = api.whoami()
orgs = [org['name'] if isinstance(org, dict) else org for org in user.get('orgs', [])]
print(f'Your orgs: {orgs}')
print(f'llamatelemetry access: {\"llamatelemetry\" in orgs}')
"
```

Expected output:
```
Your orgs: ['discord-community', 'llamatelemetry']
llamatelemetry access: True
```

### Step 4: Create Repository and Upload

Once you have access:

```bash
cd /media/waqasm86/External1/Project-Nvidia-Office/llamatelemetry

# Create the repository
python3 scripts/huggingface/create_org_binaries_repo.py

# Upload binaries
python3 scripts/huggingface/upload_to_llamatelemetry_org.py
```

## Alternative: Web Interface Method

If you have admin access, you can also do everything via web:

### Create Repository
1. Go to https://huggingface.co/new
2. Owner: **llamatelemetry** (select from dropdown)
3. Model name: **binaries**
4. License: **MIT**
5. Click "Create model"

### Upload Files
1. Go to https://huggingface.co/llamatelemetry/binaries
2. Click "Files and versions"
3. Click "Add file" → "Upload files"
4. Create folder: `v0.1.0/`
5. Upload both files:
   - `llamatelemetry-v0.1.0-cuda12-kaggle-t4x2.tar.gz`
   - `llamatelemetry-v0.1.0-cuda12-kaggle-t4x2.tar.gz.sha256`

## Troubleshooting

**If you can't find who created the organization:**

Check if you created it yourself with your current account:
1. Go to https://huggingface.co/settings/organizations
2. Look for `llamatelemetry` in your organizations
3. If it's there, you can manage it directly

**If the organization was created by mistake:**

You can create a new organization with the correct name, but HuggingFace org names must be unique, so `llamatelemetry` must already exist.

## Summary

You need to get admin/write access to the **llamatelemetry** organization (not the typo one). Once you have it, run the provided scripts to upload binaries.
