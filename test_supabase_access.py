# test_supabase_simple.py
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_KEY')

print(f"URL: {url}")
print(f"Key exists: {bool(key)}")

client = create_client(url, key)

# Check if bucket exists
print("\nChecking bucket...")
try:
    buckets = client.storage.list_buckets()
    print("Buckets found:")
    for b in buckets:
        print(f"  - {b.name} (id: {b.id})")
except Exception as e:
    print(f"Error: {e}")

# Try to list files
print("\nListing files in flood-data...")
try:
    # Try different methods
    files = client.storage.from_('flood-data').list('')
    print("Files:", files)
except Exception as e:
    print(f"Error listing: {e}")

# Check if file exists with a HEAD request
print("\nChecking if files exist via URL...")
import httpx
test_files = ['flood_depth_model.pkl', 'gampaha_divisions.shp']
for file in test_files:
    test_url = f"{url}/storage/v1/object/public/flood-data/{file}"
    print(f"\nTesting: {test_url}")
    try:
        r = httpx.head(test_url)
        print(f"  Status: {r.status_code}")
        print(f"  Headers: {dict(r.headers)}")
    except Exception as e:
        print(f"  Error: {e}")