#!/usr/bin/env python3
"""Upload a dev.to article markdown file as a DRAFT (or published) via the Forem API.

Reads the API key from the DEVTO_API_KEY env var so the key never lives in a shell command / file.
  DEVTO_API_KEY=xxx python3 upload_devto.py <article.md>
  DEVTO_API_KEY=xxx DEVTO_PUBLISH=1 python3 upload_devto.py <article.md>   # publish live instead of draft

Parses the dev.to front matter (title / description / tags), strips it from the body, and posts the rest
as body_markdown with explicit fields. Prints the resulting article id + URL.
"""
import os, sys, json, re, urllib.request, urllib.error

path = sys.argv[1] if len(sys.argv) > 1 else "dev-to-gemma4-26b-single-inf2.md"
key = os.environ.get("DEVTO_API_KEY")
if not key and os.environ.get("DEVTO_KEY_S3"):
    # fallback: read the key from a temp S3 object (staged by the user via `!`), so the key never
    # appears in a shell command. Format: s3://bucket/key
    import boto3, urllib.parse
    u = urllib.parse.urlparse(os.environ["DEVTO_KEY_S3"])
    key = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-west-2")) \
        .get_object(Bucket=u.netloc, Key=u.path.lstrip("/"))["Body"].read().decode().strip()
if not key:
    sys.exit("set DEVTO_API_KEY (or DEVTO_KEY_S3=s3://bucket/key) in the environment")
publish = os.environ.get("DEVTO_PUBLISH", "0") in ("1", "true", "True", "yes")

raw = open(path, encoding="utf-8").read()
m = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.S)
if not m:
    sys.exit("no front matter found in " + path)
fm_text, body = m.group(1), m.group(2).lstrip("\n")

fm = {}
for line in fm_text.splitlines():
    line = line.strip()
    if not line or line.startswith("#"):          # skip blank + commented (e.g. '# series:') lines
        continue
    if ":" not in line:
        continue
    k, v = line.split(":", 1)
    fm[k.strip()] = v.strip().strip('"').strip("'")

tags = [t.strip() for t in fm.get("tags", "").split(",") if t.strip()][:4]
article = {
    "title": fm.get("title", "").strip(),
    "body_markdown": body,
    "published": publish,
    "tags": tags,
}
if fm.get("description"):
    article["description"] = fm["description"]
if fm.get("series"):
    article["series"] = fm["series"]

req = urllib.request.Request(
    "https://dev.to/api/articles",
    data=json.dumps({"article": article}).encode(),
    headers={"api-key": key, "Content-Type": "application/json",
             "Accept": "application/vnd.forem.api-v1+json",
             "User-Agent": "Mozilla/5.0 (devto-uploader)"},
    method="POST",
)
try:
    resp = urllib.request.urlopen(req, timeout=30)
    d = json.load(resp)
    print("OK  id=%s  published=%s" % (d.get("id"), d.get("published")))
    print("URL:", d.get("url"))
    print("edit:", "https://dev.to/dashboard")
except urllib.error.HTTPError as e:
    sys.exit("HTTP %s: %s" % (e.code, e.read().decode()[:500]))
