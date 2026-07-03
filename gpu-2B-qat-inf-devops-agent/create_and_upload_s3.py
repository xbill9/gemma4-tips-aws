import boto3
import os
import json
import sys

def main():
    creds = {}
    if os.path.exists(".aws_creds"):
        with open(".aws_creds", "r") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    creds[k] = v

    s3 = boto3.client(
        's3',
        region_name='us-east-1',
        aws_access_key_id=creds.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=creds.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=creds.get("AWS_SESSION_TOKEN")
    )

    bucket_name = "xbill-gemma4-patches-2b"
    print(f"Checking or creating S3 bucket: {bucket_name}")

    try:
        s3.create_bucket(Bucket=bucket_name)
        print(f"Bucket {bucket_name} created successfully.")
    except Exception as e:
        if "BucketAlreadyExists" in str(e) or "BucketAlreadyOwnedByYou" in str(e):
            print(f"Bucket {bucket_name} already exists.")
        else:
            print(f"Error creating bucket, trying to use timestamp unique suffix: {e}")
            import time
            bucket_name = f"xbill-gemma4-patches-{int(time.time())}"
            try:
                s3.create_bucket(Bucket=bucket_name)
                print(f"Bucket {bucket_name} created successfully.")
            except Exception as ex:
                print(f"Fatal error creating bucket: {ex}")
                sys.exit(1)

    # Disable Public Access Block to keep permissions open as requested
    try:
        s3.delete_public_access_block(Bucket=bucket_name)
        print("Deleted public access block.")
    except Exception as e:
        print(f"Warning: could not delete public access block: {e}")

    # Set public bucket policy
    bucket_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "PublicReadGetObject",
                "Effect": "Allow",
                "Principal": "*",
                "Action": "s3:GetObject",
                "Resource": f"arn:aws:s3:::{bucket_name}/*"
            }
        ]
    }
    try:
        s3.put_bucket_policy(Bucket=bucket_name, Policy=json.dumps(bucket_policy))
        print("Applied open public read policy to bucket.")
    except Exception as e:
        print(f"Warning: could not apply public bucket policy: {e}")

    # Upload apply_all_patches.py
    file_path = "apply_all_patches.py"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found locally.")
        sys.exit(1)

    print(f"Uploading {file_path} to s3://{bucket_name}/apply_all_patches.py")
    try:
        s3.upload_file(
            file_path, 
            bucket_name, 
            "apply_all_patches.py",
            ExtraArgs={"ContentType": "text/plain"}
        )
        print("Upload successful!")
    except Exception as e:
        print(f"Upload failed: {e}")
        sys.exit(1)

    # Generate public URL and presigned URL as fallback
    public_url = f"https://{bucket_name}.s3.amazonaws.com/apply_all_patches.py"
    presigned_url = s3.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket_name, 'Key': 'apply_all_patches.py'},
        ExpiresIn=31536000  # 1 year
    )

    print("\n--- S3 URLs ---")
    print(f"Public URL (Open Permissions): {public_url}")
    print(f"Presigned URL (Always Works): {presigned_url}")
    
    # Save the bucket name to a file for other scripts to use
    with open("s3_bucket_name.txt", "w") as f:
        f.write(bucket_name)

if __name__ == "__main__":
    main()
