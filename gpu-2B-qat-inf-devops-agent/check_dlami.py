import boto3
import os

def check_region(region):
    print(f"\n--- Checking region: {region} ---")
    creds = {}
    if os.path.exists(".aws_creds"):
        with open(".aws_creds", "r") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    creds[k] = v

    ec2 = boto3.client(
        'ec2',
        region_name=region,
        aws_access_key_id=creds.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=creds.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=creds.get("AWS_SESSION_TOKEN")
    )
    
    # 1. Check AMI
    image_id = None
    try:
        images_resp = ec2.describe_images(
            Owners=["amazon"],
            Filters=[{"Name": "name", "Values": ["*Deep Learning AMI Neuron*Ubuntu 22.04*"]}]
        )
        if images_resp.get("Images"):
            sorted_images = sorted(images_resp["Images"], key=lambda x: x["CreationDate"], reverse=True)
            image_id = sorted_images[0]["ImageId"]
            print(f"Latest DLAMI Neuron AMI ID: {image_id} (Created: {sorted_images[0]['CreationDate']})")
        else:
            print("No DLAMI Neuron images found!")
    except Exception as e:
        print(f"Failed to resolve AMI: {e}")

    # 2. Check Subnets
    try:
        subnets = ec2.describe_subnets(
            Filters=[{"Name": "map-public-ip-on-launch", "Values": ["true"]}]
        )
        sub_list = []
        for s in subnets.get("Subnets", []):
            sub_list.append((s["SubnetId"], s["AvailabilityZone"]))
        print(f"Found {len(sub_list)} public subnets: {sub_list}")
    except Exception as e:
        print(f"Failed to describe subnets: {e}")

if __name__ == "__main__":
    for r in ["us-east-1", "us-east-2", "us-west-1", "us-west-2"]:
        check_region(r)
