"""Upload pipeline artifacts to MinIO via the S3 API.

MinIO is a self-hosted S3-compatible object storage service, run here as a
Docker service in the local project stack. Access is implemented with boto3,
which uses the same API as AWS S3.

For production migration to real AWS S3, only ``endpoint_url`` and credentials
need to change; the upload logic remains the same.

Portfolio purpose:
This module demonstrates the S3 storage pattern used in nearly every modern
data-engineering architecture. Practical boto3 experience gained here is
directly transferable to AWS environments.
"""

import hashlib
import logging
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError

from scripts import config

logger = logging.getLogger(__name__)


def _create_s3_client() -> boto3.client:
    """Create and return a configured S3 client for MinIO.

    Returns:
        Configured boto3 S3 client.

    Raises:
        NoCredentialsError: If MinIO credentials are not configured.
    """
    logger.info(f"Connecting to MinIO at {config.MINIO_ENDPOINT}")

    access_key = getattr(config, "MINIO_ACCESS_KEY", getattr(config, "MINIO_ROOT_USER", None))
    secret_key = getattr(config, "MINIO_SECRET_KEY", getattr(config, "MINIO_ROOT_PASSWORD", None))

    if not access_key or not secret_key:
        raise NoCredentialsError()

    s3 = boto3.client(
        "s3",
        endpoint_url=config.MINIO_ENDPOINT,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    return s3


def _calculate_md5(file_path: Path) -> str:
    """Calculate MD5 hash of a local file.

    Args:
        file_path: Path to the file to hash.

    Returns:
        Hex-encoded MD5 hash string.
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _ensure_bucket_exists(s3_client: boto3.client, bucket_name: str) -> None:
    """Create the target bucket if it does not already exist.

    Args:
        s3_client: Configured boto3 S3 client.
        bucket_name: Name of the bucket to ensure.
    """
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        logger.info(f"Bucket '{bucket_name}' already exists")
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in {"404", "NoSuchBucket"}:
            s3_client.create_bucket(Bucket=bucket_name)
            logger.info(f"Created bucket '{bucket_name}'")
        else:
            raise


def _upload_file(s3_client: boto3.client, local_path: Path, bucket: str, s3_key: str) -> tuple[bool, bool]:
    """Upload a single file to S3, checking for idempotency via MD5 hash.

    Compares the local file's MD5 hash with the remote S3 object's ETag.
    If hashes match, skips upload. Otherwise, uploads the file.

    Args:
        s3_client: Configured boto3 S3 client.
        local_path: Path to the local file.
        bucket: Target bucket name.
        s3_key: S3 object key (path within the bucket).

    Returns:
        Tuple of (uploaded: bool, skipped: bool).
        - (True, False): File was uploaded.
        - (False, True): File was skipped (unchanged).
        - (False, False): Upload failed.
    """
    if not local_path.exists():
        logger.warning(f"File not found: {local_path}")
        return False, False

    try:
        local_md5 = _calculate_md5(local_path)
        
        # Check if remote object exists and compare ETags
        try:
            response = s3_client.head_object(Bucket=bucket, Key=s3_key)
            remote_etag = response["ETag"].strip('"')
            
            if local_md5 == remote_etag:
                logger.debug(f"Skipped (unchanged): {s3_key}")
                return False, True  # Not uploaded, but skipped (not an error)
        except ClientError:
            # Object doesn't exist yet, proceed with upload
            pass
        
        # Upload the file
        s3_client.upload_file(str(local_path), bucket, s3_key)
        logger.info(f"Uploaded {local_path.name} -> s3://{bucket}/{s3_key}")
        return True, False
    except ClientError as e:
        logger.error(f"Failed to upload {local_path}: {e}")
        return False, False


def _upload_directory(s3_client: boto3.client, local_dir: Path, bucket: str, s3_prefix: str) -> tuple[int, int]:
    """Upload all files in a local directory to S3 under the given prefix.

    Recursively walks the directory. Skips non-existent directories gracefully.
    Uses MD5-based idempotency: skips files that haven't changed since last upload.

    Args:
        s3_client: Configured boto3 S3 client.
        local_dir: Local directory to upload.
        bucket: Target bucket name.
        s3_prefix: S3 key prefix (e.g. "processed/").

    Returns:
        Tuple of (uploaded_count, skipped_count).
    """
    if not local_dir.exists():
        logger.warning(f"Directory {local_dir} not found, skipping")
        return 0, 0

    uploaded_count = 0
    skipped_count = 0
    clean_prefix = s3_prefix.rstrip("/")

    for file_path in local_dir.rglob("*"):
        if not file_path.is_file():
            continue

        if ".ipynb_checkpoints" in str(file_path) or "debug" in str(file_path.name):
            continue

        relative_path = file_path.relative_to(local_dir).as_posix()
        s3_key = f"{clean_prefix}/{relative_path}"

        uploaded, skipped = _upload_file(s3_client, file_path, bucket, s3_key)
        if uploaded:
            uploaded_count += 1
        elif skipped:
            skipped_count += 1

    logger.info(f"Uploaded {uploaded_count} files, skipped {skipped_count} unchanged from {local_dir} to s3://{bucket}/{clean_prefix}")
    return uploaded_count, skipped_count


def upload_outputs() -> dict[str, int]:
    """Upload all pipeline outputs to MinIO.

    This is the main entry point, called by Prefect as the final pipeline task.
    Uploads are organized into three categories: raw data, processed data,
    and outputs. Uses MD5-based idempotency to skip unchanged files.

    Returns:
        Dictionary with upload counts per category, e.g.
        {"raw": 3, "processed": 8, "outputs": 9, "skipped": 2, "total": 20}
    """
    logger.info("=" * 80)
    logger.info("Starting Storage Upload")
    logger.info("=" * 80)

    try:
        s3 = _create_s3_client()
        bucket = config.MINIO_BUCKET

        _ensure_bucket_exists(s3, bucket)

        raw_uploaded, raw_skipped = _upload_directory(s3, config.DATA_RAW_DIR, bucket, "raw")
        processed_uploaded, processed_skipped = _upload_directory(s3, config.DATA_PROCESSED_DIR, bucket, "processed")
        outputs_uploaded, outputs_skipped = _upload_directory(s3, config.OUTPUTS_DIR, bucket, "outputs")

        total_uploaded = raw_uploaded + processed_uploaded + outputs_uploaded
        total_skipped = raw_skipped + processed_skipped + outputs_skipped
        total = total_uploaded + total_skipped

        logger.info("=" * 80)
        logger.info("Storage Summary")
        logger.info("=" * 80)
        logger.info(f"Uploaded {total_uploaded} files, skipped {total_skipped} unchanged to bucket '{bucket}'")
        logger.info(f"  Raw data: {raw_uploaded} uploaded, {raw_skipped} skipped")
        logger.info(f"  Processed: {processed_uploaded} uploaded, {processed_skipped} skipped")
        logger.info(f"  Outputs: {outputs_uploaded} uploaded, {outputs_skipped} skipped")

        return {
            "raw": raw_uploaded,
            "processed": processed_uploaded,
            "outputs": outputs_uploaded,
            "skipped": total_skipped,
            "total": total,
        }

    except EndpointConnectionError:
        logger.error(
            f"MinIO not reachable at {config.MINIO_ENDPOINT}. Skipping upload. "
            f"Pipeline results are still available locally under {config.OUTPUTS_DIR}"
        )
        return {"raw": 0, "processed": 0, "outputs": 0, "skipped": 0, "total": 0}
    except NoCredentialsError:
        logger.error(
            "MinIO credentials not configured. Check MINIO_ACCESS_KEY and "
            "MINIO_SECRET_KEY env vars."
        )
        return {}
    except Exception as e:
        logger.error(f"Unexpected storage upload error: {e}", exc_info=True)
        return {"raw": 0, "processed": 0, "outputs": 0, "skipped": 0, "total": 0}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    result = upload_outputs()
    logger.info(f"Upload result: {result}")
