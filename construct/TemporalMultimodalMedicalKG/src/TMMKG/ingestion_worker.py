"""Run KG import jobs described by local JSON files.

This is a lightweight ingestion worker for production-style file import.
It keeps the existing domain pipelines unchanged and adds a simple job layer:

    jobs/pending/*.json    -> waiting jobs
    jobs/running/*.json    -> jobs currently being processed
    jobs/succeeded/*.json  -> completed jobs
    jobs/failed/*.json     -> failed jobs with error_message

Example job:

{
  "job_id": "20260606_001",
  "domain": "HBUT",
  "source_path": "/data/tmmkg/inbox/HBUT/home_user_training.xlsx",
  "result_dir": "/data/tmmkg/jobs/20260606_001/result",
  "parquet_dir": "/data/tmmkg/jobs/20260606_001/parquet",
  "overwrite_parquet": false,
  "batch_size": 50000,
  "dry_run": false
}

Usage:
    PYTHONPATH=src python -m TMMKG.ingestion_worker --scan

    PYTHONPATH=src python -m TMMKG.ingestion_worker \
      --scan-dir /data/tmmkg/inbox --archive-dir /data/tmmkg/archive

    PYTHONPATH=src python -m TMMKG.ingestion_worker \
      --source-path /data/tmmkg/inbox/HBUT/home_user_training.xlsx

    PYTHONPATH=src python -m TMMKG.ingestion_worker --once
    PYTHONPATH=src python -m TMMKG.ingestion_worker --jobs-dir jobs --poll-seconds 10
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from TMMKG.create_HBUT_KG import run_home_based_user_training_pipeline
from TMMKG.create_L2BA_KG import run_level_2_brain_ability_pipeline
from TMMKG.create_OOTL_KG import run_output_only_task_labels_pipeline
from TMMKG.services.entity_resolver import init_entity_resolver
from TMMKG.utils.config import load_config, project_path
from TMMKG.utils.logger import get_logger, setup_logging_from_config

logger = get_logger(__name__)

DOMAIN_HBUT = "HBUT"
DOMAIN_L2BA = "L2BA"
DOMAIN_OOTL = "OOTL"
SUPPORTED_DOMAINS = {DOMAIN_HBUT, DOMAIN_L2BA, DOMAIN_OOTL}
DEFAULT_FILENAME_DOMAIN_PATTERNS = {
    DOMAIN_HBUT: (
        "hbut",
        "home_based_user_training",
        "home_user_training",
        "home user training",
    ),
    DOMAIN_L2BA: (
        "l2ba",
        "level_2_brain_ability",
        "brain_ability",
        "brain ability",
    ),
    DOMAIN_OOTL: (
        "ootl",
        "output_only_task_labels",
        "output only task labels",
    ),
}
SUPPORTED_SOURCE_SUFFIXES = {".xlsx", ".xlsm", ".xls"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_job_id(domain: str, source_path: Path) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{timestamp}_{domain}_{source_path.stem}"


def normalize_domain(domain: str) -> str:
    normalized = domain.strip().upper()
    if normalized not in SUPPORTED_DOMAINS:
        raise ValueError(
            f"Unsupported domain: {domain}. "
            f"Expected one of: {', '.join(sorted(SUPPORTED_DOMAINS))}"
        )
    return normalized


def detect_domain_from_filename(source_path: Path, config: dict[str, Any]) -> str:
    filename = source_path.name.lower()
    configured_patterns = (
        config.get("ingestion", {}).get("filename_domain_patterns", {}) or {}
    )

    patterns = {
        domain: tuple(configured_patterns.get(domain, default_patterns))
        for domain, default_patterns in DEFAULT_FILENAME_DOMAIN_PATTERNS.items()
    }

    matched_domains = [
        domain
        for domain, domain_patterns in patterns.items()
        if any(str(pattern).lower() in filename for pattern in domain_patterns)
    ]

    if len(matched_domains) == 1:
        return matched_domains[0]

    if not matched_domains:
        raise ValueError(
            f"Cannot detect domain from filename: {source_path.name}. "
            "Pass --domain explicitly or configure ingestion.filename_domain_patterns."
        )

    raise ValueError(
        f"Filename matches multiple domains: {source_path.name} -> "
        f"{', '.join(sorted(matched_domains))}"
    )


def ensure_job_dirs(jobs_dir: Path) -> dict[str, Path]:
    dirs = {
        "pending": jobs_dir / "pending",
        "running": jobs_dir / "running",
        "succeeded": jobs_dir / "succeeded",
        "failed": jobs_dir / "failed",
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    tmp_path.replace(path)


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return path.with_name(f"{path.stem}_{timestamp}{path.suffix}")


def move_job_file(src: Path, dst_dir: Path) -> Path:
    dst = unique_destination(dst_dir / src.name)
    shutil.move(str(src), str(dst))
    return dst


def move_source_file(src: Path, archive_dir: Path, status: str) -> Path:
    dst_dir = archive_dir / status
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        dst.unlink()
    shutil.move(str(src), str(dst))
    return dst


def get_job_id(job: dict[str, Any], job_path: Path) -> str:
    return str(job.get("job_id") or job_path.stem)


def resolve_job_path(job: dict[str, Any], key: str) -> Path:
    value = job.get(key)
    if not value:
        raise ValueError(f"Job is missing required field: {key}")

    path = project_path(value)
    if path is None:
        raise ValueError(f"Invalid path for field: {key}")

    return path


def default_job_work_dir(jobs_dir: Path, job_id: str) -> Path:
    return jobs_dir / "work" / job_id


def get_result_dir(job: dict[str, Any], jobs_dir: Path, job_id: str) -> Path:
    value = job.get("result_dir") or default_job_work_dir(jobs_dir, job_id) / "result"
    return project_path(value)


def get_parquet_dir(job: dict[str, Any], jobs_dir: Path, job_id: str) -> Path:
    value = job.get("parquet_dir") or default_job_work_dir(jobs_dir, job_id) / "parquet"
    return project_path(value)


def init_hbut_resolver(config: dict[str, Any]):
    infra = config.get("infra", {})
    qdrant = infra.get("qdrant", {})
    embedding = config.get("embedding", {})
    resolver_config = config.get("entity_resolver", {})

    return init_entity_resolver(
        model_name=resolver_config.get("model_name", "Qwen3-Embedding-8B"),
        model_root=embedding.get("model_root"),
        base_collection=resolver_config.get("base_collection", "entity_aliases"),
        qdrant_url=qdrant.get("uri", "http://localhost:6333"),
        score_threshold=resolver_config.get("score_threshold", 0.90),
    )


def run_job(job: dict[str, Any], config: dict[str, Any], jobs_dir: Path) -> None:
    job_id = get_job_id(job, Path(str(job.get("_job_file", "job.json"))))
    domain = normalize_domain(str(job.get("domain", "")))
    source_path = resolve_job_path(job, "source_path")

    if not source_path.exists():
        raise FileNotFoundError(f"Source file does not exist: {source_path}")

    infra = config.get("infra", {})
    neo4j = infra.get("neo4j", {})
    ontology = config.get("ontology", {})
    pipeline_config = config.get("pipelines", {}).get(domain, {})

    uri = neo4j.get("uri", "bolt://localhost:7687")
    user = neo4j.get("user", "neo4j")
    password = neo4j.get("password", "password")
    batch_size = int(job.get("batch_size", pipeline_config.get("batch_size", 50_000)))
    overwrite_parquet = bool(
        job.get("overwrite_parquet", pipeline_config.get("overwrite_parquet", False))
    )
    limit_records = job.get("limit_records")
    result_dir = get_result_dir(job, jobs_dir, job_id)
    parquet_dir = get_parquet_dir(job, jobs_dir, job_id)

    logger.info(
        "Start job %s: domain=%s source=%s result_dir=%s parquet_dir=%s",
        job_id,
        domain,
        source_path,
        result_dir,
        parquet_dir,
    )

    if domain == DOMAIN_HBUT:
        dry_run = bool(job.get("dry_run", False))
        resolver = None if dry_run else init_hbut_resolver(config)

        run_home_based_user_training_pipeline(
            uri=uri,
            user=user,
            password=password,
            xlsx_path=str(source_path),
            result_dir=str(result_dir),
            parquet_dir=str(parquet_dir),
            batch_size=batch_size,
            overwrite_parquet=overwrite_parquet,
            limit_records=limit_records,
            dry_run=dry_run,
            resolver=resolver,
            ontology_mappings_dir=project_path(ontology.get("mappings_dir")),
        )
        return

    if domain == DOMAIN_L2BA:
        run_level_2_brain_ability_pipeline(
            uri=uri,
            user=user,
            password=password,
            xlsx_path=str(source_path),
            result_dir=str(result_dir),
            parquet_dir=str(parquet_dir),
            batch_size=batch_size,
            overwrite_parquet=overwrite_parquet,
            limit_records=limit_records,
            include_fields=job.get(
                "include_fields",
                pipeline_config.get("include_fields"),
            ),
            skip_fields=job.get("skip_fields", pipeline_config.get("skip_fields")),
        )
        return

    if domain == DOMAIN_OOTL:
        run_output_only_task_labels_pipeline(
            uri=uri,
            user=user,
            password=password,
            xlsx_path=str(source_path),
            base_result_dir=str(result_dir),
            parquet_dir=str(parquet_dir),
            batch_size=batch_size,
            overwrite_parquet=overwrite_parquet,
            limit_records=limit_records,
        )
        return

    raise ValueError(f"Unsupported domain: {domain}")


def process_pending_job(job_path: Path, dirs: dict[str, Path], config: dict[str, Any]) -> None:
    running_path = move_job_file(job_path, dirs["running"])
    job = read_json(running_path)
    job_id = get_job_id(job, running_path)

    job.update(
        {
            "job_id": job_id,
            "status": "running",
            "started_at": utc_now(),
            "_job_file": str(running_path),
        }
    )
    write_json_atomic(running_path, job)

    try:
        run_job(job, config=config, jobs_dir=dirs["pending"].parent)
    except Exception as exc:
        logger.exception("Job failed: %s", job_id)
        job.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        write_json_atomic(running_path, job)
        failed_path = move_job_file(running_path, dirs["failed"])
        logger.info("Moved failed job %s to %s", job_id, failed_path)
        return

    job.update(
        {
            "status": "succeeded",
            "finished_at": utc_now(),
            "error_message": None,
        }
    )
    write_json_atomic(running_path, job)
    succeeded_path = move_job_file(running_path, dirs["succeeded"])
    logger.info("Moved succeeded job %s to %s", job_id, succeeded_path)


def run_worker(jobs_dir: Path, config: dict[str, Any], once: bool, poll_seconds: int) -> None:
    dirs = ensure_job_dirs(jobs_dir)
    logger.info("Ingestion worker started. jobs_dir=%s once=%s", jobs_dir, once)

    while True:
        pending_jobs = sorted(dirs["pending"].glob("*.json"))

        if not pending_jobs:
            if once:
                logger.info("No pending jobs.")
                return

            time.sleep(poll_seconds)
            continue

        for job_path in pending_jobs:
            process_pending_job(job_path, dirs=dirs, config=config)

        cleanup_old_job_records(jobs_dir, config)

        if once:
            return


def run_direct_import(
    config: dict[str, Any],
    jobs_dir: Path,
    domain: str | None,
    source_path: str,
    result_dir: str | None = None,
    parquet_dir: str | None = None,
    batch_size: int | None = None,
    overwrite_parquet: bool | None = None,
    dry_run: bool = False,
    limit_records: int | None = None,
) -> None:
    source = project_path(source_path)
    normalized_domain = (
        normalize_domain(domain) if domain else detect_domain_from_filename(source, config)
    )
    job_id = make_job_id(normalized_domain, source)

    job: dict[str, Any] = {
        "job_id": job_id,
        "domain": normalized_domain,
        "source_path": str(source),
        "status": "running",
        "started_at": utc_now(),
    }

    if result_dir:
        job["result_dir"] = result_dir
    if parquet_dir:
        job["parquet_dir"] = parquet_dir
    if batch_size is not None:
        job["batch_size"] = batch_size
    if overwrite_parquet is not None:
        job["overwrite_parquet"] = overwrite_parquet
    if dry_run:
        job["dry_run"] = True
    if limit_records is not None:
        job["limit_records"] = limit_records

    job_record_dir = jobs_dir / "direct"
    job_record_path = job_record_dir / f"{job_id}.json"
    write_json_atomic(job_record_path, job)

    try:
        run_job(job, config=config, jobs_dir=jobs_dir)
    except Exception as exc:
        logger.exception("Direct import failed: %s", job_id)
        job.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        write_json_atomic(job_record_path, job)
        cleanup_old_job_records(jobs_dir, config)
        raise

    job.update(
        {
            "status": "succeeded",
            "finished_at": utc_now(),
            "error_message": None,
        }
    )
    write_json_atomic(job_record_path, job)
    logger.info("Direct import succeeded: %s", job_id)
    cleanup_old_job_records(jobs_dir, config)


def cleanup_old_job_records(jobs_dir: Path, config: dict[str, Any]) -> None:
    retention_days = config.get("ingestion", {}).get("job_retention_days")
    if retention_days is None:
        return

    retention_days = int(retention_days)
    if retention_days <= 0:
        return

    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    record_dirs = [
        jobs_dir / "direct",
        jobs_dir / "succeeded",
        jobs_dir / "failed",
    ]

    removed = 0
    for record_dir in record_dirs:
        if not record_dir.exists():
            continue

        for path in record_dir.glob("*.json"):
            modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if modified_at >= cutoff:
                continue

            path.unlink()
            removed += 1

    if removed:
        logger.info(
            "Cleaned up %d job records older than %d days",
            removed,
            retention_days,
        )


def iter_source_files(scan_dir: Path) -> list[Path]:
    if not scan_dir.exists():
        raise FileNotFoundError(f"Scan directory does not exist: {scan_dir}")
    if not scan_dir.is_dir():
        raise NotADirectoryError(f"Scan path is not a directory: {scan_dir}")

    return sorted(
        path
        for path in scan_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SOURCE_SUFFIXES
        and not path.name.startswith("~$")
    )


def run_scan_import(
    config: dict[str, Any],
    jobs_dir: Path,
    scan_dir: str,
    archive_dir: str | None = None,
    batch_size: int | None = None,
    overwrite_parquet: bool | None = None,
    dry_run: bool = False,
    limit_records: int | None = None,
) -> None:
    scan_path = project_path(scan_dir)
    ingestion_config = config.get("ingestion", {})
    archive_path = project_path(
        archive_dir
        or ingestion_config.get("archive_dir")
        or jobs_dir / "archive"
    )

    source_files = iter_source_files(scan_path)
    logger.info("Found %d source files in %s", len(source_files), scan_path)

    for source_file in source_files:
        try:
            run_direct_import(
                config=config,
                jobs_dir=jobs_dir,
                domain=None,
                source_path=str(source_file),
                batch_size=batch_size,
                overwrite_parquet=overwrite_parquet,
                dry_run=dry_run,
                limit_records=limit_records,
            )
        except Exception:
            failed_path = move_source_file(source_file, archive_path, "failed")
            logger.info("Moved failed source file to %s", failed_path)
            continue

        succeeded_path = move_source_file(source_file, archive_path, "succeeded")
        logger.info("Moved succeeded source file to %s", succeeded_path)

    cleanup_old_job_records(jobs_dir, config)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pending TMMKG ingestion jobs from JSON files."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to shared YAML config. Defaults to configs/tmmkg.yaml.",
    )
    parser.add_argument(
        "--jobs-dir",
        type=str,
        default="jobs",
        help="Directory containing pending/running/succeeded/failed job folders.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process currently pending jobs once and exit.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=10,
        help="Polling interval when running continuously.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Direct import domain: HBUT, L2BA, or OOTL. If omitted, detect from filename.",
    )
    parser.add_argument(
        "--source-path",
        type=str,
        default=None,
        help="Direct import source XLSX path.",
    )
    parser.add_argument(
        "--scan-dir",
        type=str,
        default=None,
        help="Directory to scan once for source Excel files.",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan ingestion.scan_dir from config once.",
    )
    parser.add_argument(
        "--archive-dir",
        type=str,
        default=None,
        help="Directory for succeeded/failed source file archives in scan mode.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Optional direct import result directory.",
    )
    parser.add_argument(
        "--parquet-dir",
        type=str,
        default=None,
        help="Optional direct import parquet directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional direct import Neo4j batch size.",
    )
    parser.add_argument(
        "--overwrite-parquet",
        action="store_true",
        help="Overwrite existing parquet files in direct import mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Direct import dry-run mode. Currently used by HBUT.",
    )
    parser.add_argument(
        "--limit-records",
        type=int,
        default=None,
        help="Only process the first N records.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging_from_config(config, "ingestion_worker")
    jobs_dir = project_path(args.jobs_dir)

    if args.scan or args.scan_dir:
        scan_dir = args.scan_dir or config.get("ingestion", {}).get("scan_dir")
        if not scan_dir:
            parser.error("--scan requires ingestion.scan_dir in config or --scan-dir.")

        run_scan_import(
            config=config,
            jobs_dir=jobs_dir,
            scan_dir=scan_dir,
            archive_dir=args.archive_dir,
            batch_size=args.batch_size,
            overwrite_parquet=True if args.overwrite_parquet else None,
            dry_run=args.dry_run,
            limit_records=args.limit_records,
        )
        return

    if args.source_path:
        run_direct_import(
            config=config,
            jobs_dir=jobs_dir,
            domain=args.domain,
            source_path=args.source_path,
            result_dir=args.result_dir,
            parquet_dir=args.parquet_dir,
            batch_size=args.batch_size,
            overwrite_parquet=True if args.overwrite_parquet else None,
            dry_run=args.dry_run,
            limit_records=args.limit_records,
        )
        return

    if args.domain:
        parser.error("--domain requires --source-path.")

    run_worker(
        jobs_dir=jobs_dir,
        config=config,
        once=args.once,
        poll_seconds=args.poll_seconds,
    )


if __name__ == "__main__":
    main()
