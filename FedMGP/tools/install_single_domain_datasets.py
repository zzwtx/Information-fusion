#!/usr/bin/env python3
"""Install the single-domain datasets listed in DATASETS.md.

The script is intentionally conservative:
- skip files/directories that already exist;
- download archives into a temporary directory and remove them after extraction;
- leave ImageNet as a manual step because it requires registration.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import time
import tempfile
import urllib.request
import zipfile
from pathlib import Path


DATA_ROOT = Path(__file__).resolve().parents[1] / "DATA"
PYTHON = Path("/home/xsf/Information-fusion/.venv/bin/python")


def log(message: str) -> None:
    print(f"[install] {message}", flush=True)


def run(cmd: list[str]) -> None:
    log(" ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_url(url: str, dest: Path, *, no_check_certificate: bool = False) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        log(f"skip existing archive: {dest}")
        return

    ensure_dir(dest.parent)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    log(f"download: {url}")
    cmd = ["wget", "-c", "--progress=dot:giga", "-O", str(tmp), url]
    if no_check_certificate:
        cmd.insert(1, "--no-check-certificate")
    try:
        run(cmd)
    except subprocess.CalledProcessError:
        if tmp.exists():
            tmp.unlink()
        raise
    tmp.rename(dest)


def download_gdrive(file_id: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        log(f"skip existing file: {dest}")
        return

    ensure_dir(dest.parent)
    cmd = [
        str(PYTHON if PYTHON.exists() else sys.executable),
        "-m",
        "gdown",
        f"https://drive.google.com/uc?id={file_id}",
        "-O",
        str(dest),
    ]
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, 4):
        try:
            run(cmd)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            log(f"gdown failed for {dest.name}, retry {attempt}/3")
            if dest.exists() and dest.stat().st_size == 0:
                dest.unlink()
            time.sleep(5 * attempt)
    assert last_error is not None
    raise last_error


def extract_tar(archive: Path, dest: Path, marker: Path) -> None:
    if marker.exists():
        log(f"skip extracted: {marker}")
        return
    ensure_dir(dest)
    log(f"extract tar: {archive} -> {dest}")
    mode = "r:gz" if archive.suffix in {".gz", ".tgz"} or archive.name.endswith(".tar.gz") else "r:*"
    with tarfile.open(archive, mode) as tf:
        tf.extractall(dest)


def extract_zip(archive: Path, dest: Path, marker: Path) -> None:
    if marker.exists():
        log(f"skip extracted: {marker}")
        return
    ensure_dir(dest)
    log(f"extract zip: {archive} -> {dest}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)


def with_tmp_archive(name: str):
    tmp_root = Path(tempfile.mkdtemp(prefix="fedmgp_dataset_", dir="/tmp"))
    return tmp_root / name, tmp_root


def cleanup(tmp_root: Path) -> None:
    shutil.rmtree(tmp_root, ignore_errors=True)


def install_oxford_pets(root: Path) -> None:
    ds = root / "oxford_pets"
    ensure_dir(ds)
    if (ds / "images").exists():
        log(f"skip extracted: {ds / 'images'}")
    else:
        archive, tmp = with_tmp_archive("oxford_pets_images.tar.gz")
        try:
            download_url("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", archive)
            extract_tar(archive, ds, ds / "images")
        finally:
            cleanup(tmp)

    if (ds / "annotations").exists():
        log(f"skip extracted: {ds / 'annotations'}")
    else:
        archive, tmp = with_tmp_archive("oxford_pets_annotations.tar.gz")
        try:
            download_url("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", archive)
            extract_tar(archive, ds, ds / "annotations")
        finally:
            cleanup(tmp)

    download_gdrive("1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs", ds / "split_zhou_OxfordPets.json")


def install_oxford_flowers(root: Path) -> None:
    ds = root / "oxford_flowers"
    ensure_dir(ds)
    if (ds / "jpg").exists():
        log(f"skip extracted: {ds / 'jpg'}")
    else:
        archive, tmp = with_tmp_archive("102flowers.tgz")
        try:
            download_url("https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz", archive)
            extract_tar(archive, ds, ds / "jpg")
        finally:
            cleanup(tmp)

    download_url(
        "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat",
        ds / "imagelabels.mat",
    )
    download_gdrive("1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0", ds / "cat_to_name.json")
    download_gdrive("1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT", ds / "split_zhou_OxfordFlowers.json")


def install_food101(root: Path) -> None:
    marker = root / "food-101" / "images"
    if marker.exists():
        log(f"skip extracted: {marker}")
    else:
        archive, tmp = with_tmp_archive("food-101.tar.gz")
        try:
            urls = [
                "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz",
                "https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/static/food-101.tar.gz",
            ]
            last_error: Exception | None = None
            for url in urls:
                try:
                    download_url(url, archive)
                    last_error = None
                    break
                except Exception as exc:  # pragma: no cover - diagnostic fallback
                    last_error = exc
                    if archive.exists():
                        archive.unlink()
            if last_error is not None:
                raise last_error
            extract_tar(archive, root, marker)
        finally:
            cleanup(tmp)

    download_gdrive("1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl", root / "food-101" / "split_zhou_Food101.json")


def install_stanford_cars(root: Path) -> None:
    ds = root / "stanford_cars"
    ensure_dir(ds)
    items = [
        ("http://ai.stanford.edu/~jkrause/car196/cars_train.tgz", "cars_train.tgz", ds / "cars_train"),
        ("http://ai.stanford.edu/~jkrause/car196/cars_test.tgz", "cars_test.tgz", ds / "cars_test"),
        ("https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz", "car_devkit.tgz", ds / "devkit"),
    ]
    for url, name, marker in items:
        if marker.exists():
            log(f"skip extracted: {marker}")
        else:
            archive, tmp = with_tmp_archive(name)
            try:
                download_url(url, archive)
                extract_tar(archive, ds, marker)
            finally:
                cleanup(tmp)

    download_url(
        "http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat",
        ds / "cars_test_annos_withlabels.mat",
    )
    download_gdrive("1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT", ds / "split_zhou_StanfordCars.json")


def install_fgvc_aircraft(root: Path) -> None:
    ds = root / "fgvc_aircraft"
    if ds.exists() and (ds / "images").exists():
        log(f"skip extracted: {ds}")
        return

    archive, tmp = with_tmp_archive("fgvc-aircraft-2013b.tar.gz")
    try:
        download_url(
            "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz",
            archive,
        )
        extract_tar(archive, tmp, tmp / "fgvc-aircraft-2013b" / "data")
        src = tmp / "fgvc-aircraft-2013b" / "data"
        if ds.exists():
            shutil.rmtree(ds)
        shutil.move(str(src), str(ds))
    finally:
        cleanup(tmp)


def install_ucf101(root: Path) -> None:
    ds = root / "ucf101"
    ensure_dir(ds)
    if (ds / "UCF-101-midframes").exists():
        log(f"skip extracted: {ds / 'UCF-101-midframes'}")
    else:
        archive, tmp = with_tmp_archive("UCF-101-midframes.zip")
        try:
            download_gdrive("10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O", archive)
            extract_zip(archive, ds, ds / "UCF-101-midframes")
        finally:
            cleanup(tmp)

    download_gdrive("1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y", ds / "split_zhou_UCF101.json")


def install_sun397(root: Path) -> None:
    ds = root / "sun397"
    ensure_dir(ds)
    if (ds / "SUN397").exists():
        log(f"skip extracted: {ds / 'SUN397'}")
    else:
        archive, tmp = with_tmp_archive("SUN397.tar.gz")
        try:
            download_url("http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz", archive)
            extract_tar(archive, ds, ds / "SUN397")
        finally:
            cleanup(tmp)

    if (ds / "Training_01.txt").exists():
        log(f"skip extracted: {ds / 'Training_01.txt'}")
    else:
        archive, tmp = with_tmp_archive("Partitions.zip")
        try:
            download_url("https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip", archive)
            extract_zip(archive, ds, ds / "Training_01.txt")
            partitions_dir = ds / "Partitions"
            if partitions_dir.exists():
                for path in partitions_dir.iterdir():
                    target = ds / path.name
                    if not target.exists():
                        shutil.move(str(path), str(target))
                try:
                    partitions_dir.rmdir()
                except OSError:
                    pass
        finally:
            cleanup(tmp)

    download_gdrive("1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq", ds / "split_zhou_SUN397.json")


def install_eurosat(root: Path) -> None:
    ds = root / "eurosat"
    ensure_dir(ds)
    if (ds / "2750").exists():
        log(f"skip extracted: {ds / '2750'}")
    else:
        archive, tmp = with_tmp_archive("EuroSAT.zip")
        try:
            download_url(
                "https://madm.dfki.de/files/sentinel/EuroSAT.zip",
                archive,
                no_check_certificate=True,
            )
            extract_zip(archive, ds, ds / "2750")
        finally:
            cleanup(tmp)

    # CoOp's original split file. The ID in this repo's DATASETS.md can fail
    # with current gdown because the public link is no longer retrievable.
    download_gdrive("1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o", ds / "split_zhou_EuroSAT.json")


def install_cifar(root: Path) -> None:
    specs = [
        (
            root / "cifar-10",
            "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
            "cifar-10-python.tar.gz",
            root / "cifar-10" / "cifar-10-batches-py",
        ),
        (
            root / "cifar-100",
            "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
            "cifar-100-python.tar.gz",
            root / "cifar-100" / "cifar-100-python",
        ),
    ]
    for ds, url, name, marker in specs:
        ensure_dir(ds)
        if marker.exists():
            log(f"skip extracted: {marker}")
        else:
            archive, tmp = with_tmp_archive(name)
            try:
                download_url(url, archive)
                extract_tar(archive, ds, marker)
            finally:
                cleanup(tmp)


def prepare_imagenet(root: Path) -> None:
    ds = root / "imagenet"
    ensure_dir(ds / "images")
    readme = ds / "README_MANUAL_INSTALL.txt"
    if not readme.exists():
        readme.write_text(
            "ImageNet requires registration and cannot be downloaded automatically.\n"
            "For this FedMGP loader, place data as:\n"
            "  DATA/imagenet/images/train/<wnid>/*.JPEG\n"
            "  DATA/imagenet/images/val/<wnid>/*.JPEG\n"
            "Also provide DATA/imagenet/classnames.txt mapping wnid to class name.\n",
            encoding="utf-8",
        )
    log("ImageNet skipped: registration/manual download required.")


def verify(root: Path, *, include_over_1gb: bool) -> None:
    checks = {
        "oxford_pets": ["images", "annotations", "split_zhou_OxfordPets.json"],
        "oxford_flowers": ["jpg", "imagelabels.mat", "cat_to_name.json", "split_zhou_OxfordFlowers.json"],
        "dtd": ["images", "imdb", "labels", "split_zhou_DescribableTextures.json"],
        "caltech-101": ["101_ObjectCategories", "split_zhou_Caltech101.json"],
        "food-101": ["images", "meta", "split_zhou_Food101.json"],
        "eurosat": ["2750", "split_zhou_EuroSAT.json"],
        "cifar-10": ["cifar-10-batches-py"],
        "cifar-100": ["cifar-100-python"],
    }
    if include_over_1gb:
        checks.update(
            {
                "stanford_cars": ["cars_train", "cars_test", "devkit", "cars_test_annos_withlabels.mat", "split_zhou_StanfordCars.json"],
                "fgvc_aircraft": ["images", "variants.txt", "images_variant_train.txt", "images_variant_val.txt", "images_variant_test.txt"],
                "ucf101": ["UCF-101-midframes", "split_zhou_UCF101.json"],
                "sun397": ["SUN397", "ClassName.txt", "Training_01.txt", "Testing_01.txt", "split_zhou_SUN397.json"],
            }
        )
    missing = []
    for dirname, rels in checks.items():
        for rel in rels:
            path = root / dirname / rel
            if not path.exists():
                missing.append(str(path.relative_to(root)))
    if missing:
        log("missing expected paths:")
        for item in missing:
            print(f"  - {item}", flush=True)
    else:
        log("selected automatically installable single-domain datasets look complete.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--skip-heavy", action="store_true", help="Skip Food101, UCF101 and SUN397.")
    parser.add_argument(
        "--include-over-1gb",
        action="store_true",
        help="Also install datasets whose required image archives are larger than 1GB.",
    )
    args = parser.parse_args()

    root = args.data_root.resolve()
    ensure_dir(root)

    install_oxford_pets(root)
    install_oxford_flowers(root)
    if not args.skip_heavy:
        install_food101(root)

    if args.include_over_1gb:
        install_stanford_cars(root)
        install_fgvc_aircraft(root)
        if not args.skip_heavy:
            install_ucf101(root)
            install_sun397(root)
    else:
        log("skip Stanford Cars, FGVC Aircraft, UCF101 and SUN397: required image archives exceed 1GB.")

    install_eurosat(root)
    install_cifar(root)
    prepare_imagenet(root)
    verify(root, include_over_1gb=args.include_over_1gb)


if __name__ == "__main__":
    main()
