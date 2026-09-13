#!/usr/bin/env python3
from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_IMPORTS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "Pillow": "PIL",
    "requests": "requests",
    "PyYAML": "yaml",
    "folium": "folium",
    "matplotlib": "matplotlib",
    "torch": "torch",
    "ultralytics": "ultralytics",
}

REQUIRED_FILES = [
    ROOT / "models" / "pyl_poil_final_yolov8n.pt",
    ROOT / "models" / "pyl_poil_level1_weak_baseline.pt",
    ROOT / "data" / "reference" / "supports_yvelines.csv",
    ROOT / "config" / "default.yaml",
]


def main() -> int:
    failures = 0
    print("=== Dépendances ===")
    for package, module in REQUIRED_IMPORTS.items():
        try:
            imported = import_module(module)
            version = getattr(imported, "__version__", "version inconnue")
            print(f"[OK] {package}: {version}")
        except Exception as exc:
            failures += 1
            print(f"[ERREUR] {package}: {exc}")

    print("\n=== Fichiers ===")
    for path in REQUIRED_FILES:
        if path.exists():
            print(f"[OK] {path.relative_to(ROOT)}")
        else:
            failures += 1
            print(f"[ERREUR] absent: {path.relative_to(ROOT)}")

    if failures:
        print(f"\n{failures} problème(s) détecté(s).")
        return 1
    print("\nInstallation prête.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
