"""Arma el conjunto independiente de evaluacion (Parte 1, seccion 3.4 del enunciado).

Toma imagenes de dos datasets publicos que NO son el 70 Dog Breeds de Kaggle:
  - Oxford-IIIT Pet: razas que coinciden con las nuestras + gatos (imagenes sin perros).
  - Stanford Dogs: razas extra y escenas con varios perros (segun sus anotaciones).

Resultado en data/dataset/independent/ (no se versiona):
  <raza>/       imagenes "limpias" de un perro, para las Etapas 1 y 2
  _escenas/     escenas con 2 o mas perros, para la Etapa 3
  _sin_perro/   gatos, para verificar falsos positivos del detector
  fuentes.csv   origen y licencia de cada imagen

Uso:
    python scripts/build_independent_set.py
"""
from __future__ import annotations

import csv
import random
import sys
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = ROOT / "data" / "dataset" / "_downloads"
DEST = ROOT / "data" / "dataset" / "independent"

SEED = 42
POR_RAZA = 6
N_ESCENAS = 12
N_GATOS = 10

OXFORD_IMAGES = "https://thor.robots.ox.ac.uk/datasets/pets/images.tar.gz"
STANFORD_IMAGES = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
STANFORD_ANNOTATIONS = "http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar"

OXFORD_LICENCIA = "Oxford-IIIT Pet Dataset (Parkhi et al., 2012) - CC BY-SA 4.0"
STANFORD_LICENCIA = (
    "Stanford Dogs Dataset (Khosla et al., 2011) - imagenes de ImageNet, uso academico no comercial"
)

# raza en Oxford-IIIT Pet -> carpeta de nuestro dataset
OXFORD_A_RAZA = {
    "basset_hound": "Basset",
    "beagle": "Beagle",
    "boxer": "Boxer",
    "chihuahua": "Chihuahua",
    "great_pyrenees": "Great Perenees",
    "japanese_chin": "Japanese Spaniel",
    "newfoundland": "Newfoundland",
    "pomeranian": "Pomeranian",
    "pug": "Pug",
    "saint_bernard": "Saint Bernard",
    "scottish_terrier": "Scotch Terrier",
    "shiba_inu": "Shiba Inu",
    "yorkshire_terrier": "Yorkie",
    "american_pit_bull_terrier": "Pit Bull",
}

# raza en Stanford Dogs (sufijo de la carpeta, en minusculas) -> carpeta de nuestro dataset
STANFORD_A_RAZA = {
    "dalmatian": "Dalmation",
    "siberian_husky": "Siberian Husky",
    "golden_retriever": "Golden Retriever",
    "labrador_retriever": "Labrador",
    "german_shepherd": "German Sheperd",
    "border_collie": "Border Collie",
    "rottweiler": "Rottweiler",
    "doberman": "Doberman",
}


def descargar(url: str) -> Path:
    destino = DOWNLOADS / url.rsplit("/", 1)[1]
    if destino.exists():
        print(f"Ya descargado: {destino.name}")
        return destino
    print(f"Descargando {url} ...")
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_suffix(destino.suffix + ".part")
    urllib.request.urlretrieve(url, temporal)
    temporal.rename(destino)
    return destino


def nombres_en_tar(ruta: Path) -> list[str]:
    with tarfile.open(ruta) as tar:
        return [m.name for m in tar if m.isfile()]


def extraer(ruta: Path, elegidos: dict[str, Path]) -> None:
    """Copia los miembros elegidos del tar a su destino, en una sola pasada.

    Se lee el contenido con extractfile y se escribe a una ruta armada por nosotros,
    asi ningun nombre dentro del tar puede escribir fuera de DEST.
    """
    with tarfile.open(ruta) as tar:
        for miembro in tar:
            destino = elegidos.get(miembro.name)
            if destino is None:
                continue
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_bytes(tar.extractfile(miembro).read())


def contar_perros_stanford(ruta_anotaciones: Path) -> dict[str, int]:
    """id de imagen (ej. n02085620_10074) -> cantidad de perros anotados."""
    cantidades: dict[str, int] = {}
    with tarfile.open(ruta_anotaciones) as tar:
        for miembro in tar:
            if not miembro.isfile():
                continue
            raiz = ET.fromstring(tar.extractfile(miembro).read())
            cantidades[Path(miembro.name).name] = len(raiz.findall("object"))
    return cantidades


def main() -> None:
    if DEST.exists() and any(DEST.iterdir()):
        sys.exit(f"{DEST} ya existe y no esta vacia. Borrala si queres regenerarla.")

    rng = random.Random(SEED)
    filas: list[dict[str, str]] = []

    # --- Oxford-IIIT Pet: razas coincidentes + gatos ---
    oxford = descargar(OXFORD_IMAGES)
    por_raza: dict[str, list[str]] = {}
    gatos: list[str] = []
    for nombre in nombres_en_tar(oxford):
        archivo = Path(nombre).name
        if not archivo.endswith(".jpg"):
            continue
        clave = archivo.rsplit("_", 1)[0]
        if archivo[0].isupper():  # en Oxford-IIIT Pet los gatos empiezan con mayuscula
            gatos.append(nombre)
        elif clave in OXFORD_A_RAZA:
            por_raza.setdefault(OXFORD_A_RAZA[clave], []).append(nombre)

    elegidos: dict[str, Path] = {}
    for raza, nombres in sorted(por_raza.items()):
        for nombre in rng.sample(sorted(nombres), POR_RAZA):
            destino = DEST / raza / f"oxford_{Path(nombre).name}"
            elegidos[nombre] = destino
            filas.append({"archivo": destino.relative_to(DEST).as_posix(), "raza": raza,
                          "tipo": "limpia", "fuente": f"Oxford-IIIT Pet: {Path(nombre).name}",
                          "licencia": OXFORD_LICENCIA})
    for nombre in rng.sample(sorted(gatos), N_GATOS):
        destino = DEST / "_sin_perro" / f"oxford_{Path(nombre).name}"
        elegidos[nombre] = destino
        filas.append({"archivo": destino.relative_to(DEST).as_posix(), "raza": "",
                      "tipo": "sin_perro", "fuente": f"Oxford-IIIT Pet: {Path(nombre).name}",
                      "licencia": OXFORD_LICENCIA})
    print(f"Oxford-IIIT Pet: extrayendo {len(elegidos)} imagenes ...")
    extraer(oxford, elegidos)

    # --- Stanford Dogs: razas extra + escenas con varios perros ---
    anotaciones = descargar(STANFORD_ANNOTATIONS)
    stanford = descargar(STANFORD_IMAGES)
    perros_por_imagen = contar_perros_stanford(anotaciones)

    una_por_raza: dict[str, list[str]] = {}
    escenas: list[tuple[str, str]] = []
    for nombre in nombres_en_tar(stanford):
        carpeta = Path(nombre).parent.name  # ej. n02110341-dalmatian
        clave = carpeta.split("-", 1)[1].lower()
        cantidad = perros_por_imagen.get(Path(nombre).stem, 0)
        if clave in STANFORD_A_RAZA and cantidad == 1:
            una_por_raza.setdefault(STANFORD_A_RAZA[clave], []).append(nombre)
        if cantidad >= 2:
            escenas.append((nombre, carpeta.split("-", 1)[1]))

    elegidos = {}
    for raza, nombres in sorted(una_por_raza.items()):
        for nombre in rng.sample(sorted(nombres), POR_RAZA):
            destino = DEST / raza / f"stanford_{Path(nombre).name}"
            elegidos[nombre] = destino
            filas.append({"archivo": destino.relative_to(DEST).as_posix(), "raza": raza,
                          "tipo": "limpia", "fuente": f"Stanford Dogs: {Path(nombre).name}",
                          "licencia": STANFORD_LICENCIA})
    for nombre, raza_original in rng.sample(sorted(escenas), N_ESCENAS):
        destino = DEST / "_escenas" / f"stanford_{Path(nombre).name}"
        elegidos[nombre] = destino
        filas.append({"archivo": destino.relative_to(DEST).as_posix(),
                      "raza": raza_original, "tipo": "escena",
                      "fuente": f"Stanford Dogs: {Path(nombre).name} "
                                f"({perros_por_imagen[Path(nombre).stem]} perros anotados)",
                      "licencia": STANFORD_LICENCIA})
    print(f"Stanford Dogs: extrayendo {len(elegidos)} imagenes ...")
    extraer(stanford, elegidos)

    with open(DEST / "fuentes.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["archivo", "raza", "tipo", "fuente", "licencia"])
        writer.writeheader()
        writer.writerows(filas)

    razas = {f["raza"] for f in filas if f["tipo"] == "limpia"}
    print(f"Listo: {len(filas)} imagenes en {DEST}")
    print(f"  limpias: {sum(f['tipo'] == 'limpia' for f in filas)} ({len(razas)} razas)")
    print(f"  escenas con varios perros: {sum(f['tipo'] == 'escena' for f in filas)}")
    print(f"  sin perro (gatos): {sum(f['tipo'] == 'sin_perro' for f in filas)}")


if __name__ == "__main__":
    main()
