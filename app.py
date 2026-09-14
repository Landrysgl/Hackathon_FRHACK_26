import json
import math
from pathlib import Path
import re
import sys

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, ImageDraw

# YOLO est optionnel au démarrage pour permettre d'utiliser l'app
# même si ultralytics n'est pas encore installé.
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PYL-POIL",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Si le fichier est lancé avec ``python app.py`` au lieu de Streamlit, on
# affiche une instruction claire dans le terminal. Le script reste robuste et
# ne tombe plus sur un NameError si le référentiel n'est pas trouvé.
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx(suppress_warning=True) is None:
        print(
            "[PYL-POIL] Cette interface doit être lancée avec :\n"
            f'  "{sys.executable}" -m streamlit run "{Path(__file__).resolve()}"\n'
        )
except Exception:
    pass

APP_DIR = Path(__file__).resolve().parent


def _candidate_project_roots():
    """Retourne les emplacements plausibles de la racine du projet.

    L'application est normalement placée à la racine du dépôt. On accepte
    également un lancement depuis un sous-dossier ou depuis le répertoire de
    travail courant afin d'éviter de dépendre d'un chemin absolu.
    """
    roots = [APP_DIR, Path.cwd().resolve()]
    for base in (APP_DIR, Path.cwd().resolve()):
        roots.extend(base.parents)
    return list(dict.fromkeys(roots))


def find_project_root():
    """Détecte la racine contenant le dossier ``code/`` de Pyl-Poil."""
    for root in _candidate_project_roots():
        if (root / "code" / "models").exists() or (
            root / "code" / "data" / "reference"
        ).exists():
            return root
    return APP_DIR


PROJECT_ROOT = find_project_root()

# --------------------------------------------------------------------
# Références du projet Pyl-Poil
# --------------------------------------------------------------------
# L'application sait lire :
#   1) le gros export ANFR historique, s'il est présent à la racine ;
#   2) le référentiel propre livré avec Pyl-Poil (Yvelines).
ANFR_CSV_CANDIDATES = [
    PROJECT_ROOT / "SUP_SUPPORT_AVEC_TYPE.csv",
    PROJECT_ROOT / "code" / "data" / "reference" / "supports_yvelines.csv",
    APP_DIR / "SUP_SUPPORT_AVEC_TYPE.csv",
    APP_DIR / "code" / "data" / "reference" / "supports_yvelines.csv",
]

# Le modèle final doit être privilégié explicitement. Cela évite de charger
# accidentellement un ancien best.pt ou la baseline faible du Niveau 1.
PREFERRED_MODEL_CANDIDATES = [
    PROJECT_ROOT / "code" / "models" / "pyl_poil_final_yolov8n.pt",
    PROJECT_ROOT / "models" / "pyl_poil_final_yolov8n.pt",
    APP_DIR / "code" / "models" / "pyl_poil_final_yolov8n.pt",
    APP_DIR / "models" / "pyl_poil_final_yolov8n.pt",
]

# Paramètres validés pour le modèle final Pyl-Poil.
RECOMMENDED_CONFIDENCE = 0.05
RECOMMENDED_IMGSZ = 1024
RECOMMENDED_IOU = 0.50

# Pour une image centrée sur un site ANFR connu, le rayon de proximité local
# est volontairement plus large que dans la première version de l'interface.
# La calibration humaine du projet montre que le décalage ANFR -> objet visible
# peut atteindre plusieurs dizaines de mètres. Ce rayon LOCAL ne doit pas être
# confondu avec le filtre Niveau 3 de 150 m appliqué aux scans géoréférencés.
DEFAULT_MATCH_RADIUS_METERS = 50.0
DEFAULT_IMAGE_SIZE_METERS = 100.0


def resolve_existing_path(candidates):
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


ANFR_CSV = resolve_existing_path(ANFR_CSV_CANDIDATES)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background-color: #020817;
        }

        section[data-testid="stSidebar"] {
            background-color: #06122E;
        }

        h1, h2, h3 {
            color: #F8FAFC;
        }

        p, label {
            color: #CBD5E1;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        div[data-testid="stMetric"] {
            background-color: #06122E;
            border: 1px solid #123A8C;
            padding: 15px;
            border-radius: 10px;
        }

        div[data-testid="stMetricLabel"] {
            color: #94A3B8;
        }

        div[data-testid="stMetricValue"] {
            color: #008CFF;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# OUTILS
# ============================================================

def dms_to_decimal(degrees, minutes, seconds, direction):
    """Convertit des coordonnées DMS en coordonnées décimales."""
    try:
        decimal = (
            float(degrees)
            + float(minutes) / 60
            + float(seconds) / 3600
        )

        if str(direction).upper().strip() in ["S", "W"]:
            decimal *= -1

        return decimal
    except (ValueError, TypeError):
        return None


def find_model_files():
    """Retourne les poids disponibles en privilégiant le modèle final."""
    candidates = []

    # Le modèle officiel est toujours proposé en premier s'il existe.
    for path in PREFERRED_MODEL_CANDIDATES:
        if path.exists():
            candidates.append(path)

    # On ne parcourt pas récursivement tout le dépôt : un environnement .venv
    # CUDA peut contenir plusieurs gigaoctets et ralentir fortement le démarrage.
    search_roots = [
        PROJECT_ROOT / "code" / "models",
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "runs",
        PROJECT_ROOT / "code" / "runs",
        APP_DIR / "code" / "models",
        APP_DIR / "models",
    ]

    for root in search_roots:
        if not root.exists():
            continue
        for pattern in ("*.pt", "*.onnx"):
            candidates.extend(root.rglob(pattern))

    candidates = list(dict.fromkeys(candidates))

    def priority(path):
        name = path.name.lower()
        full = str(path).lower()
        if name == "pyl_poil_final_yolov8n.pt":
            rank = 0
        elif "pyl_poil_final" in name:
            rank = 1
        elif name == "best.pt":
            rank = 2
        elif "level1" in full or "weak" in full:
            rank = 4
        else:
            rank = 3
        return (rank, len(path.parts), str(path))

    candidates.sort(key=priority)
    return candidates


def _read_csv_robust(path):
    """Lit un CSV virgule ou point-virgule sans imposer le format historique."""
    # sep=None + moteur Python détecte correctement les deux variantes livrées.
    try:
        return pd.read_csv(
            path,
            sep=None,
            engine="python",
            dtype=str,
        )
    except Exception:
        # Fallback explicite utile pour certains exports ANFR volumineux.
        for sep in (";", ","):
            try:
                df = pd.read_csv(path, sep=sep, dtype=str, low_memory=False)
                if len(df.columns) > 1:
                    return df
            except Exception:
                pass
        raise


def _numeric_series(series):
    """Convertit une série numérique en tolérant la virgule décimale."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )


@st.cache_data(show_spinner=False)
def load_data():
    """Charge le référentiel ANFR, brut ou prétraité par Pyl-Poil."""
    if not ANFR_CSV.exists():
        searched = "\n".join(f"- {p}" for p in ANFR_CSV_CANDIDATES)
        raise FileNotFoundError(
            "Aucun référentiel ANFR n'a été trouvé. Chemins testés :\n"
            + searched
        )

    df = _read_csv_robust(ANFR_CSV)
    df.columns = [str(c).strip() for c in df.columns]

    # Format propre Pyl-Poil : latitude / longitude déjà en décimal.
    if {"latitude", "longitude"}.issubset(df.columns):
        df["latitude"] = _numeric_series(df["latitude"])
        df["longitude"] = _numeric_series(df["longitude"])
    else:
        # Format ANFR historique : coordonnées DMS.
        required = [
            "COR_NB_DG_LAT",
            "COR_NB_MN_LAT",
            "COR_NB_SC_LAT",
            "COR_CD_NS_LAT",
            "COR_NB_DG_LON",
            "COR_NB_MN_LON",
            "COR_NB_SC_LON",
            "COR_CD_EW_LON",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                "Le référentiel ne contient ni latitude/longitude décimales "
                "ni toutes les colonnes DMS ANFR. Colonnes manquantes : "
                + ", ".join(missing)
            )

        lat_deg = _numeric_series(df["COR_NB_DG_LAT"])
        lat_min = _numeric_series(df["COR_NB_MN_LAT"])
        lat_sec = _numeric_series(df["COR_NB_SC_LAT"])
        lon_deg = _numeric_series(df["COR_NB_DG_LON"])
        lon_min = _numeric_series(df["COR_NB_MN_LON"])
        lon_sec = _numeric_series(df["COR_NB_SC_LON"])

        lat = lat_deg + lat_min / 60.0 + lat_sec / 3600.0
        lon = lon_deg + lon_min / 60.0 + lon_sec / 3600.0

        lat_dir = df["COR_CD_NS_LAT"].astype(str).str.upper().str.strip()
        lon_dir = df["COR_CD_EW_LON"].astype(str).str.upper().str.strip()
        df["latitude"] = lat.where(~lat_dir.isin(["S"]), -lat)
        df["longitude"] = lon.where(~lon_dir.isin(["W"]), -lon)

    # Harmonisation du nom de la nature du support entre les deux sources.
    if "TYPE" not in df.columns:
        if "NAT_LB_NOM" in df.columns:
            df["TYPE"] = df["NAT_LB_NOM"]
        else:
            df["TYPE"] = "Non renseigné"

    # Colonnes utilisées par l'interface : on les crée si besoin afin de ne pas
    # faire planter la cartographie avec le référentiel compact Pyl-Poil.
    for col in [
        "SUP_ID",
        "STA_NM_ANFR",
        "SUP_NM_HAUT",
        "ADR_LB_LIEU",
        "ADR_NM_CP",
        "COM_CD_INSEE",
    ]:
        if col not in df.columns:
            df[col] = pd.NA

    # SUP_ID doit rester textuel pour faire correspondre proprement les noms
    # d'images du type ANFR_105408.jpg ou 105408.jpg.
    df["SUP_ID"] = (
        df["SUP_ID"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )

    return df


@st.cache_resource
def load_yolo_model(weights_path):
    """Charge et met en cache le modèle YOLO."""
    if YOLO is None:
        raise ImportError(
            "Le package ultralytics n'est pas installé. "
            "Installez-le avec : pip install ultralytics"
        )

    return YOLO(weights_path)


def image_geometry(image, image_size_meters=DEFAULT_IMAGE_SIZE_METERS):
    """Retourne la géométrie métrique supposée d'une image centrée ANFR.

    ``image_size_meters`` est la largeur/hauteur au sol de l'orthophoto. La
    valeur par défaut (100 m) correspond au générateur historique de l'app.
    Pour une image quelconque sans géoréférencement, cette conversion ne doit
    pas être interprétée comme une distance ANFR fiable.
    """
    width, height = image.size
    return (
        width,
        height,
        float(image_size_meters) / width,
        float(image_size_meters) / height,
    )


def extract_detections(result, image=None, image_size_meters=DEFAULT_IMAGE_SIZE_METERS):
    """Transforme un résultat Ultralytics en DataFrame exploitable."""
    detections = []

    names = result.names if hasattr(result, "names") else {}

    if result.boxes is None:
        return pd.DataFrame(
            columns=[
                "classe",
                "confiance",
                "x1",
                "y1",
                "x2",
                "y2",
                "centre_x",
                "centre_y",
                "distance_centre_m",
            ]
        )

    for box in result.boxes:
        xyxy = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])

        if isinstance(names, dict):
            class_name = names.get(cls_id, str(cls_id))
        else:
            class_name = (
                names[cls_id]
                if cls_id < len(names)
                else str(cls_id)
            )

        x1, y1, x2, y2 = xyxy
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # Les patches du projet sont centrés sur le support ANFR et couvrent
        # 100 m x 100 m. Utiliser la taille réelle évite de supposer 512x512.
        if image is not None:
            width, height, px_to_m_x, px_to_m_y = image_geometry(image, image_size_meters)
        else:
            width = height = RECOMMENDED_IMGSZ
            px_to_m_x = px_to_m_y = (
                float(image_size_meters) / RECOMMENDED_IMGSZ
            )

        dx_m = (cx - width / 2) * px_to_m_x
        dy_m = (cy - height / 2) * px_to_m_y
        distance_m = math.sqrt(dx_m**2 + dy_m**2)

        detections.append(
            {
                "classe": class_name,
                "confiance": round(conf, 4),
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1),
                "centre_x": round(cx, 1),
                "centre_y": round(cy, 1),
                "distance_centre_m": round(distance_m, 2),
            }
        )

    columns = [
        "classe", "confiance", "x1", "y1", "x2", "y2",
        "centre_x", "centre_y", "distance_centre_m",
    ]
    return pd.DataFrame(detections, columns=columns)


def annotate_image(
    image,
    result,
    match_radius_m,
    image_size_meters=DEFAULT_IMAGE_SIZE_METERS,
    draw_reference=True,
):
    """Dessine les détections YOLO et le point de référence ANFR au centre."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    # Le centre ne représente une position ANFR que si l'image est bien une
    # orthophoto centrée sur le support de référence.
    width, height, px_to_m_x, px_to_m_y = image_geometry(
        image, image_size_meters
    )
    center = (width / 2, height / 2)

    if draw_reference:
        radius_px_x = match_radius_m / px_to_m_x
        radius_px_y = match_radius_m / px_to_m_y

        draw.ellipse(
            (
                center[0] - radius_px_x,
                center[1] - radius_px_y,
                center[0] + radius_px_x,
                center[1] + radius_px_y,
            ),
            outline=(255, 180, 0),
            width=3,
        )
        draw.ellipse(
            (
                center[0] - 5,
                center[1] - 5,
                center[0] + 5,
                center[1] + 5,
            ),
            fill=(255, 180, 0),
        )

    if result.boxes is not None:
        names = result.names if hasattr(result, "names") else {}

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            if isinstance(names, dict):
                class_name = names.get(cls_id, str(cls_id))
            else:
                class_name = (
                    names[cls_id]
                    if cls_id < len(names)
                    else str(cls_id)
                )

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            distance_m = math.hypot(
                (cx - width / 2) * px_to_m_x,
                (cy - height / 2) * px_to_m_y,
            )

            if draw_reference:
                is_match = distance_m <= match_radius_m
                outline = (0, 220, 120) if is_match else (255, 90, 70)
            else:
                # Sans référence géographique fiable, on ne code pas les
                # boîtes en vert/rouge : ce serait suggérer à tort une
                # correspondance ou une anomalie ANFR.
                outline = (40, 170, 255)

            draw.rectangle((x1, y1, x2, y2), outline=outline, width=4)

            label = f"{class_name} {conf:.0%}"
            draw.rectangle(
                (x1, max(0, y1 - 22), x1 + max(100, len(label) * 8), y1),
                fill=outline,
            )
            draw.text(
                (x1 + 4, max(0, y1 - 20)),
                label,
                fill=(0, 0, 0),
            )

    return annotated


def reference_for_image(df, filename):
    """
    Retrouve le support ANFR correspondant au nom de fichier.

    Le pipeline du repo sauvegarde les images sous la forme SUP_ID.jpg.
    """
    stem = Path(filename).stem.strip()

    if "SUP_ID" not in df.columns:
        return None

    ids = df["SUP_ID"].astype(str).str.strip()
    match = df[ids == stem]

    # Also accept names such as "SUP_ID_12345" or "12345_xxx".
    if match.empty:
        candidates = re.findall(r"\d+", stem)
        for candidate in candidates:
            match = df[ids == candidate]
            if not match.empty:
                break

    if match.empty:
        return None

    return match.iloc[0]


def compare_detections_to_reference(detections, reference, radius_m):
    """Produit le statut de correspondance IA / ANFR."""
    if reference is None:
        return detections.assign(
            statut="Référence ANFR introuvable",
            correspondance=False,
        )

    if detections.empty:
        return detections.assign(
            statut="🔴 Support ANFR non détecté par l'IA",
            correspondance=False,
        )

    result = detections.copy()
    result["correspondance"] = (
        result["distance_centre_m"] <= radius_m
    )
    result["statut"] = result["correspondance"].map(
        {
            True: "🟢 Détection IA proche du support ANFR",
            False: "🟠 Détection IA éloignée du support ANFR",
        }
    )
    return result


def make_json_download(
    detections,
    image=None,
    image_size_meters=DEFAULT_IMAGE_SIZE_METERS,
):
    payload = {
        "image_size_meters": float(image_size_meters),
        "image_pixels": list(image.size) if image is not None else None,
        "detections": detections.to_dict(orient="records"),
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")


# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

REFERENCE_COLUMNS = [
    "SUP_ID",
    "STA_NM_ANFR",
    "TYPE",
    "SUP_NM_HAUT",
    "ADR_LB_LIEU",
    "ADR_NM_CP",
    "COM_CD_INSEE",
    "latitude",
    "longitude",
]


def empty_reference_dataframe():
    """DataFrame vide mais compatible avec toutes les pages de l'interface."""
    frame = pd.DataFrame(columns=REFERENCE_COLUMNS)
    frame["latitude"] = pd.Series(dtype="float64")
    frame["longitude"] = pd.Series(dtype="float64")
    return frame


DATA_LOAD_ERROR = None
try:
    df = load_data()
except Exception as e:
    # Ne jamais laisser ``df`` indéfini. C'était la cause du NameError observé
    # lorsqu'un utilisateur lançait app.py sans le référentiel à proximité.
    DATA_LOAD_ERROR = str(e)
    df = empty_reference_dataframe()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("📡 PYL-POIL")
    st.caption("FRHACK 2026")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🗺️ Cartographie",
            "🤖 Détection IA",
            "🔎 Comparaison IA / ANFR",
            "📏 Évaluation",
            "📊 Dashboard",
        ],
    )

    st.divider()
    st.subheader("⚙️ Modèle YOLO")

    model_files = find_model_files()

    if model_files:
        # Les chemins affichés restent lisibles même si app.py n'est pas
        # exactement à la racine du dépôt.
        model_options = []
        model_lookup = {}
        for path in model_files:
            try:
                label = str(path.relative_to(PROJECT_ROOT))
            except ValueError:
                label = str(path)
            model_options.append(label)
            model_lookup[label] = path

        selected_model = st.selectbox(
            "Poids du modèle",
            model_options,
            index=0,
            help=(
                "Pyl-Poil privilégie explicitement le checkpoint final "
                "code/models/pyl_poil_final_yolov8n.pt lorsqu'il est présent."
            ),
        )
        weights_path = model_lookup[selected_model]
    else:
        weights_input = st.text_input(
            "Chemin vers les poids (.pt)",
            value="code/models/pyl_poil_final_yolov8n.pt",
        )
        weights_path = Path(weights_input)
        if not weights_path.is_absolute():
            weights_path = APP_DIR / weights_path

    is_final_model = weights_path.name == "pyl_poil_final_yolov8n.pt"
    if is_final_model:
        st.success("✓ Modèle final Pyl-Poil sélectionné")
    else:
        st.warning(
            "Vous n'utilisez pas le checkpoint final Pyl-Poil. Les résultats "
            "peuvent différer des métriques du rapport."
        )

    st.caption(
        "Paramètres validés : confiance 0,05 · image 1024 px · IoU NMS 0,50"
    )

    conf_threshold = st.slider(
        "Seuil de confiance",
        0.01,
        0.95,
        RECOMMENDED_CONFIDENCE,
        0.01,
        help=(
            "Le modèle final a été évalué à conf=0,05. Monter ce seuil peut "
            "faire disparaître de nombreux petits supports aériens."
        ),
    )

    imgsz = st.select_slider(
        "Résolution YOLO",
        options=[640, 768, 960, 1024, 1280],
        value=RECOMMENDED_IMGSZ,
        help="Le modèle final Pyl-Poil a été validé à imgsz=1024.",
    )

    nms_iou = st.slider(
        "IoU NMS",
        0.10,
        0.90,
        RECOMMENDED_IOU,
        0.05,
        help="Valeur validée dans le pipeline final : 0,50.",
    )

    st.markdown("##### Correspondance locale ANFR")
    match_radius = st.slider(
        "Rayon de proximité (m)",
        10.0,
        150.0,
        DEFAULT_MATCH_RADIUS_METERS,
        5.0,
        help=(
            "Rayon local pour une image centrée sur un site ANFR connu. "
            "Ce paramètre n'est PAS le filtre candidat Niveau 3 de 150 m."
        ),
    )

    image_size_meters = st.number_input(
        "Emprise supposée de l'image (m × m)",
        min_value=20.0,
        max_value=1000.0,
        value=DEFAULT_IMAGE_SIZE_METERS,
        step=10.0,
        help=(
            "Utilisé uniquement pour convertir la distance en pixels vers "
            "des mètres lors de la comparaison locale. Le générateur "
            "historique de l'application produit des crops de 100 × 100 m."
        ),
    )

    st.divider()
    st.caption(
        "⚠️ Une détection sans correspondance ANFR est un élément à examiner, "
        "pas la preuve d'un site non déclaré."
    )
    st.caption(
        "Niveau 3 officiel : distance au support ANFR le plus proche >= 150 m, "
        "puis déduplication à 30 m et revue humaine."
    )


# ============================================================
# HEADER
# ============================================================

st.title("📡 PYL-POIL")
st.caption(
    "Détection et qualification de structures radioélectriques "
    "sur images aériennes — FRHACK 2026"
)
if DATA_LOAD_ERROR:
    st.warning(
        "Référentiel ANFR non chargé. La détection IA reste utilisable, mais "
        "la cartographie et la comparaison ANFR seront limitées."
    )
    st.caption(
        "Placez `SUP_SUPPORT_AVEC_TYPE.csv` à la racine du dépôt ou conservez "
        "`code/data/reference/supports_yvelines.csv` dans l'arborescence Pyl-Poil."
    )
    with st.expander("Détail du chargement ANFR"):
        st.code(DATA_LOAD_ERROR)
else:
    st.caption(
        f"Référentiel chargé : `{ANFR_CSV}` · {len(df):,} supports".replace(",", " ")
    )


# ============================================================
# PAGE : CARTOGRAPHIE
# ============================================================

if page == "🗺️ Cartographie":

    st.header("🗺️ Cartographie des supports")

    st.write(
        "Visualisation géographique du référentiel ANFR utilisé "
        "pour le challenge."
    )

    col1, col2 = st.columns(2)

    with col1:
        types = sorted(
            df["TYPE"].dropna().unique().tolist()
        )

        selected_types = st.multiselect(
            "Type de support",
            types,
            default=types,
        )

    with col2:
        search_id = st.text_input(
            "Rechercher un SUP_ID",
            placeholder="Ex : 74167",
        )

    filtered_df = df[df["TYPE"].isin(selected_types)].copy()

    if search_id:
        filtered_df = filtered_df[
            filtered_df["SUP_ID"]
            .astype(str)
            .str.contains(
                search_id,
                case=False,
                na=False,
            )
        ]

    map_df = filtered_df.dropna(
        subset=["latitude", "longitude"]
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Supports totaux", len(df))

    with col2:
        st.metric("Supports filtrés", len(filtered_df))

    with col3:
        st.metric(
            "Types de supports",
            filtered_df["TYPE"].nunique(),
        )

    st.divider()

    if map_df.empty:
        st.warning(
            "Aucun support avec des coordonnées valides "
            "ne correspond aux filtres."
        )
    else:
        map_kwargs = dict(
            data_frame=map_df,
            lat="latitude",
            lon="longitude",
            color="TYPE",
            hover_name="SUP_ID",
            hover_data={
                "SUP_ID": True,
                "STA_NM_ANFR": True,
                "TYPE": True,
                "ADR_LB_LIEU": True,
                "ADR_NM_CP": True,
                "COM_CD_INSEE": True,
                "SUP_NM_HAUT": True,
                "latitude": False,
                "longitude": False,
            },
            zoom=5,
            height=650,
        )

        if hasattr(px, "scatter_map"):
            fig = px.scatter_map(**map_kwargs, map_style="open-street-map")
        else:
            fig = px.scatter_mapbox(
                **map_kwargs,
                mapbox_style="open-street-map",
            )

        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="#020817",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.subheader("📋 Données des supports")

    columns_to_show = [
        "SUP_ID",
        "STA_NM_ANFR",
        "TYPE",
        "SUP_NM_HAUT",
        "ADR_LB_LIEU",
        "ADR_NM_CP",
        "COM_CD_INSEE",
        "latitude",
        "longitude",
    ]

    existing_columns = [
        col for col in columns_to_show
        if col in filtered_df.columns
    ]

    st.dataframe(
        filtered_df[existing_columns],
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# PAGE : DÉTECTION IA
# ============================================================

elif page == "🤖 Détection IA":

    st.header("🤖 Détection IA")

    st.write(
        "Chargez une image aérienne du dataset puis lancez "
        "l'inférence avec votre modèle YOLO entraîné."
    )

    if YOLO is None:
        st.warning(
            "⚠️ Ultralytics n'est pas installé. "
            "Lancez : `pip install ultralytics`"
        )

    if not weights_path.exists():
        st.warning(
            f"⚠️ Poids YOLO introuvables : `{weights_path}`. "
            "Placez votre `best.pt` dans le repo ou indiquez son chemin "
            "dans la barre latérale."
        )

    uploaded_file = st.file_uploader(
        "Importer une image aérienne",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Image originale")
            st.image(
                image,
                use_container_width=True,
            )

        with col2:
            st.subheader("Référence ANFR")

            reference = reference_for_image(
                df,
                uploaded_file.name,
            )

            if reference is not None:
                st.success(
                    f"Support ANFR associé : **{reference['SUP_ID']}**"
                )

                reference_cols = [
                    c for c in [
                        "SUP_ID",
                        "STA_NM_ANFR",
                        "TYPE",
                        "SUP_NM_HAUT",
                        "ADR_LB_LIEU",
                        "ADR_NM_CP",
                        "latitude",
                        "longitude",
                    ]
                    if c in reference.index
                ]

                st.dataframe(
                    pd.DataFrame(
                        {
                            "Champ": reference_cols,
                            "Valeur": [
                                reference[c]
                                for c in reference_cols
                            ],
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.info(
                    "Le nom du fichier ne correspond pas à un SUP_ID "
                    "du référentiel ANFR. La détection YOLO reste disponible, "
                    "mais la comparaison métrique au support ANFR sera désactivée."
                )

        spatial_reference_valid = False
        if reference is not None:
            spatial_reference_valid = st.checkbox(
                "Cette image est bien centrée sur le support ANFR indiqué",
                value=True,
                help=(
                    "Cochez uniquement si l'image provient du générateur de "
                    "crops ANFR ou si vous savez que le support est placé au "
                    "centre. Pour une image quelconque, décochez cette option."
                ),
            )
        else:
            st.caption(
                "ℹ️ Sans géoréférencement fiable, Pyl-Poil affiche les boîtes "
                "YOLO mais ne conclut pas à une correspondance ANFR."
            )

        if st.button(
            "🚀 Lancer la détection",
            type="primary",
            use_container_width=True,
        ):

            if YOLO is None:
                st.error(
                    "Installez ultralytics avant de lancer la détection."
                )
                st.stop()

            if not weights_path.exists():
                st.error(
                    f"Fichier de poids introuvable : {weights_path}"
                )
                st.stop()

            try:
                with st.spinner("Chargement du modèle YOLO..."):
                    model = load_yolo_model(str(weights_path))

                with st.spinner(
                    f"Détection en cours — imgsz={imgsz}, conf={conf_threshold:.2f}"
                ):
                    results = model.predict(
                        source=image,
                        conf=conf_threshold,
                        imgsz=imgsz,
                        iou=nms_iou,
                        verbose=False,
                    )

                result = results[0]
                detections = extract_detections(
                    result,
                    image=image,
                    image_size_meters=image_size_meters,
                )

                st.session_state["last_image_name"] = uploaded_file.name
                st.session_state["last_detections"] = detections
                st.session_state["last_reference"] = reference
                st.session_state["last_result"] = result
                st.session_state["last_spatial_reference_valid"] = (
                    spatial_reference_valid
                )
                st.session_state["last_match_radius"] = match_radius
                st.session_state["last_image_size_meters"] = image_size_meters

                annotated = annotate_image(
                    image,
                    result,
                    match_radius,
                    image_size_meters=image_size_meters,
                    draw_reference=spatial_reference_valid,
                )

                st.divider()
                st.subheader("🎯 Résultat de la détection")

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric(
                        "Détections",
                        len(detections),
                    )

                with c2:
                    st.metric(
                        "Confiance max",
                        (
                            f"{detections['confiance'].max():.0%}"
                            if not detections.empty
                            else "—"
                        ),
                    )

                with c3:
                    if spatial_reference_valid and not detections.empty:
                        matches = int(
                            (
                                detections["distance_centre_m"]
                                <= match_radius
                            ).sum()
                        )
                        st.metric("Proches du support ANFR", matches)
                    elif spatial_reference_valid:
                        st.metric("Proches du support ANFR", 0)
                    else:
                        st.metric("Proches du support ANFR", "—")

                st.image(
                    annotated,
                    caption=(
                        "Vert = détection dans le rayon local ; rouge = "
                        "détection hors rayon. Jaune = position ANFR théorique "
                        "uniquement lorsque l'image est déclarée centrée."
                    ),
                    use_container_width=True,
                )

                if detections.empty:
                    st.warning(
                        "Aucune structure détectée au-dessus du seuil de "
                        "confiance. Vérifiez d'abord que le modèle final est "
                        "sélectionné et utilisez les paramètres recommandés "
                        "conf=0,05, imgsz=1024, IoU=0,50."
                    )
                else:
                    st.dataframe(
                        detections,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.download_button(
                        "⬇️ Télécharger les détections JSON",
                        data=make_json_download(
                            detections,
                            image=image,
                            image_size_meters=image_size_meters,
                        ),
                        file_name=(
                            f"{Path(uploaded_file.name).stem}"
                            "_detections.json"
                        ),
                        mime="application/json",
                    )

            except Exception as e:
                st.error(
                    f"❌ Erreur pendant l'inférence YOLO : {e}"
                )


# ============================================================
# PAGE : COMPARAISON
# ============================================================

elif page == "🔎 Comparaison IA / ANFR":

    st.header("🔎 Comparaison IA / ANFR")

    st.write(
        "La comparaison locale n'est activée que pour une image déclarée "
        "centrée sur un support ANFR. Elle ne remplace pas le pipeline "
        "géoréférencé de recherche de candidats du Niveau 3."
    )

    detections = st.session_state.get("last_detections")
    reference = st.session_state.get("last_reference")
    image_name = st.session_state.get("last_image_name")
    spatial_reference_valid = st.session_state.get(
        "last_spatial_reference_valid", False
    )
    comparison_radius = st.session_state.get(
        "last_match_radius", match_radius
    )

    if detections is None:
        st.info(
            "Lancez d'abord une détection dans l'onglet "
            "**🤖 Détection IA**."
        )
    else:
        if not spatial_reference_valid:
            st.warning(
                "La dernière image n'a pas été déclarée comme centrée sur une "
                "référence ANFR fiable. La comparaison spatiale est donc "
                "désactivée afin d'éviter une conclusion trompeuse."
            )
            st.stop()

        comparison = compare_detections_to_reference(
            detections,
            reference,
            comparison_radius,
        )

        total = len(detections)
        matched = (
            int(comparison["correspondance"].sum())
            if "correspondance" in comparison.columns
            else 0
        )
        outside_local_radius = total - matched

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Détections IA", total)

        with col2:
            st.metric("Correspondances", matched)

        with col3:
            st.metric("Hors rayon local", outside_local_radius)

        if reference is not None:
            st.success(
                f"Référence ANFR : **{reference['SUP_ID']}** "
                f"— image : `{image_name}`"
            )
        else:
            st.warning(
                "Aucune référence ANFR n'a pu être associée à cette image."
            )

        st.divider()

        if comparison.empty:
            st.error(
                "🔴 Aucun objet détecté par l'IA alors qu'une "
                "référence ANFR est attendue."
            )
        else:
            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Interprétation")

            if matched > 0:
                st.success(
                    "🟢 Au moins une détection IA est spatialement "
                    "proche de la position ANFR attendue."
                )

            if outside_local_radius > 0:
                st.warning(
                    "🟠 Certaines détections sont hors du rayon local autour "
                    "du support connu. Cela ne suffit pas à en faire des "
                    "candidats Niveau 3."
                )

        st.info(
            "⚠️ Cette page réalise une comparaison LOCALE sur une image "
            "centrée ANFR. Le Niveau 3 officiel de Pyl-Poil recherche les "
            "candidats sur des scans géoréférencés puis applique une distance "
            "au support ANFR le plus proche >= 150 m et une déduplication à "
            "30 m. Une détection isolée reste un candidat à examiner, jamais "
            "la preuve automatique d'un site non déclaré."
        )


# ============================================================
# PAGE : ÉVALUATION
# ============================================================

elif page == "📏 Évaluation":

    st.header("📏 Évaluation du modèle")

    st.write(
        "Évaluation du modèle YOLO. Si le dataset YOLO complet n'est pas "
        "présent localement, les métriques du holdout final livré dans le "
        "dépôt sont affichées comme référence."
    )

    data_root = PROJECT_ROOT / "data"
    yaml_candidates = [
        data_root / "antennes.yaml",
        data_root / "data.yaml",
        PROJECT_ROOT / "antennes.yaml",
        PROJECT_ROOT / "data.yaml",
        PROJECT_ROOT / "code" / "data" / "antennes.yaml",
        PROJECT_ROOT / "code" / "data" / "data.yaml",
        APP_DIR / "data" / "antennes.yaml",
        APP_DIR / "data" / "data.yaml",
    ]

    yaml_path = next((p for p in yaml_candidates if p.exists()), None)
    holdout_candidates = [
        PROJECT_ROOT / "code" / "results" / "metrics" / "level2_holdout_final.json",
        PROJECT_ROOT / "results" / "metrics" / "level2_holdout_final.json",
        APP_DIR / "code" / "results" / "metrics" / "level2_holdout_final.json",
        APP_DIR / "results" / "metrics" / "level2_holdout_final.json",
    ]
    holdout_path = next((p for p in holdout_candidates if p.exists()), None)

    if yaml_path is not None and YOLO is not None and weights_path.exists():
        st.success(f"Dataset YOLO trouvé : `{yaml_path}`")

        if st.button("📊 Évaluer le modèle", type="primary"):
            try:
                with st.spinner("Évaluation en cours..."):
                    model = load_yolo_model(str(weights_path))
                    metrics = model.val(
                        data=str(yaml_path),
                        imgsz=imgsz,
                        verbose=False,
                    )

                box_metrics = metrics.box
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("mAP@50", f"{box_metrics.map50:.3f}")
                with col2:
                    st.metric("mAP@50-95", f"{box_metrics.map:.3f}")
                with col3:
                    st.metric("Précision", f"{box_metrics.mp:.3f}")
                with col4:
                    st.metric("Rappel", f"{box_metrics.mr:.3f}")

                st.success(
                    "Évaluation terminée. Conservez également un jeu de "
                    "validation manuel indépendant pour le livrable final."
                )
            except Exception as e:
                st.error(f"❌ Impossible d'évaluer le modèle : {e}")

    elif holdout_path is not None:
        try:
            holdout = json.loads(holdout_path.read_text(encoding="utf-8"))

            st.info(
                "Le dataset YOLO annoté complet n'est pas versionné dans le "
                "dépôt. Voici les métriques du holdout final déjà calculées "
                "et livrées avec le projet."
            )

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("mAP@50", f"{holdout.get('map50_manual', 0):.3f}")
            with col2:
                st.metric(
                    "mAP@50-95",
                    f"{holdout.get('map50_95_manual', 0):.3f}",
                )
            with col3:
                st.metric(
                    "Précision",
                    f"{holdout.get('ultralytics_precision_manual', 0):.3f}",
                )
            with col4:
                st.metric(
                    "Rappel",
                    f"{holdout.get('ultralytics_recall_manual', 0):.3f}",
                )

            st.subheader("Détails du holdout final")
            st.dataframe(
                pd.DataFrame(
                    [
                        ("Images manuelles", holdout.get("manual_images_total")),
                        ("Images utilisables", holdout.get("manual_images_used")),
                        ("Visibles", holdout.get("visible")),
                        ("Non visibles", holdout.get("non_visible")),
                        ("Ambiguës exclues", holdout.get("ambigu_excluded")),
                        ("TP", holdout.get("TP")),
                        ("FP", holdout.get("FP")),
                        ("FN", holdout.get("FN")),
                        ("Seuil de confiance", holdout.get("conf_threshold_for_tp_fp_fn")),
                        ("IoU de correspondance", holdout.get("match_iou_threshold")),
                    ],
                    columns=["Mesure", "Valeur"],
                ),
                hide_index=True,
                use_container_width=True,
            )

            st.caption(f"Source locale : `{holdout_path}`")
        except Exception as e:
            st.error(f"❌ Impossible de lire les métriques livrées : {e}")

    elif YOLO is None:
        st.warning(
            "Ultralytics n'est pas installé et aucune métrique locale "
            "n'a été trouvée."
        )
    elif not weights_path.exists():
        st.warning(f"Poids introuvables : `{weights_path}`")
    else:
        st.warning(
            "Aucun `data.yaml` / `antennes.yaml` n'est présent et aucun "
            "fichier de métriques holdout n'a été trouvé."
        )

# ============================================================
# PAGE : DASHBOARD
# ============================================================

elif page == "📊 Dashboard":

    st.header("📊 Dashboard")

    total_supports = len(df)
    total_types = df["TYPE"].nunique()

    valid_coordinates = df[
        df["latitude"].notna()
        & df["longitude"].notna()
    ].shape[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Nombre de supports",
            f"{total_supports:,}".replace(",", " "),
        )

    with col2:
        st.metric(
            "Types de supports",
            total_types,
        )

    with col3:
        st.metric(
            "Coordonnées valides",
            f"{valid_coordinates:,}".replace(",", " "),
        )

    st.divider()

    st.subheader("📡 Répartition des supports par type")

    type_counts = (
        df["TYPE"]
        .value_counts()
        .reset_index()
    )

    type_counts.columns = [
        "TYPE",
        "COUNT",
    ]

    fig_bar = px.bar(
        type_counts,
        x="TYPE",
        y="COUNT",
        labels={
            "TYPE": "Type de support",
            "COUNT": "Nombre de supports",
        },
    )

    fig_bar.update_layout(
        height=500,
        paper_bgcolor="#020817",
        plot_bgcolor="#020817",
        font=dict(color="#F8FAFC"),
    )

    st.plotly_chart(
        fig_bar,
        use_container_width=True,
    )

    st.subheader("📊 Distribution des infrastructures")

    fig_pie = px.pie(
        type_counts,
        names="TYPE",
        values="COUNT",
        hole=0.45,
    )

    fig_pie.update_layout(
        height=500,
        paper_bgcolor="#020817",
        font=dict(color="#F8FAFC"),
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True,
    )