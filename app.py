import json
import math
from pathlib import Path
import re

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

APP_DIR = Path(__file__).resolve().parent

# The repository is normally launched from its root. Keep a small fallback
# list so the app also works if the reference CSV is moved into code/data.
ANFR_CSV_CANDIDATES = [
    APP_DIR / "SUP_SUPPORT_AVEC_TYPE.csv",
    APP_DIR / "code" / "data" / "reference" / "supports_yvelines.csv",
]

def resolve_existing_path(candidates):
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]

ANFR_CSV = resolve_existing_path(ANFR_CSV_CANDIDATES)

# Le script 3_creation_DB_ORTHO_images.py du repo génère des images
# centrées sur le support ANFR, en 512x512 px, avec une emprise 100x100 m.
IMAGE_PIXELS = 512
IMAGE_SIZE_METERS = 100.0
DEFAULT_MATCH_RADIUS_METERS = 20.0


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
    """Cherche automatiquement les poids YOLO présents dans le repo."""
    candidates = []

    for pattern in ("*.pt", "*.onnx"):
        candidates.extend(APP_DIR.rglob(pattern))

    # On évite les doublons et on privilégie les best.pt.
    candidates = list(dict.fromkeys(candidates))
    candidates.sort(
        key=lambda p: (
            0 if p.name.lower() == "best.pt" else 1,
            len(p.parts),
            str(p),
        )
    )
    return candidates


@st.cache_data
def load_data():
    """Charge le référentiel ANFR et calcule les coordonnées décimales."""
    if not ANFR_CSV.exists():
        raise FileNotFoundError(
            f"Le fichier {ANFR_CSV.name} est introuvable."
        )

    df = pd.read_csv(
        ANFR_CSV,
        sep=";",
        dtype=str,
        low_memory=False,
    )

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
            "Colonnes de coordonnées manquantes : "
            + ", ".join(missing)
        )

    df["latitude"] = df.apply(
        lambda row: dms_to_decimal(
            row["COR_NB_DG_LAT"],
            row["COR_NB_MN_LAT"],
            row["COR_NB_SC_LAT"],
            row["COR_CD_NS_LAT"],
        ),
        axis=1,
    )

    df["longitude"] = df.apply(
        lambda row: dms_to_decimal(
            row["COR_NB_DG_LON"],
            row["COR_NB_MN_LON"],
            row["COR_NB_SC_LON"],
            row["COR_CD_EW_LON"],
        ),
        axis=1,
    )

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

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


def image_geometry(image):
    """Retourne la géométrie métrique de l'image.

    Les patches du projet couvrent 100 m x 100 m. On conserve donc cette
    emprise, mais on utilise la taille réelle de l'image afin d'éviter un
    décalage lorsque l'utilisateur charge une image qui n'est pas 512x512.
    """
    width, height = image.size
    return width, height, IMAGE_SIZE_METERS / width, IMAGE_SIZE_METERS / height


def extract_detections(result, image=None):
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
            width, height, px_to_m_x, px_to_m_y = image_geometry(image)
        else:
            width = height = IMAGE_PIXELS
            px_to_m_x = px_to_m_y = IMAGE_SIZE_METERS / IMAGE_PIXELS

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

    return pd.DataFrame(detections)


def annotate_image(image, result, match_radius_m):
    """Dessine les détections YOLO et le point de référence ANFR au centre."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    # Centre = position théorique du support ANFR dans les images produites
    # par le pipeline du repo.
    width, height, px_to_m_x, px_to_m_y = image_geometry(image)
    center = (width / 2, height / 2)
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

            is_match = distance_m <= match_radius_m
            outline = (0, 220, 120) if is_match else (255, 90, 70)

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


def make_json_download(detections):
    payload = {
        "image_size_meters": IMAGE_SIZE_METERS,
        "image_pixels": IMAGE_PIXELS,
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

try:
    df = load_data()
except Exception as e:
    st.error(f"❌ Erreur lors du chargement des données ANFR : {e}")
    st.info(
        "Vérifiez que `SUP_SUPPORT_AVEC_TYPE.csv` est bien à la racine "
        "du dépôt et que le fichier est séparé par `;`."
    )
    st.stop()


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
        model_options = [str(p.relative_to(APP_DIR)) for p in model_files]
        default_index = 0

        selected_model = st.selectbox(
            "Poids du modèle",
            model_options,
            index=default_index,
            help=(
                "Le repo est parcouru automatiquement à la recherche "
                "de fichiers .pt/.onnx. Un best.pt est privilégié."
            ),
        )
        weights_path = APP_DIR / selected_model
    else:
        weights_input = st.text_input(
            "Chemin vers les poids (.pt)",
            value="runs/detect/antennes/weights/best.pt",
        )
        weights_path = Path(weights_input)
        if not weights_path.is_absolute():
            weights_path = APP_DIR / weights_path

    conf_threshold = st.slider(
        "Seuil de confiance",
        0.05,
        0.95,
        0.25,
        0.05,
    )

    imgsz = st.select_slider(
        "Résolution YOLO",
        options=[640, 768, 960, 1280],
        value=1280,
    )

    match_radius = st.slider(
        "Rayon de correspondance ANFR (m)",
        5.0,
        50.0,
        DEFAULT_MATCH_RADIUS_METERS,
        5.0,
        help=(
            "Dans le dataset généré par le repo, le support ANFR "
            "est au centre d'une image de 100 m x 100 m."
        ),
    )

    st.divider()
    st.caption(
        "⚠️ Une détection sans correspondance ANFR est un candidat "
        "à examiner, pas la preuve d'un site non déclaré."
    )


# ============================================================
# HEADER
# ============================================================

st.title("📡 PYL-POIL")
st.caption(
    "Détection et qualification de structures radioélectriques "
    "sur images aériennes — FRHACK 2026"
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
                    "du référentiel ANFR. Renommez l'image avec le "
                    "SUP_ID correspondant si elle provient du dataset "
                    "généré par le repo."
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
                        verbose=False,
                    )

                result = results[0]
                detections = extract_detections(result, image=image)

                st.session_state["last_image_name"] = uploaded_file.name
                st.session_state["last_detections"] = detections
                st.session_state["last_reference"] = reference
                st.session_state["last_result"] = result

                annotated = annotate_image(
                    image,
                    result,
                    match_radius,
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
                    matches = (
                        int(
                            (
                                detections["distance_centre_m"]
                                <= match_radius
                            ).sum()
                        )
                        if not detections.empty
                        else 0
                    )
                    st.metric(
                        "Proches du support ANFR",
                        matches,
                    )

                st.image(
                    annotated,
                    caption=(
                        "Vert = détection dans le rayon de "
                        "correspondance ANFR ; rouge = candidate éloignée. "
                        "Jaune = position théorique du support ANFR."
                    ),
                    use_container_width=True,
                )

                if detections.empty:
                    st.warning(
                        "Aucune structure détectée au-dessus du seuil "
                        "de confiance."
                    )
                else:
                    st.dataframe(
                        detections,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.download_button(
                        "⬇️ Télécharger les détections JSON",
                        data=make_json_download(detections),
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
        "La comparaison exploite le fait que les images générées "
        "par le pipeline du repo sont centrées sur le support ANFR."
    )

    detections = st.session_state.get("last_detections")
    reference = st.session_state.get("last_reference")
    image_name = st.session_state.get("last_image_name")

    if detections is None:
        st.info(
            "Lancez d'abord une détection dans l'onglet "
            "**🤖 Détection IA**."
        )
    else:
        comparison = compare_detections_to_reference(
            detections,
            reference,
            match_radius,
        )

        total = len(detections)
        matched = (
            int(comparison["correspondance"].sum())
            if "correspondance" in comparison.columns
            else 0
        )
        anomalies = total - matched

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Détections IA", total)

        with col2:
            st.metric("Correspondances", matched)

        with col3:
            st.metric("Candidates / anomalies", anomalies)

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

            if anomalies > 0:
                st.warning(
                    "🟠 Certaines détections sont éloignées du support "
                    "ANFR attendu. Elles constituent des candidats à "
                    "examiner, pas des preuves de sites non déclarés."
                )

        st.info(
            "⚠️ Cette comparaison est une qualification de candidat. "
            "Une absence de correspondance peut venir d'un faux positif, "
            "d'un décalage géographique, d'une différence de date, "
            "d'une installation récente/démontée ou d'une erreur "
            "de classification."
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

    data_root = APP_DIR / "data"
    yaml_candidates = [
        data_root / "antennes.yaml",
        data_root / "data.yaml",
        APP_DIR / "antennes.yaml",
        APP_DIR / "data.yaml",
        APP_DIR / "code" / "data" / "antennes.yaml",
        APP_DIR / "code" / "data" / "data.yaml",
    ]

    yaml_path = next((p for p in yaml_candidates if p.exists()), None)
    holdout_candidates = [
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
