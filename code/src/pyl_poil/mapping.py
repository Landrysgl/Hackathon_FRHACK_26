from __future__ import annotations

from pathlib import Path

import folium
import pandas as pd

from .geo import cartoradio_url


def _map_center(candidates: pd.DataFrame, supports: pd.DataFrame | None) -> tuple[float, float]:
    """Centre d'abord la carte sur les candidats, sinon sur le catalogue de référence."""
    for frame in (candidates, supports):
        if frame is None or frame.empty:
            continue
        if {"latitude", "longitude"}.issubset(frame.columns):
            lat = pd.to_numeric(frame["latitude"], errors="coerce").dropna()
            lon = pd.to_numeric(frame["longitude"], errors="coerce").dropna()
            if len(lat) and len(lon):
                return float(lat.mean()), float(lon.mean())
    return 48.8, 2.1


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "oui"}


def build_interactive_map(
    candidates: pd.DataFrame,
    output_path: str | Path,
    *,
    supports: pd.DataFrame | None = None,
    final_ids: set[str] | None = None,
    title: str = "Pyl-Poil - candidats détectés",
) -> Path:
    """Génère une carte HTML autonome avec couches ANFR, statuts et candidats finaux."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_ids = final_ids or set()

    fmap = folium.Map(
        location=_map_center(candidates, supports),
        zoom_start=13,
        control_scale=True,
        tiles="OpenStreetMap",
    )

    if supports is not None and not supports.empty:
        support_layer = folium.FeatureGroup(name="Supports ANFR connus", show=False)
        for row in supports.to_dict("records"):
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            popup = (
                f"<b>Support ANFR {row.get('SUP_ID', '')}</b><br>"
                f"Nature: {row.get('NAT_LB_NOM', '')}<br>"
                f"<a href='{cartoradio_url(lat, lon)}' target='_blank'>Ouvrir Cartoradio</a>"
            )
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                color="#2f6bff",
                fill=True,
                fill_opacity=0.65,
                popup=folium.Popup(popup, max_width=350),
            ).add_to(support_layer)
        support_layer.add_to(fmap)

    layers = {
        "plausible": folium.FeatureGroup(name="Candidats plausibles", show=True),
        "faux_positif": folium.FeatureGroup(name="Faux positifs revus", show=False),
        "incertain": folium.FeatureGroup(name="Candidats incertains", show=True),
        "non_revu": folium.FeatureGroup(name="Candidats non revus", show=True),
        "final": folium.FeatureGroup(name="3 candidats finaux", show=True),
    }
    colors = {
        "plausible": "#ff8c00",
        "faux_positif": "#777777",
        "incertain": "#8a2be2",
        "non_revu": "#ff8c00",
        "final": "#d00000",
    }

    id_column = "unique_candidate_id" if "unique_candidate_id" in candidates.columns else "candidate_id"
    has_selection_column = "selection_finale" in candidates.columns

    for row in candidates.to_dict("records"):
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        candidate_id = str(row.get(id_column, ""))
        is_final = candidate_id in final_ids or _as_bool(row.get("selection_finale", False))
        status = str(row.get("review_status", "non_revu") or "non_revu").strip()
        if status not in {"plausible", "faux_positif", "incertain"}:
            status = "non_revu"
        layer_key = "final" if is_final else status
        support_type = str(row.get("support_type", "") or "")
        distance = float(row.get("nearest_ANFR_distance_m", 0.0))
        confidence = float(row.get("confidence", 0.0))
        popup = (
            f"<b>{candidate_id or 'Candidat'}</b><br>"
            f"Statut: {status}<br>"
            f"Type visuel: {support_type}<br>"
            f"Confiance YOLO: {confidence:.3f}<br>"
            f"Distance ANFR: {distance:.1f} m<br>"
            f"Coordonnées: {lat:.6f}, {lon:.6f}<br>"
            f"<a href='{cartoradio_url(lat, lon)}' target='_blank'>Ouvrir Cartoradio</a><br>"
            "<small>Un candidat n'est pas une preuve de site radioélectrique non déclaré.</small>"
        )
        folium.CircleMarker(
            location=[lat, lon],
            radius=8 if is_final else 5,
            color=colors[layer_key],
            fill=True,
            fill_opacity=0.85,
            weight=3 if is_final else 2,
            popup=folium.Popup(popup, max_width=430),
            tooltip=f"{candidate_id} - {support_type or status}",
        ).add_to(layers[layer_key])

    for key in ("plausible", "faux_positif", "incertain", "non_revu"):
        layers[key].add_to(fmap)
    if final_ids or has_selection_column:
        layers["final"].add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    title_html = (
        "<div style='position: fixed; top: 8px; left: 50px; z-index:9999; "
        "background:white; padding:8px 12px; border:1px solid #bbb; border-radius:4px'>"
        f"<b>{title}</b></div>"
    )
    fmap.get_root().html.add_child(folium.Element(title_html))
    fmap.save(str(output_path))
    return output_path
