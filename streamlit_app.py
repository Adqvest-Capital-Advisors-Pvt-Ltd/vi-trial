import base64
import io
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from dotenv import load_dotenv


# -------------------------
# Configuration & Constants
# -------------------------
API_BASE_URL: str = "https://api-hub.ventureintelligence.com/vendor/cfs"
FILINGS_DOWNLOAD_PATH: str = "/filings-download/"
FILINGS_VIEW_PATH: str = "/filings-view/"  # not strictly required, kept for completeness
FINANCIALS_PATH: str = "/financials/"


def load_settings() -> Tuple[Optional[str], Optional[str]]:
    """Load API settings from environment.

    Returns a tuple of (api_key, csrf_token). If not found, returns (None, None) for the missing values.
    """
    load_dotenv(override=False)
    api_key: Optional[str] = os.getenv("API_KEY")
    # Support both custom and generic env var names for CSRF token
    csrf_token: Optional[str] = os.getenv("VI_CSRF_TOKEN") or os.getenv("X_CSRF_TOKEN")
    return api_key, csrf_token


def make_headers(api_key: str, csrf_token: Optional[str]) -> Dict[str, str]:
    headers: Dict[str, str] = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "x-api-key": api_key,
    }
    if csrf_token:
        headers["X-CSRFTOKEN"] = csrf_token
    return headers


def post_json(url: str, payload: Dict[str, str], headers: Dict[str, str]) -> Tuple[Optional[Dict], Optional[str]]:
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=45)
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}: {resp.text[:500]}"
        return resp.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


@st.cache_data(show_spinner=False)
def fetch_filings(company_name: str, api_key: str, csrf_token: Optional[str]) -> Tuple[List[Dict[str, str]], Optional[str]]:
    url = f"{API_BASE_URL}{FILINGS_DOWNLOAD_PATH}"
    headers = make_headers(api_key, csrf_token)
    data, err = post_json(url, {"company_name": company_name}, headers)
    if err:
        return [], err
    filings: List[Dict[str, str]] = data.get("filings", []) if isinstance(data, dict) else []
    # Normalize and sort by name
    normalized: List[Dict[str, str]] = []
    for item in filings:
        name = item.get("name") if isinstance(item, dict) else None
        url_value = item.get("url") if isinstance(item, dict) else None
        if name and url_value:
            normalized.append({"name": name, "url": url_value})
    normalized.sort(key=lambda x: x["name"].lower())
    return normalized, None


@st.cache_data(show_spinner=False)
def fetch_financials(company_name: str, api_key: str, csrf_token: Optional[str]) -> Tuple[Optional[Dict], Optional[str]]:
    url = f"{API_BASE_URL}{FINANCIALS_PATH}"
    headers = make_headers(api_key, csrf_token)
    data, err = post_json(url, {"company_name": company_name}, headers)
    if err:
        return None, err
    return data, None


@st.cache_data(show_spinner=False)
def download_pdf_bytes(file_url: str) -> Tuple[Optional[bytes], Optional[str]]:
    try:
        resp = requests.get(file_url, timeout=60)
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code} while downloading PDF"
        return resp.content, None
    except requests.RequestException as exc:
        return None, str(exc)


def render_pdf_viewer(pdf_bytes: bytes, *, scroll_to_page: Optional[int] = None, height: int = 900) -> None:
    """Render a PDF using streamlit-pdf-viewer with optional page scrolling.

    Falls back gracefully if parameters are out of expected range.
    """
    page_param: Optional[int] = None
    if isinstance(scroll_to_page, int) and scroll_to_page > 0:
        page_param = scroll_to_page

    # Display using pdf.js-based component (supports smooth scrolling and text layer)
    pdf_viewer(
        pdf_bytes,
        width="100%",
        height=height,
        render_text=True,
        show_page_separator=True,
        pages_vertical_spacing=6,
        scroll_to_page=page_param,
    )


def build_financials_df(records: List[Dict], fin_type: str) -> pd.DataFrame:
    """Build a transposed financials table, preserving vendor key order.

    Rows are metric names (in the order encountered from vendor JSON),
    columns are fiscal years (ascending): fy11, fy12, ..., fy22.
    """
    filtered: List[Dict] = [r for r in (records or []) if r.get("fin_type") == fin_type]
    if not filtered:
        return pd.DataFrame()

    # Preserve key order as provided by vendor across all records.
    metrics_order: List[str] = []
    for record in filtered:
        for key in record.keys():
            if key in ("fy", "fin_type"):
                continue
            if key not in metrics_order:
                metrics_order.append(key)

    # Collect columns (fy labels) in ascending numeric order
    fy_labels: List[str] = []
    fy_to_record: Dict[str, Dict] = {}
    for r in filtered:
        fy_val = str(r.get("fy")) if r.get("fy") is not None else ""
        if fy_val not in fy_to_record:
            fy_to_record[fy_val] = r
    # Sort by numeric value when possible
    def _fy_sort_key(val: str) -> Tuple[int, str]:
        try:
            return (int(val), val)
        except Exception:
            return (10**9, val)

    for fy in sorted(fy_to_record.keys(), key=_fy_sort_key):
        label = f"fy{fy}" if not str(fy).lower().startswith("fy") else str(fy)
        fy_labels.append(label)

    # Build a dict-of-series with index as metrics_order
    data_by_fy: Dict[str, List[Optional[float]]] = {}
    for fy, rec in fy_to_record.items():
        col_label = f"fy{fy}" if not str(fy).lower().startswith("fy") else str(fy)
        # Align values to metrics_order
        column_values: List[Optional[float]] = [rec.get(metric) for metric in metrics_order]
        data_by_fy[col_label] = column_values

    df = pd.DataFrame(data_by_fy, index=metrics_order)
    return df


def _scale_value_to_millions(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return round(value / 1_000_000, 1)
    # Attempt to parse numeric strings
    if isinstance(value, str):
        try:
            num = float(value)
            return round(num / 1_000_000, 1)
        except Exception:
            return value
    return value


def scale_df_to_millions(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    return df.applymap(_scale_value_to_millions)


def render_filings_tab(company_name: str, api_key: str, csrf_token: Optional[str]) -> None:
    st.subheader("Filings download and PDF view")
    if not company_name:
        st.info("Enter a company name to see filings.")
        return

    with st.spinner("Fetching filings..."):
        filings, err = fetch_filings(company_name, api_key, csrf_token)

    if err:
        st.error(f"Failed to fetch filings: {err}")
        return

    if not filings:
        st.warning("No filings found.")
        return

    names = [f["name"] for f in filings]
    selection = st.selectbox("Select a filing to view:", names, index=0)
    selected = next((f for f in filings if f["name"] == selection), None)

    if not selected:
        st.warning("Please select a filing.")
        return

    pdf_url = selected["url"]
    # Controls row
    left, mid, right = st.columns([3, 1.5, 1])
    with left:
        st.caption("PDF Preview")
    with mid:
        go_to_page: int = st.number_input(
            "Go to page",
            min_value=1,
            step=1,
            value=st.session_state.get("pdf_scroll_to_page", 1),
            help="Enter a page number to jump to",
        )
    with right:
        refresh = st.button("Refresh PDF")
        jump = st.button("Go")

    # Fetch and render PDF
    with st.spinner("Loading PDF..."):
        pdf_bytes, pdf_err = download_pdf_bytes(pdf_url)

    if pdf_err or not pdf_bytes:
        st.error(f"Unable to load PDF: {pdf_err or 'Empty content'}")
        return

    if jump:
        st.session_state["pdf_scroll_to_page"] = int(go_to_page) if go_to_page and go_to_page > 0 else 1

    render_pdf_viewer(pdf_bytes, scroll_to_page=st.session_state.get("pdf_scroll_to_page", 1))

    st.download_button(
        label="Download selected PDF",
        data=io.BytesIO(pdf_bytes),
        file_name=selection,
        mime="application/pdf",
        use_container_width=True,
    )


def render_financials_tab(company_name: str, api_key: str, csrf_token: Optional[str]) -> None:
    st.subheader("Financials view")
    if not company_name:
        st.info("Enter a company name to see financials.")
        return

    with st.spinner("Fetching financials..."):
        data, err = fetch_financials(company_name, api_key, csrf_token)

    if err:
        st.error(f"Failed to fetch financials: {err}")
        return

    if not data or not isinstance(data, dict):
        st.warning("No financials data available.")
        return

    results: Dict = (data or {}).get("results", {})
    profit_loss = results.get("profit_loss", [])
    balance_sheet = results.get("balance_sheet", [])
    cash_flow = results.get("cash_flow", [])

    fin_type = st.radio(
        "Level of consolidation",
        options=["Consolidated", "Standalone"],
        index=0,
        horizontal=True,
    )

    tabs = st.tabs(["Profit & Loss", "Balance Sheet", "Cash Flow"])

    with tabs[0]:
        df_pl = build_financials_df(profit_loss, fin_type)
        df_pl = scale_df_to_millions(df_pl)
        if df_pl.empty:
            st.info("No Profit & Loss data for this level.")
        else:
            st.caption("All values in millions")
            st.dataframe(df_pl, use_container_width=True)
            st.download_button(
                label="Download Profit & Loss (CSV)",
                data=df_pl.to_csv(index=True).encode("utf-8"),
                file_name=f"{company_name.replace(' ', '_')}_profit_loss_{fin_type}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with tabs[1]:
        df_bs = build_financials_df(balance_sheet, fin_type)
        df_bs = scale_df_to_millions(df_bs)
        if df_bs.empty:
            st.info("No Balance Sheet data for this level.")
        else:
            st.caption("All values in millions")
            st.dataframe(df_bs, use_container_width=True)
            st.download_button(
                label="Download Balance Sheet (CSV)",
                data=df_bs.to_csv(index=True).encode("utf-8"),
                file_name=f"{company_name.replace(' ', '_')}_balance_sheet_{fin_type}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with tabs[2]:
        df_cf = build_financials_df(cash_flow, fin_type)
        df_cf = scale_df_to_millions(df_cf)
        if df_cf.empty:
            st.info("No Cash Flow data for this level.")
        else:
            st.caption("All values in millions")
            st.dataframe(df_cf, use_container_width=True)
            st.download_button(
                label="Download Cash Flow (CSV)",
                data=df_cf.to_csv(index=True).encode("utf-8"),
                file_name=f"{company_name.replace(' ', '_')}_cash_flow_{fin_type}.csv",
                mime="text/csv",
                use_container_width=True,
            )


def render_profile_sidebar(profile: Dict) -> None:
    if not profile:
        return

    short_name = profile.get("short_name") or profile.get("full_name") or ""
    full_name = profile.get("full_name") or ""
    st.subheader(short_name)
    if full_name and full_name != short_name:
        st.caption(full_name)

    fields = [
        ("Industry", "industry"),
        ("Sector", "sector"),
        ("Company status", "company_status"),
        ("Listing", "listing_status"),
        ("Transacted", "transacted_status"),
        ("Incorp. year", "incorp_year"),
        ("CIN", "cin"),
        ("ROC code", "roc_code"),
        ("ROC reg no.", "roc_reg_number"),
        ("Authorised capital", "authorised_capital"),
        ("Paid-up capital", "paid_capital"),
        ("Phone", "phone"),
        ("Email", "email"),
    ]

    for label, key in fields:
        value = profile.get(key)
        if value in (None, "", 0):
            continue
        st.markdown(f"**{label}**: {value}")

    address = profile.get("address")
    if address:
        st.markdown(f"**Address**: {address}")

    links = []
    if profile.get("website"):
        links.append(f"[Website]({profile['website']})")
    if profile.get("linkedin"):
        links.append(f"[LinkedIn]({profile['linkedin']})")
    if links:
        st.markdown(" • ".join(links))


def main() -> None:
    st.set_page_config(page_title="Company Financials Explorer", layout="wide")
    st.title("Company Financials Explorer")
    st.caption("Search, view filings, and explore financial statements.")

    env_api_key, env_csrf = load_settings()

    with st.sidebar:
        st.header("Settings")
        st.markdown("The API Key is read from `.env` by default. You can override below.")
        default_company = "THINK & LEARN PRIVATE LIMITED"
        st.caption("Enter a company name as used in the APIs.")
        with st.form("company_form"):
            api_key_input = st.text_input("API Key", type="password", value=st.session_state.get("api_key", env_api_key) or "")
            csrf_input = st.text_input("CSRF Token (optional)", type="password", value=st.session_state.get("csrf_token", env_csrf) or "")
            company_name_input = st.text_input("Company name", value=st.session_state.get("active_company_name") or default_company)
            submitted = st.form_submit_button("Submit")

        if submitted:
            st.session_state["api_key"] = (api_key_input or "").strip()
            st.session_state["csrf_token"] = (csrf_input or "").strip()
            st.session_state["active_company_name"] = (company_name_input or "").strip()

        api_key = st.session_state.get("api_key") or env_api_key
        csrf_token = st.session_state.get("csrf_token") or env_csrf
        active_company = st.session_state.get("active_company_name")

        st.markdown("---")
        if api_key and active_company:
            with st.spinner("Loading profile..."):
                data, _err = fetch_financials(active_company, api_key, csrf_token)
            profile = ((data or {}).get("results") or {}).get("profile")
            render_profile_sidebar(profile)

    if not (st.session_state.get("api_key") or env_api_key):
        st.error("API Key is required. Please set `API_KEY` in `.env` or provide it in the sidebar and click Submit.")
        st.stop()

    company_name = st.session_state.get("active_company_name")

    tab1, tab2 = st.tabs(["Filings & PDF View", "Financials View"])

    with tab1:
        render_filings_tab(company_name or "", st.session_state.get("api_key") or env_api_key, st.session_state.get("csrf_token") or env_csrf)

    with tab2:
        render_financials_tab(company_name or "", st.session_state.get("api_key") or env_api_key, st.session_state.get("csrf_token") or env_csrf)


if __name__ == "__main__":
    main()

