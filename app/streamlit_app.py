"""Interface de test : telechargement des donnees nettoyees/pretraitees et essais sur valeurs aleatoires.

Lancement : `make ui` (ou `streamlit run app/streamlit_app.py`).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from obesity import config as cfg
from obesity.analysis import load_models
from obesity.consistency import bmi, bmi_to_class
from obesity.data import load_clean
from obesity.preprocessing import feature_columns, load_split
from obesity.sampling import CHOICES, FLOAT_RANGES, INT_RANGES, random_samples

st.set_page_config(page_title="obesity-levels", layout="wide")
TARGET = cfg.load_config()["target"]


@st.cache_resource
def models():
    return load_models()


@st.cache_data
def clean_df() -> pd.DataFrame:
    return load_clean()


def csv(df: pd.DataFrame, index: bool = False) -> bytes:
    return df.to_csv(index=index).encode("utf-8")


def predict(clf, reg, X: pd.DataFrame) -> pd.DataFrame:
    """Predictions des deux modeles champions sur des individus (colonnes du pipeline B)."""
    out = X.copy()
    out["classe_predite (A)"] = clf.predict(X[list(clf.feature_names_in_)])
    if hasattr(clf, "predict_proba"):
        out["confiance (A)"] = clf.predict_proba(X[list(clf.feature_names_in_)]).max(axis=1).round(3)
    w = reg.predict(X[list(reg.feature_names_in_)])
    out["poids_predit_kg (B)"] = w.round(1)
    out["IMC_predit"] = bmi(w, X["Height"]).round(1)
    out["classe_derivee (C)"] = bmi_to_class(out["IMC_predit"])
    return out


st.title("Niveaux d'obésité — interface de test")
try:
    clf, reg = models()
except FileNotFoundError:
    st.error("Modèles introuvables. Lancez d'abord `make all` (ou `make train-clf train-reg export`).")
    st.stop()

df = clean_df()
train, test = load_split(df)
prep = clf.named_steps["prep"]
cols_a = feature_columns("classification")

tab1, tab2, tab3 = st.tabs(["1. Données", "2. Test sur valeurs aléatoires", "3. Saisie manuelle"])

with tab1:
    st.subheader("Données nettoyées et pré-traitées")
    st.caption("Nettoyage : doublons supprimés, ordinales arrondies, `CALC=Always` fusionnée. "
               "Pré-traitement : encodage et standardisation du pipeline A, appris sur le train.")
    X_pre = prep.transform(df[cols_a]).assign(**{TARGET: df[TARGET].values})
    c1, c2 = st.columns(2)
    c1.metric("Lignes nettoyées", len(df))
    c2.metric("Colonnes pré-traitées (pipeline A)", X_pre.shape[1] - 1)
    b1, b2, b3, b4 = st.columns(4)
    b1.download_button("Données nettoyées (CSV)", csv(df), "obesity_clean.csv", "text/csv")
    b2.download_button("Données pré-traitées (CSV)", csv(X_pre), "obesity_preprocessed.csv", "text/csv")
    b3.download_button("Jeu de test (nettoyé)", csv(test), "obesity_test.csv", "text/csv")
    b4.download_button("Jeu d'entraînement (nettoyé)", csv(train), "obesity_train.csv", "text/csv")
    st.markdown("**Aperçu des données pré-traitées**")
    st.dataframe(X_pre.head(20), width="stretch")

    st.subheader("Tester un fichier CSV")
    st.caption("Téléversez un CSV (nettoyé) : les modèles champions prédisent chaque ligne. "
               "Colonnes attendues : celles du pipeline B (avec `Height`).")
    up = st.file_uploader("CSV", type="csv")
    if up is not None:
        data = pd.read_csv(up)
        missing = [c for c in feature_columns("regression") if c not in data.columns]
        if missing:
            st.error(f"Colonnes manquantes : {missing}")
        else:
            res = predict(clf, reg, data)
            st.dataframe(res, width="stretch")
            st.download_button("Télécharger les prédictions", csv(res), "predictions.csv", "text/csv")

with tab2:
    st.subheader("Valeurs aléatoires")
    st.caption("Individus tirés uniformément dans les plages valides. Classifieur (A) : habitudes de vie "
               "uniquement. Régression (B) : poids, puis classe dérivée de l'IMC (C).")
    n = st.slider("Nombre d'individus", 1, 200, 10)
    seed = st.number_input("Graine (reproductibilité)", value=42, step=1)
    if st.button("Générer et prédire", type="primary") or "rand" not in st.session_state:
        st.session_state["rand"] = predict(clf, reg, random_samples(n, int(seed)))
    res = st.session_state["rand"]
    st.dataframe(res, width="stretch")
    st.bar_chart(res["classe_predite (A)"].value_counts())
    st.download_button("Télécharger", csv(res), "random_predictions.csv", "text/csv")

    st.subheader("Individus réels du jeu de test (vérité terrain connue)")
    k = st.slider("Nombre de lignes du test", 5, 100, 20)
    s = test.sample(k, random_state=int(seed))
    r = predict(clf, reg, s.drop(columns=[TARGET, "Weight"]))
    r["classe_reelle"] = s[TARGET].values
    r["poids_reel_kg"] = s["Weight"].values
    r["A correcte"] = r["classe_predite (A)"] == r["classe_reelle"]
    r["C correcte"] = r["classe_derivee (C)"] == r["classe_reelle"]
    st.dataframe(r, width="stretch")
    a, c = st.columns(2)
    a.metric("Accuracy A (échantillon)", f"{r['A correcte'].mean():.0%}")
    c.metric("Accuracy C (échantillon)", f"{r['C correcte'].mean():.0%}")

with tab3:
    st.subheader("Saisie manuelle")
    vals: dict = {}
    cols = st.columns(4)
    items = [*CHOICES, *INT_RANGES, *FLOAT_RANGES]
    for i, name in enumerate(items):
        box = cols[i % 4]
        if name in CHOICES:
            vals[name] = box.selectbox(name, CHOICES[name])
        elif name in INT_RANGES:
            lo, hi = INT_RANGES[name]
            vals[name] = box.slider(name, lo, hi, (lo + hi) // 2)
        else:
            lo, hi = FLOAT_RANGES[name]
            vals[name] = box.number_input(name, float(lo), float(hi), float((lo + hi) / 2),
                                          step=0.01 if name == "Height" else 0.5)
    out = predict(clf, reg, pd.DataFrame([vals]))
    st.success(f"Classe prédite (A) : **{out['classe_predite (A)'][0]}**")
    st.info(f"Poids prédit (B) : **{out['poids_predit_kg (B)'][0]} kg** — IMC {out['IMC_predit'][0]} "
            f"→ classe dérivée (C) : **{out['classe_derivee (C)'][0]}**")
    if hasattr(clf, "predict_proba"):
        p = pd.Series(clf.predict_proba(pd.DataFrame([vals])[list(clf.feature_names_in_)])[0],
                      index=clf.classes_)
        st.bar_chart(p)
