"""Interface de test : donnees nettoyees/pretraitees, test d'un fichier CSV et saisie manuelle.

Lancement : `make ui` (ou `streamlit run app/streamlit_app.py`).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from obesity import config as cfg
from obesity.analysis import load_models
from obesity.consistency import bmi
from obesity.data import load_clean
from obesity.preprocessing import feature_columns, load_split
from obesity.sampling import CHOICES, FLOAT_RANGES, INT_RANGES
from obesity.ui import results
from obesity.ui.text import CATEGORY_FR, SCALE_FR, VARIABLES, fr_class, normalise

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

tab1, tab2 = st.tabs(["1. Données", "2. Saisie manuelle"])

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
    st.caption("Le poids d'origine (`Weight`) est conservé dans les données nettoyées, d'entraînement et de "
               "test. Il n'est pas dans les données pré-traitées du pipeline A, qui n'utilise ni le poids "
               "ni la taille.")
    st.markdown("**Aperçu des données pré-traitées**")
    st.dataframe(X_pre.head(20), width="stretch")

    st.subheader("Tester un fichier CSV")
    st.caption("Téléversez un fichier CSV (par exemple le jeu de test téléchargé ci-dessus) : les "
               "modèles analysent chaque personne et les résultats s'affichent ici sous forme de "
               "graphiques expliqués. Colonnes attendues : celles décrites dans l'onglet « Saisie "
               "manuelle » (dont la taille `Height`). Facultatif : la colonne `NObeyesdad` (vrai "
               "niveau) et la colonne `Weight` (vrai poids d'origine du jeu de données) permettent de "
               "vérifier si les modèles ont raison. Le vrai poids n'est jamais donné aux modèles : il "
               "sert uniquement à la comparaison. Le fichier « Jeu de test (nettoyé) » contient les deux.")
    up = st.file_uploader("Fichier CSV", type="csv")
    if up is not None:
        data = pd.read_csv(up)
        missing = [c for c in feature_columns("regression") if c not in data.columns]
        if missing:
            st.error(f"Colonnes manquantes dans votre fichier : {missing}")
        else:
            data = normalise(data)
            truth = data[TARGET] if TARGET in data.columns else None
            true_weight = data[cfg.load_config()["weight_col"]] if cfg.load_config()["weight_col"] in data.columns else None
            results.render(results.predict_frame(clf, reg, data), truth, true_weight)

with tab2:
    st.subheader("Saisie manuelle : testez une personne")
    st.caption("Renseignez les 15 informations ci-dessous. Passez la souris sur le « ? » à côté de "
               "chaque champ pour son explication, ou ouvrez le guide juste en dessous.")

    with st.expander("Guide des variables : que signifie chaque information ?"):
        guide = pd.DataFrame(
            [(label, desc, group) for label, desc, group in VARIABLES.values()],
            columns=["Information", "Signification", "Catégorie"],
        )
        st.table(guide.set_index("Information"))

    vals: dict = {}
    for group in ("Profil", "Alimentation", "Mode de vie"):
        st.markdown(f"**{group}**")
        names = [k for k, v in VARIABLES.items() if v[2] == group]
        cols = st.columns(3)
        for i, name in enumerate(names):
            label, desc, _ = VARIABLES[name]
            box = cols[i % 3]
            if name in SCALE_FR:
                opts = list(SCALE_FR[name])
                lo, hi = INT_RANGES[name]
                vals[name] = box.select_slider(label, opts, value=opts[len(opts) // 2], help=desc,
                                               format_func=SCALE_FR[name].get)
            elif name in CHOICES:
                fr = CATEGORY_FR[name]
                opts = [o for o in CHOICES[name] if o != "Always" or name != "CALC"]
                vals[name] = box.selectbox(label, opts, format_func=fr.get, help=desc)
            else:
                lo, hi = FLOAT_RANGES[name]
                vals[name] = box.number_input(label, float(lo), float(hi), float(round((lo + hi) / 2, 2)),
                                              step=0.01 if name == "Height" else 1.0, help=desc)

    person = pd.DataFrame([vals])
    classe = clf.predict(person[list(clf.feature_names_in_)])[0]
    poids = float(reg.predict(person[list(reg.feature_names_in_)])[0])
    imc = float(bmi(poids, vals["Height"]))
    real_w = None
    if st.checkbox("Je connais le vrai poids de cette personne : comparer avec l'estimation"):
        real_w = st.number_input("Vrai poids (kg)", 30.0, 250.0, 70.0, step=1.0,
                                 help="Sert uniquement à la comparaison, jamais au calcul de la prédiction.")
    st.divider()
    st.markdown("### Résultat")
    r1, r2 = st.columns(2)
    r1.success(f"**Niveau prédit : {fr_class(classe)}**\n\nDéduit uniquement des habitudes de vie "
               "(sans le poids ni la taille).")
    r2.info(f"**Poids estimé : {poids:.0f} kg** (IMC {imc:.1f})\n\nÀ ± 18 kg environ : c'est une "
            "estimation grossière.")
    if real_w is not None:
        diff = poids - real_w
        d1, d2, d3 = st.columns(3)
        d1.metric("Poids estimé", f"{poids:.0f} kg")
        d2.metric("Vrai poids", f"{real_w:.0f} kg")
        d3.metric("Écart", f"{diff:+.0f} kg")
        st.caption("Une erreur de l'ordre de 18 kg est normale pour ce modèle : il ne connaît pas le poids, "
                   "seulement la taille et les habitudes de vie.")
    if hasattr(clf, "predict_proba"):
        p = pd.Series(clf.predict_proba(person[list(clf.feature_names_in_)])[0], index=clf.classes_)
        p = p.reindex(cfg.load_config()["class_order"]).fillna(0)
        fig, ax = plt.subplots(figsize=(8, 3.4))
        ax.barh([fr_class(c) for c in p.index], p.values * 100, color=results.COLORS)
        ax.invert_yaxis()
        ax.set(xlabel="Probabilité (%)", title="Le modèle hésite-t-il entre plusieurs niveaux ?")
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Chaque barre est la probabilité donnée à un niveau. Une barre nettement plus "
                   "longue que les autres = le modèle est sûr ; plusieurs barres proches = il hésite "
                   "(en général entre niveaux voisins). Outil pédagogique, pas un avis médical.")
