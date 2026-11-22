"""Affichage Streamlit des resultats d'un fichier CSV : figures expliquees pour non-specialistes."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

from obesity import config as cfg
from obesity.consistency import bmi
from obesity.evaluate import accuracy_pm1
from obesity.ui.compare import weight_comparison
from obesity.ui.text import fr_class

ORDER = cfg.load_config()["class_order"]
TARGET = cfg.load_config()["target"]
COLORS = ["#4C9BE8", "#52B788", "#F2C14E", "#F29E4C", "#E76F51", "#D1495B", "#8E3B46"]


def predict_frame(clf, reg, X: pd.DataFrame) -> pd.DataFrame:
    """Predictions des deux modeles champions pour chaque personne."""
    out = pd.DataFrame(index=X.index)
    out["classe"] = clf.predict(X[list(clf.feature_names_in_)])
    if hasattr(clf, "predict_proba"):
        out["confiance"] = clf.predict_proba(X[list(clf.feature_names_in_)]).max(axis=1)
    out["poids"] = reg.predict(X[list(reg.feature_names_in_)])
    out["imc"] = bmi(out["poids"], X["Height"])
    return out


def _show(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


def render(res: pd.DataFrame, truth: pd.Series | None, true_weight: pd.Series | None = None) -> None:
    """Resume, figures et explications."""
    n = len(res)
    top = res["classe"].value_counts().idxmax()
    c1, c2, c3 = st.columns(3)
    c1.metric("Personnes analysées", n)
    c2.metric("Niveau le plus fréquent", fr_class(top))
    c3.metric("Poids estimé moyen", f"{res['poids'].mean():.0f} kg")

    st.markdown("### 1. Dans quel niveau le modèle place-t-il chaque personne ?")
    counts = res["classe"].value_counts().reindex(ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh([fr_class(c) for c in ORDER], counts.values, color=COLORS)
    ax.invert_yaxis()
    ax.set(xlabel="Nombre de personnes", title="Niveau d'obésité prédit")
    for i, v in enumerate(counts.values):
        ax.text(v, i, f" {int(v)}", va="center")
    _show(fig)
    st.info("**Comment lire ce graphique :** chaque barre compte les personnes du fichier que le "
            "modèle range dans ce niveau, de « poids insuffisant » (en haut) à « obésité type III » "
            "(en bas). Le modèle se base uniquement sur les habitudes de vie (alimentation, sport, "
            "âge…), sans connaître ni le poids ni la taille.")

    if "confiance" in res:
        st.markdown("### 2. Le modèle est-il sûr de lui ?")
        fig, ax = plt.subplots(figsize=(8, 3.4))
        ax.hist(res["confiance"] * 100, bins=np.arange(0, 105, 5), color="#4C9BE8", edgecolor="white")
        ax.set(xlabel="Confiance du modèle (%)", ylabel="Nombre de personnes",
               title="Niveau de confiance des prédictions")
        _show(fig)
        st.info("**Comment lire ce graphique :** pour chaque personne, le modèle donne un "
                "pourcentage de confiance dans sa réponse. Près de 100 %, il est très sûr ; vers "
                "30-50 %, il hésite entre plusieurs niveaux. Une confiance élevée ne garantit pas "
                "que la réponse soit juste, c'est seulement un indicateur.")

    st.markdown("### 3. Quel poids le modèle estime-t-il ?")
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.hist(res["poids"], bins=20, color="#52B788", edgecolor="white")
    ax.set(xlabel="Poids estimé (kg)", ylabel="Nombre de personnes",
           title="Poids estimé des personnes")
    _show(fig)
    st.info("**Comment lire ce graphique :** un second modèle estime le poids de chaque personne à "
            "partir de ses habitudes et de sa taille. **Attention :** il se trompe en moyenne "
            "d'environ 10 kg sur le jeu de test, et d'environ 15 kg sur des profils réellement collectés "
            "(les lignes synthétiques sont plus faciles). C'est une estimation, à ne jamais prendre pour une mesure.")

    wc = weight_comparison(res["poids"], true_weight) if true_weight is not None else None
    if wc is not None and wc["n"] > 0:
        st.markdown("#### Comparaison avec le vrai poids")
        a, b, c = st.columns(3)
        a.metric("Erreur moyenne", f"{wc['mae']:.1f} kg")
        b.metric("Estimation à ±10 kg", f"{wc['within10']:.0%}")
        c.metric("Part de variation expliquée (R²)", f"{wc['r2']:.2f}")
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        lo, hi = float(min(wc["real"].min(), wc["pred"].min())), float(max(wc["real"].max(), wc["pred"].max()))
        ax[0].scatter(wc["real"], wc["pred"], s=10, alpha=0.5, color="#4C9BE8")
        ax[0].plot([lo, hi], [lo, hi], color="#D1495B", lw=1.5, label="estimation parfaite")
        ax[0].set(xlabel="Vrai poids (kg)", ylabel="Poids estimé (kg)", title="Estimé contre réel")
        ax[0].legend()
        ax[1].hist(wc["error"], bins=25, color="#52B788", edgecolor="white")
        ax[1].axvline(0, color="#D1495B", lw=1.5)
        ax[1].set(xlabel="Estimé − réel (kg)", ylabel="Nombre de personnes", title="Erreurs d'estimation")
        fig.tight_layout()
        _show(fig)
        st.info("**Comment lire ces graphiques :** à gauche, chaque point est une personne. Plus il est "
                "proche de la ligne rouge, plus l'estimation est juste ; au-dessus, le modèle "
                "surestime, en dessous il sous-estime. À droite, la répartition des erreurs : "
                "centrée sur 0 mais étalée, avec des erreurs de 20 kg ou plus pour certaines "
                "personnes. Le modèle ne connaît ni le poids ni l'IMC, seulement la taille et les "
                "habitudes de vie : il ne peut donc donner qu'un ordre de grandeur.")
    elif true_weight is None:
        st.caption("Votre fichier ne contient pas la colonne `Weight` (vrai poids) : impossible de "
                   "comparer le poids estimé au poids réel. Ajoutez-la pour obtenir cette comparaison.")

    if truth is not None:
        st.markdown("### 4. Le modèle a-t-il raison ? (comparaison avec les vraies réponses)")
        t = truth.loc[res.index]
        acc = float((res["classe"] == t).mean())
        acc1 = accuracy_pm1(t, res["classe"])
        a, b = st.columns(2)
        a.metric("Bonnes réponses exactes", f"{acc:.0%}")
        b.metric("Bonne réponse à un niveau près", f"{acc1:.0%}")
        labels = [c for c in ORDER if c in set(t) | set(res["classe"])]
        cm = confusion_matrix(t, res["classe"], labels=labels)
        cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(7, 5.5))
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)), [fr_class(c) for c in labels], rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), [fr_class(c) for c in labels])
        ax.set(xlabel="Niveau prédit par le modèle", ylabel="Vrai niveau",
               title="Prédictions comparées à la réalité")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{cmn[i, j]:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if cmn[i, j] > 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
        _show(fig)
        st.info("**Comment lire ce graphique :** chaque ligne est un vrai niveau, chaque colonne le "
                "niveau prédit. Sur une ligne, les pourcentages indiquent où le modèle range ces "
                "personnes. Plus la **diagonale** (de haut gauche à bas droite) est foncée, mieux le "
                "modèle réussit. Les erreurs tombent surtout dans les niveaux voisins.")
    else:
        st.caption("Votre fichier ne contient pas la colonne `NObeyesdad` (vrai niveau) : impossible "
                   "de vérifier si le modèle a raison. Ajoutez-la pour obtenir cette comparaison.")

    st.markdown("### Aperçu des 20 premières personnes")
    show = pd.DataFrame({
        "Niveau prédit": res["classe"].map(fr_class),
        "Confiance": (res["confiance"] * 100).round(0).astype(int).astype(str) + " %"
        if "confiance" in res else "",
        "Poids estimé": res["poids"].round(0).astype(int).astype(str) + " kg",
        "IMC estimé": res["imc"].map(lambda v: f"{v:.1f}"),
    })
    if wc is not None and wc["n"] > 0:
        real = pd.to_numeric(true_weight.loc[res.index], errors="coerce")
        show["Vrai poids"] = real.round(0).map(lambda v: "" if pd.isna(v) else f"{int(v)} kg")
        show["Écart"] = (res["poids"] - real).round(0).map(lambda v: "" if pd.isna(v) else f"{int(v):+d} kg")
    show = show.head(20)
    st.table(show)
    st.caption("L'IMC (indice de masse corporelle) = poids ÷ taille². En dessous de 18,5 : poids "
               "insuffisant ; 18,5 à 25 : normal ; 25 à 30 : surpoids ; 30 et plus : obésité. "
               "Outil pédagogique : ce n'est pas un avis médical.")
