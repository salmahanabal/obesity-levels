"""Textes en francais pour l'interface (sans dependance a Streamlit) : libelles et explications."""
from __future__ import annotations

import pandas as pd

CLASS_FR = {
    "Insufficient_Weight": "Poids insuffisant",
    "Normal_Weight": "Poids normal",
    "Overweight_Level_I": "Surpoids niveau I",
    "Overweight_Level_II": "Surpoids niveau II",
    "Obesity_Type_I": "Obésité type I",
    "Obesity_Type_II": "Obésité type II",
    "Obesity_Type_III": "Obésité type III",
}

YES_NO = {"yes": "Oui", "no": "Non"}
FREQ = {"no": "Jamais", "Sometimes": "Parfois", "Frequently": "Souvent", "Always": "Toujours"}

# nom -> (libelle, explication pour non-specialistes, groupe)
VARIABLES = {
    "Gender": ("Genre", "Femme ou homme.", "Profil"),
    "Age": ("Âge", "Âge en années (14 à 65 ans).", "Profil"),
    "Height": (
        "Taille (m)",
        ("Taille en mètres. Elle sert à estimer le poids et l'IMC, mais pas à prédire le niveau "
        "d'obésité (sinon on donnerait la réponse au modèle)."),
        "Profil",
    ),
    "family_history_with_overweight": (
        "Antécédent familial de surpoids",
        "Un membre de la famille a-t-il (ou a-t-il eu) un surpoids ?",
        "Profil",
    ),
    "FAVC": ("Aliments très caloriques", ("Mange-t-on souvent des aliments très caloriques "
             "(fast-food, sucreries, fritures) ?"), "Alimentation"),
    "FCVC": ("Légumes", ("À quelle fréquence mange-t-on des légumes : 1 = jamais, 2 = parfois, "
             "3 = à chaque repas."), "Alimentation"),
    "NCP": ("Repas principaux par jour", "Nombre de vrais repas par jour (1 à 4).", "Alimentation"),
    "CAEC": ("Grignotage", "Mange-t-on entre les repas ? Jamais, parfois, souvent ou toujours.",
             "Alimentation"),
    "CH2O": ("Eau", ("Quantité d'eau bue chaque jour : 1 = peu (moins d'1 litre), 2 = moyen "
             "(1 à 2 litres), 3 = beaucoup (plus de 2 litres)."), "Alimentation"),
    "SCC": ("Suivi des calories", "Surveille-t-on les calories que l'on mange ?", "Alimentation"),
    "CALC": ("Alcool", "Fréquence de consommation d'alcool : jamais, parfois ou souvent.",
             "Alimentation"),
    "FAF": ("Activité physique", ("Sport ou activité physique : 0 = aucune, 1 = 1 à 2 jours par "
            "semaine, 2 = 2 à 4 jours, 3 = 4 à 5 jours."), "Mode de vie"),
    "TUE": ("Temps d'écran", ("Temps passé sur téléphone, télé ou ordinateur : 0 = 0 à 2 h par "
            "jour, 1 = 3 à 5 h, 2 = plus de 5 h."), "Mode de vie"),
    "SMOKE": ("Fumeur", "Fume-t-on ?", "Mode de vie"),
    "MTRANS": ("Transport habituel", "Moyen de transport le plus utilisé.", "Mode de vie"),
}

MTRANS_FR = {
    "Automobile": "Voiture",
    "Motorbike": "Moto",
    "Bike": "Vélo",
    "Public_Transportation": "Transports en commun",
    "Walking": "Marche",
}
SCALE_FR = {
    "FCVC": {1: "1 · Jamais", 2: "2 · Parfois", 3: "3 · Toujours"},
    "CH2O": {1: "1 · Peu", 2: "2 · Moyen", 3: "3 · Beaucoup"},
    "FAF": {0: "0 · Aucune", 1: "1 · 1-2 j/sem.", 2: "2 · 2-4 j/sem.", 3: "3 · 4-5 j/sem."},
    "TUE": {0: "0 · 0-2 h", 1: "1 · 3-5 h", 2: "2 · plus de 5 h"},
    "NCP": {1: "1 repas", 2: "2 repas", 3: "3 repas", 4: "4 repas"},
}
CATEGORY_FR = {
    "Gender": {"Female": "Femme", "Male": "Homme"},
    "family_history_with_overweight": YES_NO,
    "FAVC": YES_NO,
    "SMOKE": YES_NO,
    "SCC": YES_NO,
    "CAEC": FREQ,
    "CALC": FREQ,
    "MTRANS": MTRANS_FR,
}

ORDINAL_INT = ["FCVC", "NCP", "CH2O", "FAF", "TUE"]


def normalise(data: pd.DataFrame) -> pd.DataFrame:
    """Rend un CSV tolerant : arrondit les echelles entieres et fusionne CALC=Always."""
    d = data.copy()
    for c in ORDINAL_INT:
        if c in d:
            d[c] = d[c].round().astype(int)
    if "CALC" in d:
        d["CALC"] = d["CALC"].replace({"Always": "Frequently"})
    return d


def fr_class(label: str) -> str:
    return CLASS_FR.get(label, label)
