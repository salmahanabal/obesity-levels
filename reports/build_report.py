"""Genere reports/rapport_obesity_levels.pdf a partir des tableaux et figures du projet.

Usage : python reports/build_report.py   (apres `make all`)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
FIG, TAB = ROOT / "reports" / "figures", ROOT / "reports" / "tables"
import os

OUT = Path(os.environ.get("REPORT_OUT", ROOT / "reports" / "rapport_obesity_levels.pdf"))

ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DV", str(ttf / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", str(ttf / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DV-I", str(ttf / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-B")

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["BodyText"], fontName="DV", fontSize=9.8, leading=14,
                      alignment=TA_JUSTIFY, spaceAfter=5)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="DV-B", fontSize=14, spaceBefore=10, spaceAfter=6,
                    textColor=colors.HexColor("#1f3a5f"))
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="DV-B", fontSize=11, spaceBefore=6, spaceAfter=4,
                    textColor=colors.HexColor("#1f3a5f"))
TITLE = ParagraphStyle("title", parent=ss["Title"], fontName="DV-B", fontSize=20, leading=25, spaceAfter=8)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=8, leading=10, textColor=colors.grey, alignment=0)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8, leading=10, alignment=0, spaceAfter=0)

story: list = []


def p(text: str, style=BODY):
    story.append(Paragraph(text, style))


def fig(name: str, width_cm: float, caption: str):
    from PIL import Image as PILImage

    w, h = PILImage.open(FIG / name).size
    story.append(Image(str(FIG / name), width=width_cm * cm, height=width_cm * cm * h / w))
    p(caption, CAP)


def table(df: pd.DataFrame, widths=None, caption: str | None = None):
    data = [[Paragraph(f"<b>{c}</b>", CELL) for c in df.columns]]
    data += [[Paragraph(str(v), CELL) for v in row] for row in df.itertuples(index=False)]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce6f2")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    block = [t] + ([Paragraph(caption, CAP)] if caption else [])
    story.append(KeepTogether(block))
    story.append(Spacer(1, 4))


def f3(x):
    return f"{x:.3f}".replace(".", ",")


# ------------------------------------------------------------------ page de titre
p("Prédire le niveau d'obésité et le poids à partir des habitudes de vie", TITLE)
p("Classification multiclasse, régression linéaire et MLOps léger", H2)
p("Réalisé par HANABAL Salma · Dépôt : github.com/salmahanabal/obesity-levels", CAP)
story.append(Spacer(1, 6))

p("1. Introduction et problématique", H1)
p("L'obésité est un enjeu de santé publique majeur. Le jeu de données UCI <i>Estimation of Obesity Levels "
  "Based on Eating Habits and Physical Condition</i> (Mexique, Pérou, Colombie) décrit 2 111 personnes par "
  "17 variables : caractéristiques physiques (âge, taille, poids), habitudes alimentaires, activité physique, "
  "transport et antécédents familiaux. La cible est un niveau d'obésité en sept classes ordonnées, du poids "
  "insuffisant à l'obésité de type III.")
p("<b>Problématique.</b> Dans quelle mesure les habitudes alimentaires et le mode de vie permettent-ils de "
  "prédire le niveau d'obésité (classification multiclasse) et le poids (régression linéaire), et ces deux "
  "modélisations sont-elles cohérentes ? Nous suivons le flux complet exploration, préprocessing, "
  "modélisation, évaluation et analyse critique, complété par un suivi d'expériences (MLflow), une API, un "
  "conteneur Docker, des tests et une intégration continue.")
p("<b>Trois pipelines, un seul découpage.</b> Le pipeline A classe les individus sans utiliser le poids ni la "
  "taille. Le pipeline B prédit le poids (la taille est incluse). Le pipeline C calcule l'IMC à partir du poids "
  "prédit puis en déduit la classe, pour la comparer à A. Un run de contraste, volontairement fautif, ajoute "
  "poids et taille au classifieur pour mesurer la fuite de données. Tous les pipelines partagent le même "
  "découpage apprentissage/test.")

# ------------------------------------------------------------------ exploration
p("2. Exploration des données", H1)
p("Le jeu ne contient aucune valeur manquante. Il compte <b>24 doublons</b>, supprimés avant le découpage "
  "(2 111 à 2 087 lignes), afin qu'une même ligne ne soit pas à la fois dans l'apprentissage et le test.")
p("<b>Fuite de données.</b> La classe est dérivée de l'IMC (poids ÷ taille²). Le tableau ci-dessous donne les "
  "bornes d'IMC observées par classe : donner le poids et la taille au classifieur revient à lui donner la "
  "réponse. Les bornes se chevauchent légèrement entre classes voisines (par exemple 22,8 à 28,8 pour "
  "<i>Overweight_Level_I</i>), ce qui s'explique par la nature synthétique d'une partie des lignes.")
b = pd.read_csv(TAB / "bmi_by_class.csv")
b.columns = ["Classe", "IMC min", "IMC max"]
b[["IMC min", "IMC max"]] = b[["IMC min", "IMC max"]].round(1)
table(b, [6 * cm, 3 * cm, 3 * cm], "Tableau 1 — IMC minimal et maximal observés par classe.")
p("<b>Données synthétiques.</b> Environ 77 % des lignes ont été générées par SMOTE. Nous utilisons comme "
  "indicateur (heuristique, non vérifiable) la présence d'au moins une variable ordinale non entière avant "
  "arrondi : 76 % des lignes sont concernées. Ces lignes contiennent des valeurs décimales absurdes pour des "
  "échelles entières (par exemple 2,45 repas).")
p("<b>Biais de genre.</b> Les classes sont très déséquilibrées selon le genre : <i>Obesity_Type_II</i> compte "
  "295 hommes pour 2 femmes, et <i>Obesity_Type_III</i> 323 femmes pour 1 homme. Ce biais provient de la "
  "construction du jeu de données ; il est mesuré et discuté, non corrigé silencieusement.")
fig("eda_genre_x_classe.png", 9.5, "Figure 1 — Répartition du genre selon le niveau d'obésité.")

# ------------------------------------------------------------------ preprocessing
p("3. Préprocessing", H1)
p("Chaque choix est justifié et appris uniquement sur l'apprentissage, pour éviter toute fuite vers le test.")
for t in [
    ("<b>Nettoyage déterministe (avant découpage)</b> : suppression des doublons ; arrondi à l'entier des "
    "échelles ordinales (FCVC, NCP, CH2O, FAF, TUE), dont les décimales sont des artefacts SMOTE ; fusion de "
    "la modalité quasi absente <i>CALC = Always</i> (environ 1 ligne) avec <i>Frequently</i>."),
    ("<b>Découpage unique</b> 70/30 stratifié sur la classe (graine 42) : 1 460 individus d'apprentissage et "
    "627 de test. Les indices sont figés et réutilisés partout ; un test unitaire vérifie l'absence de "
    "recouvrement."),
    ("<b>Encodage</b> : variables binaires en 0/1 ; grignotage et alcool en ordinal avec l'ordre explicite "
    "no &lt; Sometimes &lt; Frequently &lt; Always ; genre et transport en one-hot (première modalité "
    "supprimée pour éviter la colinéarité) ; variables numériques standardisées (indispensable pour le SVM, "
    "le réseau de neurones et la comparaison des coefficients)."),
]:
    p("• " + t)

# ------------------------------------------------------------------ modélisation
p("4. Modélisation et résultats", H1)
p("4.1 Classification multiclasse (pipeline A)", H2)
p("Cinq modèles sont optimisés par GridSearchCV (F1 macro, validation croisée stratifiée à 5 plis) : arbre "
  "de décision, SVM, forêt aléatoire, et deux ajoutés après les premiers résultats, le gradient boosting et "
  "le réseau de neurones MLP. Le champion est choisi sur la validation "
  "croisée de l'apprentissage, jamais sur le test, qui n'est évalué qu'une fois par modèle final.")
c = pd.read_csv(TAB / "classification_results.csv")
main = c[~c["model"].str.endswith("_LEAKAGE")]
ct = pd.DataFrame({
    "Modèle": main["model"], "F1 macro (CV)": main["cv_f1_macro"].map(f3),
    "Accuracy (test)": main["test_accuracy"].map(f3), "F1 macro (test)": main["test_f1_macro"].map(f3),
    "Recall macro (test)": main["test_recall_macro"].map(f3), "Accuracy ±1 classe": main["test_accuracy_pm1"].map(f3),
})
table(ct, None, "Tableau 2 — Comparaison des classifieurs (pipeline A, sans poids ni taille).")
p("<b>Effet des modèles ajoutés.</b> Le gradient boosting et la forêt aléatoire sont à égalité en validation "
  "croisée (0,790 contre 0,789) et sur le test (F1 macro 0,775 contre 0,776) : le gradient boosting est "
  "champion par la règle de la validation croisée, mais l'écart est dans le bruit. Le réseau de neurones "
  "(0,726) ne dépasse pas le SVM. Changer d'algorithme ne lève donc pas le plafond d'environ 0,78, qui "
  "tient aux données et non au modèle.")
sv = pd.read_csv(TAB / "svm_ovr_ovo.csv")
p("<b>SVM : One-vs-Rest ou One-vs-One.</b> En validation croisée, le F1 macro est de "
  f"{f3(sv.iloc[0, 1])} pour le SVC natif (One-vs-One interne), {f3(sv.iloc[1, 1])} pour One-vs-Rest et "
  f"{f3(sv.iloc[2, 1])} pour One-vs-One explicite, avec des écarts-types d'environ 0,03 à 0,04 : les "
  "différences sont inférieures à la variabilité, le choix de la stratégie est sans effet notable.")
p("4.2 Régression linéaire du poids (pipeline B)", H2)
r = pd.read_csv(TAB / "regression_comparison.csv")
rt = pd.DataFrame({"Modèle": r["model"], "MAE (test)": r["test_mae"].map(lambda x: f"{x:.1f}".replace(".", ",")),
                   "RMSE (test)": r["test_rmse"].map(lambda x: f"{x:.1f}".replace(".", ",")),
                   "R² (test)": r["test_r2"].map(f3), "R² (train)": r["train_r2"].map(f3)})
table(rt, None, "Tableau 3 — Régressions Linear, Ridge et Lasso (poids en kg).")
p("Les trois modèles sont quasi identiques : la régularisation n'apporte rien car les données sont peu "
  "colinéaires (VIF maximal de 7,0, pour <i>CALC_Sometimes</i>) et le modèle ne sur-apprend pas. L'OLS "
  "statsmodels (R² d'apprentissage 0,58 avec toutes les indicatrices) fournit coefficients, intervalles de "
  "confiance à 95 % et VIF. Le test de Breusch-Pagan révèle une <b>hétéroscédasticité marquée</b> "
  "(p ≈ 7·10⁻³⁶) ; une cible en log(poids) améliore légèrement le test (RMSE 17,7 kg, R² 0,54).")
fig("model_residuals_ols.png", 15.5, "Figure 2 — Diagnostics de l'OLS : résidus contre prédictions, QQ-plot, "
    "histogramme des résidus.")

# ------------------------------------------------------------------ évaluation
p("5. Évaluation", H1)
p("Le gradient boosting est le champion (F1 macro de 0,775 sur le test, AUC OvR macro de 0,96). Sa matrice de "
  "confusion montre que les erreurs se concentrent entre classes voisines, surtout autour du poids normal "
  "et du surpoids de niveau I (rappels de 0,65 et 0,59), alors que l'obésité de type III est presque "
  "parfaitement reconnue (0,99). Cette "
  "performance doit être relativisée : c'est la classe quasi exclusivement féminine (section 6).")
fig("model_cm_GradientBoosting.png", 16, "Figure 3 — Matrice de confusion du gradient boosting (comptes et "
    "normalisée par ligne).")
p("<b>Cohérence (pipeline C).</b> Prédire le poids puis en déduire la classe donne une accuracy de 0,228 "
  "(±1 classe : 0,707 ; F1 macro 0,193), contre 0,781 (±1 classe : 0,884) pour la classification directe. "
  "Une erreur moyenne de 18 kg sur le poids déplace l'IMC de plusieurs points, souvent d'une classe à "
  "l'autre. Les deux démarches ne sont donc pas cohérentes à l'échelle individuelle : les habitudes de vie "
  "renseignent mieux sur la <i>classe</i> que sur le <i>poids exact</i>.")

# ------------------------------------------------------------------ analyse critique
p("6. Analyse critique", H1)
p("<b>Surapprentissage.</b> L'arbre et la forêt atteignent un F1 macro de 0,998 sur l'apprentissage "
  "(0,987 pour le gradient boosting, 0,96 pour le MLP) contre 0,71 à 0,78 sur le test. La courbe d'apprentissage du classifieur montre une validation encore "
  "croissante avec la taille des données, sans rattraper l'apprentissage : le modèle bénéficierait de plus "
  "de données. À l'inverse, pour la régression les courbes se rejoignent à environ −17,5 kg : il n'y a pas "
  "de surapprentissage mais un biais élevé (le modèle linéaire sous-apprend).")
fig("analysis_learning_curves.png", 15.5, "Figure 4 — Courbes d'apprentissage du classifieur champion et de "
    "la régression champion.")
p("<b>Données synthétiques.</b> Sur le test, le champion obtient une accuracy de 0,86 (F1 macro 0,73) sur les "
  "lignes que l'heuristique juge synthétiques, contre 0,56 (F1 macro 0,40) sur les lignes jugées collectées. "
  "Les scores globaux sont donc optimistes : SMOTE crée des interpolations proches les unes des autres, "
  "retrouvées à la fois dans l'apprentissage et le test. Ce constat vaut aussi pour la validation croisée.")
p("<b>Biais de genre.</b> Le F1 macro est de 0,77 pour les femmes et de 0,62 pour les hommes. Le genre est la "
  "deuxième variable par permutation sur le test (0,19 de perte de F1, après l'âge à 0,34) : le modèle l'utilise comme "
  "raccourci pour les classes quasi exclusivement féminines ou masculines. Son équité n'est pas garantie.")
imp = pd.read_csv(TAB / "feature_importances.csv").head(6)
it = pd.DataFrame({
    "Variable": imp["variable"],
    "Permutation (test)": imp["perm_importance_test"].map(f3),
    "Rang régression (|coef.|)": imp["rang_abs_coef_regression"],
})
p("<b>Facteurs clés.</b> Les classements diffèrent selon la méthode : l'âge puis le genre dominent par "
  "permutation, tandis que la régression retient d'abord "
  "l'antécédent familial de surpoids (environ 20 kg d'écart), puis le transport et le grignotage. Les "
  "classements sont à interpréter avec prudence : variables corrélées et données synthétiques.")
table(it, None, "Tableau 4 — Six variables les plus importantes (permutation sur le test) et leurs rangs.")
l = pd.read_csv(TAB / "leakage_comparison.csv")
p("<b>Fuite de données.</b> Avec le poids et la taille, la forêt atteint une accuracy de "
  f"{f3(l.iloc[1]['test_accuracy'])} et une accuracy à ±1 classe de {f3(l.iloc[1]['test_accuracy_pm1'])}, "
  f"contre {f3(l.iloc[0]['test_accuracy'])} et {f3(l.iloc[0]['test_accuracy_pm1'])} sans eux. L'écart mesure "
  "ce que l'on obtiendrait en trichant ; ce run est tagué <i>leakage_reference</i> dans MLflow et n'est "
  "jamais retenu comme modèle final.")

# ------------------------------------------------------------------ MLOps
p("7. Couche MLOps", H1)
for t in [
    ("<b>MLflow</b> : deux expériences (classification, régression), un run par modèle (paramètres, "
    "métriques d'apprentissage, de validation croisée et de test, matrices de confusion, rapport par classe, "
    "modèle signé avec exemple d'entrée), tags <i>git_commit</i>, <i>dataset_hash</i> et <i>split_seed</i> ; "
    "les deux champions sont enregistrés dans le registre avec l'alias <i>champion</i>."),
    ("<b>API FastAPI</b> : <i>/health</i>, <i>/predict/class</i> (habitudes de vie, sans taille ni poids) et "
    "<i>/predict/weight</i> (avec la taille, renvoie poids, IMC et classe). Schémas Pydantic stricts "
    "(bornes et modalités autorisées), erreur 422 explicite en cas de valeur invalide."),
    ("<b>Qualité</b> : tests pytest (découpage, absence de fuite, seuils IMC, pipelines, API, interface), "
    "ruff, intégration continue GitHub Actions ; un second workflow rejoue sur machine propre "
    "l'ensemble (<i>make all</i>, notebooks, construction de l'image Docker et appel de l'API dans le "
    "conteneur)."),
    ("<b>Interface Streamlit</b> : téléchargement des données nettoyées et pré-traitées, test d'un fichier "
    "CSV avec résultats commentés, saisie manuelle avec guide des variables."),
]:
    p("• " + t)

# ------------------------------------------------------------------ conclusion
p("8. Conclusion, limites et recommandations", H1)
p("Les habitudes de vie permettent de prédire le niveau d'obésité de façon modérée (F1 macro d'environ 0,78 "
  "pour le gradient boosting comme pour la forêt aléatoire, sans gain avec des modèles plus puissants) mais "
  "pas le poids avec précision (erreur d'environ 18 kg), et les deux "
  "modélisations ne sont pas cohérentes individuellement (accuracy de 0,23 pour la classe dérivée du poids).")
p("<b>Limites.</b> (i) Les trois quarts des données sont synthétiques : les scores sont optimistes et un test "
  "composé uniquement de données collectées serait plus honnête (accuracy de 0,56 mesurée sur ce sous-groupe). "
  "(ii) Le biais de genre sur les classes extrêmes rend les résultats non transférables. (iii) L'indicateur "
  "de lignes synthétiques est une heuristique. (iv) Le modèle de régression est linéaire et hétéroscédastique.")
p("<b>Recommandations.</b> Collecter davantage de données réelles et évaluer sur des données non synthétiques ; "
  "traiter l'IMC comme cible continue ou ordinale plutôt que sept classes indépendantes ; tester des approches "
  "ordinales et des transformations de la cible (log) ; auditer l'équité du modèle par genre avant tout usage.")
p("<b>Reproductibilité.</b> <i>make setup</i>, <i>make all</i> (données, entraînements, évaluation, export), "
  "<i>make test</i>, <i>make serve</i>, <i>make ui</i> ; graine 42 lue depuis <i>config.yaml</i>. Outil "
  "pédagogique : ce travail ne constitue pas un avis médical.")


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DV", 7.5)
    canvas.setFillColor(colors.grey)
    canvas.drawCentredString(A4[0] / 2, 1.1 * cm, f"obesity-levels — HANABAL Salma — page {doc.page}")
    canvas.restoreState()


SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                  bottomMargin=1.8 * cm, title="obesity-levels — rapport", author="HANABAL Salma"
                  ).build(story, onFirstPage=footer, onLaterPages=footer)
print("écrit :", OUT)
