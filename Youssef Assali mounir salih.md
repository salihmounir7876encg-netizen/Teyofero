<img src="https://image2url.com/images/1765362786985-df3bb0b1-e113-40f7-a0cc-80d894c711cb.jpg"
     alt="Logo marouane izlitne"
     style="height:300px; margin-right:300px; float:left; border-radius:10px;">

<br><br clear="left"/>
# IZLITNE_Marouane 22006529 CAC2

<img src="ziadnoubair.png" style="height:464px;margin-right:432px"/>

# NOUBAIR_Ziad 22007685 CAC2

# École Nationale de Commerce et de Gestion (ENCG) - 4ème Année
``` python
from google.colab import ai
response = ai.generate_text("What is the capital of France?")
```
``` python
import numpy as np
import IPython.display as display
from matplotlib import pyplot as plt
import io
import base64

ys = 200 + np.random.randn(100)
x = [x for x in range(len(ys))]

fig = plt.figure(figsize=(4, 3), facecolor='w')
plt.plot(x, ys, '-')
plt.fill_between(x, ys, 195, where=(ys > 195), facecolor='g', alpha=0.6)
plt.title("Sample Visualization", fontsize=10)

data = io.BytesIO()
plt.savefig(data)
image = F"data:image/png;base64,{base64.b64encode(data.getvalue()).decode()}"
alt = "Sample Visualization"
display.display(display.Markdown(F"""![{alt}]({image})"""))
plt.close(fig)
```

Les notebooks Colab exécutent ce code sur les serveurs cloud de Google.
Vous pouvez donc bénéficier de toute la puissance du matériel Google, y
compris les `<a href="#using-accelerated-hardware">`{=html}GPU et
TPU`</a>`{=html}, quelle que soit la puissance de votre ordinateur. Vous
n'avez besoin que d'un navigateur.

Par exemple, si vous attendez que le code
`<strong>`{=html}pandas`</strong>`{=html} termine de s'exécuter et que
vous souhaitez accélérer le processus, vous pouvez passer à un
environnement d'exécution GPU et utiliser des bibliothèques telles que
`<a href="https://rapids.ai/cudf-pandas">`{=html}RAPIDS
cuDF`</a>`{=html}, qui fournissent une accélération sans modification de
code.

Pour en savoir plus sur l'accélération de pandas dans Colab, consultez
le
`<a href="https://colab.research.google.com/github/rapidsai-community/showcase/blob/main/getting_started_tutorials/cudf_pandas_colab_demo.ipynb">`{=html}guide
de 10 minutes`</a>`{=html} ou la
`<a href="https://colab.research.google.com/github/rapidsai-community/showcase/blob/main/getting_started_tutorials/cudf_pandas_stocks_demo.ipynb">`{=html}démo
d'analyse des données boursières aux États-Unis`</a>`{=html}.

::: markdown-google-sans
## Machine learning
:::

Colab vous permet d'importer un ensemble de données d'images,
d'entraîner un classificateur d'images sur cet ensemble et d'évaluer le
modèle, tout cela avec
`<a href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/quickstart/beginner.ipynb">`{=html}quelques
lignes de code`</a>`{=html}.

Colab est très largement utilisé par la communauté du machine learning,
par exemple dans les applications suivantes : - Premiers pas avec
TensorFlow - Développement et entraînement de réseaux de neurones -
Expérimentation avec les TPU - Dissémination de la recherche en IA -
Création de tutoriels

Pour voir comment les notebooks Colab sont utilisés dans des
applications de machine learning, reportez-vous aux
`<a href="#machine-learning-examples">`{=html}exemples de machine
learning`</a>`{=html} ci-dessous.

``` python
# =========================
# TP PARTIE 1 — STATISTIQUES & LOI NORMALE EN FINANCE
# =========================

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# -------------------------
# DONNÉES
# -------------------------
rendements_A = np.array([
    1.2, 0.8, -0.5, 1.5, 0.9, 1.1, 0.7, 1.3, 1.0, 0.6, 1.4, 0.8,
    1.1, 0.9, -0.3, 1.2, 1.0, 1.5, 0.8, 1.3, 0.9, 1.1, 1.2, 1.0
])

rendements_B = np.array([
    4.5, -2.1, 6.2, -3.5, 5.8, 7.1, -1.8, 4.9, 3.2, -4.2, 8.5, -2.7,
    5.1, 6.8, -3.1, 7.3, 4.5, -2.9, 6.7, 5.3, -3.8, 7.9, 4.2, 5.5
])

capital = 500_000
perte_max_toleree = 50_000
taux_sans_risque = 3.0  # % annuel


print("=" * 80)
print("TP PARTIE 1 — STATISTIQUES ET LOI NORMALE EN FINANCE")
print("=" * 80)


# -------------------------
# QUESTION 1.1 — STATS
# -------------------------
def calculer_stats_portefeuille(rendements: np.ndarray, nom: str) -> dict:
    """Calcule statistiques descriptives pour un portefeuille (rendements en % mensuels)."""
    moyenne_mensuelle = float(np.mean(rendements))
    ecart_type_mensuel = float(np.std(rendements, ddof=1))  # ddof=1 pour échantillon
    mediane = float(np.median(rendements))

    # Annualisation
    rendement_annuel = ((1 + moyenne_mensuelle / 100) ** 12 - 1) * 100
    volatilite_annuelle = ecart_type_mensuel * np.sqrt(12)

    return {
        "nom": nom,
        "moyenne_mensuelle": moyenne_mensuelle,
        "ecart_type_mensuel": ecart_type_mensuel,
        "mediane": mediane,
        "rendement_annuel": float(rendement_annuel),
        "volatilite_annuelle": float(volatilite_annuelle),
    }


stats_A = calculer_stats_portefeuille(rendements_A, "A (Conservative)")
stats_B = calculer_stats_portefeuille(rendements_B, "B (Agressif)")

print("\n" + "=" * 80)
print("QUESTION 1.1 — STATISTIQUES DESCRIPTIVES")
print("=" * 80)

for s in [stats_A, stats_B]:
    print(f"\n📊 PORTEFEUILLE {s['nom']}")
    print(f"  • Rendement mensuel moyen : {s['moyenne_mensuelle']:.2f}%")
    print(f"  • Écart-type mensuel      : {s['ecart_type_mensuel']:.2f}%")
    print(f"  • Médiane                 : {s['mediane']:.2f}%")
    print(f"  • Rendement annualisé     : {s['rendement_annuel']:.2f}%")
    print(f"  • Volatilité annualisée   : {s['volatilite_annuelle']:.2f}%")


# -------------------------
# QUESTION 1.2 — VISU
# -------------------------
print("\n" + "=" * 80)
print("QUESTION 1.2 — VISUALISATION DISTRIBUTIONS")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogrammes superposés
ax1 = axes[0]
ax1.hist(rendements_A, bins=10, alpha=0.6, edgecolor="black", label="A (Conservative)", density=True)
ax1.hist(rendements_B, bins=10, alpha=0.6, edgecolor="black", label="B (Agressif)", density=True)

ax1.axvline(stats_A["moyenne_mensuelle"], linestyle="--", linewidth=2, label=f"Moyenne A = {stats_A['moyenne_mensuelle']:.2f}%")
ax1.axvline(stats_B["moyenne_mensuelle"], linestyle="--", linewidth=2, label=f"Moyenne B = {stats_B['moyenne_mensuelle']:.2f}%")

ax1.set_title("Distributions rendements mensuels")
ax1.set_xlabel("Rendement mensuel (%)")
ax1.set_ylabel("Densité")
ax1.grid(True, alpha=0.3)
ax1.legend()

# Boxplots
ax2 = axes[1]
bp = ax2.boxplot([rendements_A, rendements_B], labels=["A", "B"], patch_artist=True, widths=0.6)
ax2.axhline(0, linestyle=":", linewidth=1)
ax2.set_title("Boxplots comparatifs (outliers)")
ax2.set_ylabel("Rendement mensuel (%)")
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.show()

print("✓ Graphiques générés")


# -------------------------
# QUESTION 1.3 — VaR 95%
# -------------------------
print("\n" + "=" * 80)
print("QUESTION 1.3 — VALUE AT RISK (VaR 95%)")
print("=" * 80)

def calculer_var_portefeuille(stats_dict: dict, capital: float, alpha: float = 0.05) -> dict:
    """Calcule VaR paramétrique mensuelle et annuelle (en % et en €)."""
    z_alpha = stats.norm.ppf(alpha)  # ~ -1.645

    var_mensuelle_pct = stats_dict["moyenne_mensuelle"] + z_alpha * stats_dict["ecart_type_mensuel"]
    var_annuelle_pct = stats_dict["rendement_annuel"] + z_alpha * stats_dict["volatilite_annuelle"]

    var_mensuelle_euros = capital * (var_mensuelle_pct / 100)
    var_annuelle_euros = capital * (var_annuelle_pct / 100)

    return {
        "var_mensuelle_pct": float(var_mensuelle_pct),
        "var_annuelle_pct": float(var_annuelle_pct),
        "var_mensuelle_euros": float(var_mensuelle_euros),
        "var_annuelle_euros": float(var_annuelle_euros),
    }


var_A = calculer_var_portefeuille(stats_A, capital)
var_B = calculer_var_portefeuille(stats_B, capital)

print(f"\n💰 CAPITAL INVESTI : €{capital:,.0f}")
print(f"🚨 PERTE MAX TOLÉRÉE : €{perte_max_toleree:,.0f} (-{perte_max_toleree/capital*100:.0f}%)")

print("\n📉 PORTEFEUILLE A")
print(f"  • VaR 95% mensuelle : {var_A['var_mensuelle_pct']:.2f}% → €{var_A['var_mensuelle_euros']:,.0f}")
print(f"  • VaR 95% annuelle  : {var_A['var_annuelle_pct']:.2f}% → €{var_A['var_annuelle_euros']:,.0f}")

print("\n📉 PORTEFEUILLE B")
print(f"  • VaR 95% mensuelle : {var_B['var_mensuelle_pct']:.2f}% → €{var_B['var_mensuelle_euros']:,.0f}")
print(f"  • VaR 95% annuelle  : {var_B['var_annuelle_pct']:.2f}% → €{var_B['var_annuelle_euros']:,.0f}")

contrainte_A = abs(var_A["var_annuelle_euros"]) <= perte_max_toleree
contrainte_B = abs(var_B["var_annuelle_euros"]) <= perte_max_toleree

print("\n✅ CONTRAINTE CLIENT (|VaR annuelle| ≤ €50,000) :")
print(f"  • A : {'✓ OK' if contrainte_A else '✗ NON'} (VaR = €{var_A['var_annuelle_euros']:,.0f})")
print(f"  • B : {'✓ OK' if contrainte_B else '✗ NON'} (VaR = €{var_B['var_annuelle_euros']:,.0f})")

# Test normalité Shapiro-Wilk
stat_A, p_A = stats.shapiro(rendements_A)
stat_B, p_B = stats.shapiro(rendements_B)

print("\n🔬 TEST NORMALITÉ (Shapiro-Wilk)")
print(f"  • A : stat={stat_A:.4f}, p-value={p_A:.4f} → {'Normal (p>0.05)' if p_A>0.05 else 'Non-normal (p≤0.05)'}")
print(f"  • B : stat={stat_B:.4f}, p-value={p_B:.4f} → {'Normal (p>0.05)' if p_B>0.05 else 'Non-normal (p≤0.05)'}")


# -------------------------
# QUESTION 1.4 — Sharpe
# -------------------------
print("\n" + "=" * 80)
print("QUESTION 1.4 — RATIO SHARPE")
print("=" * 80)

sharpe_A = (stats_A["rendement_annuel"] - taux_sans_risque) / stats_A["volatilite_annuelle"]
sharpe_B = (stats_B["rendement_annuel"] - taux_sans_risque) / stats_B["volatilite_annuelle"]

print(f"\n📊 Sharpe A : {sharpe_A:.3f}")
print(f"📊 Sharpe B : {sharpe_B:.3f}")

print("\n🎯 Recommandation (à rédiger en 3–5 phrases dans le rapport) :")
print("  - Compare VaR vs contrainte, Sharpe, et la normalité (Shapiro).")
```
