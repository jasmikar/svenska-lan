# Kreditriskmodell — Svenska Bolån

Ett maskininlärningsprojekt för att prediktera default(betalnings)-risk på bolån, byggt som ett portföljprojekt för att demonstrera kunskaper inom dataanalys, kreditrisk och Python.

## Innehåll

| Fil | Beskrivning |
|---|---|
| `bolan_analys.ipynb` | Fullständig analys — EDA, feature engineering, modellträning och validering |
| `app.ipynb` | Interaktiv kalkylator för riskbedömning i realtid |
| `bolan_data.csv` | Syntetisk bolånsdata, 10 000 observationer |

## Arbetsflöde

1. **Datainläsning** — 10 000 syntetiska bolån med realistiska korrelationer
2. **Explorativ analys** — Korrelationsmatris och default-rate per bostadsort
3. **Feature engineering** — LTI-kvot (lån/inkomst) och månadskostnad
4. **Modellträning** — Logistisk regression med StandardScaler
5. **Validering** — Gini-koefficient, ROC-kurva, konfusionsmatris
6. **Interaktivt verktyg** — Kalkylator för att simulera låntagarprofiler

## Resultat

- **Gini-koefficient: 0.43** — acceptabel diskrimineringsförmåga för kreditrisk
- Viktigaste riskdrivare: skuldkvot och lån-till-inkomst-kvot
- Hög kontantinsats minskar default-risken

## Tekniker

- Python — pandas, numpy, scikit-learn, matplotlib, seaborn
- Logistisk regression med standardisering
- Utvärdering enligt branschstandard (Gini, AUC, IFRS 9-perspektiv)
- Interaktivt gränssnitt med ipywidgets

## Kom igång

```bash
git clone https://github.com/jasmikar/svenska-bolan
cd svenska-bolan
pip install -r requirements.txt
jupyter notebook bolan_analys.ipynb
```

## Notering

Datan är syntetisk och genererad för demonstrationssyfte. I produktion ersätts CSV-inläsningen med en BigQuery-koppling mot historisk lånedata.
