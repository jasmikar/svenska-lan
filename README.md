# Kreditriskmodell — Svenska Lån

Ett maskininlärningsprojekt för att förutsäga risken att en låntagare slutar betala, byggt som ett portföljprojekt för att demonstrera kunskaper inom dataanalys, kreditrisk och Python.

Projektet innehåller två modeller — en för bolån och en för privatlån — för att visa hur kreditriskbedömning skiljer sig beroende på lånetyp.

---

## Projekt 1 — Bolån

| Fil | Beskrivning |
|---|---|
| `bolan_analys.ipynb` | Analys — EDA, feature engineering, modellträning och validering |
| `app.ipynb` | Interaktiv kalkylator för riskbedömning i realtid |
| `bolan_data.csv` | Syntetisk bolånsdata, 10 000 observationer |

**Variabler:** lånebelopp, inkomst, skuldkvot, kontantinsats, bostadsort, ränta  
**Gini-koefficient:** 0.43  
**Notering:** Bolån har säkerhet (bostaden) vilket ger lägre ränta (2–5%) och annorlunda riskprofil än privatlån

---

## Projekt 2 — Privatlån

| Fil | Beskrivning |
|---|---|
| `privatlan_analys.ipynb` | Analys — EDA, feature engineering, modellträning och validering |
| `privatlan_data.csv` | Syntetisk privatlånsdata, 10 000 observationer |

**Variabler:** lånebelopp, inkomst, ålder, anställningsform, syfte, ränta, amorteringstid, betalningsbörda, skuldsättningsgrad, betalningsanmärkning  
**Gini-koefficient:** ~0.55  
**Notering:** Privatlån saknar säkerhet vilket ger högre ränta (4–25%) och gör anställningsform och betalningshistorik till viktigare riskdrivare

---

## Arbetsflöde (båda projekten)

1. **Datainläsning** — syntetisk data, i produktion via BigQuery
2. **Explorativ analys** — korrelationsmatris och betalningsproblem per grupp
3. **Feature engineering** — LTI-kvot, månadskostnad, betalningsbörda
4. **Modellträning** — logistisk regression med `class_weight='balanced'`
5. **Validering** — Gini-koefficient, ROC-kurva, konfusionsmatris
6. **Interaktivt verktyg** — kalkylator för att simulera låntagarprofiler (bolån)

---

## Tekniker

- Python — pandas, numpy, scikit-learn, matplotlib, seaborn, ipywidgets
- Logistisk regression med standardisering och balanserad klassvikt
- One-hot encoding av kategoriska variabler
- Utvärdering enligt branschstandard (Gini, AUC, IFRS 9-perspektiv)

---

## Kom igång

```bash
git clone https://github.com/jasmikar/svenska-bolan
cd svenska-bolan
pip install -r requirements.txt
jupyter notebook privatlan_analys.ipynb
```

---

## Notering

All data är syntetisk och genererad för demonstrationssyfte. I produktion ersätts CSV-inläsningen med en BigQuery-koppling och modellerna tränas och driftsätts i Vertex AI på Google Cloud Platform.
