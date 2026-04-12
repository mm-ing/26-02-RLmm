# RL Use Cases

## Inhaltsübersicht

| UE  | Use Case                                      |
|-----|-----------------------------------------------|
| 1   | Lieferkettenoptimierung                       |
| 2   | Dynamische Preisgestaltung in Echtzeit        |
| 3   | Robotik und autonome Systeme                  |
| 4   | Energieoptimierung in Gebäuden und Fabriken   |
| 5   | Finanzportfolio-Management                    |
| 6   | Weitere Use Cases + Ausblick                  |

---

## UE 1: Lieferkettenoptimierung

### Überblick

Lieferketten sind hochkomplexe Systeme mit tausenden Entscheidungsvariablen:
Lagerbestand, Transportrouten, Lieferantenauswahl, Produktionsplanung — alles beeinflusst sich gegenseitig und verändert sich dynamisch.

Klassische Ansätze (lineare Programmierung, Heuristiken) kommen bei dynamischen Störungen (Lieferausfälle, Nachfragespitzen, Pandemien) schnell an Grenzen.

**RL-Formulierung:**

```

Zustand s:

- Aktueller Lagerbestand je Standort
- Offene Bestellungen
- Lieferzeiten
- Aktuelle Nachfrageprognose
- Transportkapazitäten

Aktion a:

- Bestellmenge je Lieferant
- Transportroute wählen
- Lagerallokation anpassen

Reward r:

- Lieferung pünktlich erfüllt

- Lagerkosten (Überbestand)
- Fehllieferung / Stockout-Kosten
- Transportkosten

```

### Reales Beispiel: Google Supply Chain

Google DeepMind optimierte intern die Kühlsysteme von Rechenzentren (Vorstufe zur Supply Chain Optimierung) — Ergebnis: **40 % Reduktion der Kühlenergie**.

Amazon nutzt RL-basierte Systeme für die Lagerroboter-Koordination in Fulfillment-Centern: Tausende Kiva-Roboter werden gleichzeitig gesteuert.

### Concretes Anwendungsbeispiel: Lagerbestand

```python
# Vereinfachtes Inventory-RL-Problem:

Zustand:   [bestand=50, nachfrage_prognose=70, lieferzeit=3 Tage]
Aktion:    bestelle 30 Einheiten
Reward:    -3 * lagerkosten + -5 * stockout_units + -2 * transportkosten

Ohne RL (klassisch):    Fixe Bestellregel: "Bestellpunkt = 40 Einheiten"
Mit RL:                 Dynamisch: berücksichtigt Prognose, Saison, Lieferanten-Delays
→ Typische Verbesserung: 15–30 % Kostensenkung in Simulationen
```

### Herausforderungen

| Herausforderung | Beschreibung |
|----------------|-------------|
| Große Zustandsräume | Tausende Standorte × Produktkategorien |
| Nicht-stationäre Nachfrage | Saison, Trends, Krisen |
| Multi-Agent | Mehrere Lieferanten und Lager interagieren |
| Sparse Rewards | Stockout-Kosten treten selten, aber teuer auf |
| Simulationsqualität | Schlechtes Simulations-Modell → schlechte Policy |

### Empfohlene Videos & Ressourcen

- **Google DeepMind: Cooling Data Centers with AI**  
  YouTube-Suche: `"DeepMind AI reduces Google data centre cooling bill"`  
  *(Originalvideo auf dem DeepMind YouTube-Kanal)*

- **Amazon Robotics — Fulfillment Center Tour**  
  YouTube-Suche: `"Amazon Robotics fulfillment center RL"`

- **OR & RL for Supply Chain (MIT OpenCourseWare)**  
  YouTube-Suche: `"MIT reinforcement learning supply chain inventory"`

- **Paper:** Oroojlooyjadid et al. (2021) — *A Review of Deep Reinforcement Learning for Inventory Optimization*  
  *(Google Scholar: "deep reinforcement learning inventory optimization review")*

---

## UE 2: Dynamische Preisgestaltung in Echtzeit

### Überblick

**Dynamische Preisgestaltung** (Dynamic Pricing) bedeutet: Preise in Echtzeit an Angebot, Nachfrage, Wettbewerb und Kundenverhalten anpassen.

Früher: manuelle Preisstrategen und statische Regeln.  
Heute: RL-Agenten reagieren in Millisekunden auf Marktveränderungen.

**Bekannte Anwender:** Amazon (Preisänderungen alle 10 Minuten), Uber (Surge Pricing), Airline-Ticketing, Hotelportale.

### RL-Formulierung

```
Zustand s:
  - Aktueller Preis des eigenen Produkts
  - Wettbewerberpreise (Web-Scraping / API)
  - Tageszeit, Wochentag, Saison
  - Lagerbestand
  - Klickrate / Conversion-Rate der letzten Stunde
  - Kundensegment (falls bekannt)

Aktion a:
  - Preis erhöhen (+5 %, +10 %, ...)
  - Preis senken (-5 %, -10 %, ...)
  - Preis beibehalten

Reward r:
  Umsatz - Kosten - Strafterm für zu starke Preisschwankungen
```

### Konkretes Beispiel: E-Commerce Pricing

```
Ausgangssituation:
  Produkt: USB-Kabel
  Basispreis: 12,99 €
  Wettbewerber: 11,50 € (Amazon) | 13,20 € (MediaMarkt)
  Lagerbestand: 800 Stück
  Uhrzeit: Montag 14:00 Uhr

RL-Agent entscheidet:
  → 12,49 € (leicht unter Wettbewerb, hoher Bestand)
  → Konversionsrate steigt um 8 %
  → Reward: positiv

Nächste Situation (Freitag 18:00):
  Lagerbestand: 50 Stück, Wettbewerber ausverkauft
  RL-Agent entscheidet:
  → 15,99 € (Knappheit, keine Konkurrenz)
  → Reward: hoch trotz weniger Verkäufen (Marge!)
```

### Uber Surge Pricing als RL-analoges System

```
Zustand:    Fahrerangebot vs. Nachfrage in Region X
Aktion:     Surge-Multiplikator 1.0x → 4.5x
Reward:     Fahrer nehmen Auftrag an + Passagier find Fahrt
Constraint: Zu hoher Surge → Passagiere wechseln zu Taxi

→ RL-ähnliche adaptive Anpassung in Echtzeit
```

### Ethische Überlegungen (Diskussionspunkt)

```
Diskussion im Kurs:
  - Ist Surge Pricing während einer Naturkatastrophe fair?
  - Dürfen Preise nach Kundenprofil differenziert werden?
  - EU-Regulierung (Digital Markets Act) vs. algorithmisches Pricing
  - Transparenzpflicht für automatisierte Preissysteme?
```

### Empfohlene Videos & Ressourcen

- **Uber Surge Pricing erklärt**  
  YouTube-Suche: `"Uber surge pricing algorithm explained"`

- **AWS Reinforcement Learning for Pricing (re:Invent)**  
  YouTube-Suche: `"AWS re:Invent reinforcement learning pricing optimization"`

- **Dynamic Pricing mit RL — Praxisbeispiel**  
  YouTube-Suche: `"reinforcement learning dynamic pricing e-commerce tutorial"`

- **Paper:** Kastius & Schlosser (2022) — *Dynamic Pricing using Reinforcement Learning*  
  *(Google Scholar: "dynamic pricing reinforcement learning Kastius 2022")*

---

## UE 3: Robotik und Autonome Systeme

### Überblick

Robotik ist das Paradebeispiel für RL in der physischen Welt: Ein Agent (Roboter) muss durch Interaktion mit einer realen, unsicheren Umgebung lernen — Bewegung, Greifaufgaben, Navigation, menschliche Kollaboration.

**Warum RL statt klassischer Robotik?**

```
Klassische Robotik:      Handcodierte Bewegungsplanung
                         → funktioniert für bekannte, feste Szenarien
                         → scheitert bei Störungen, unbekannten Objekten

RL-Robotik:              Lernt durch Erfahrung
                         → generalisiert auf neue Situationen
                         → adaptiert bei veränderten Bedingungen
```

### Anwendungsfelder

| Anwendung | Methode | Ergebnis |
|-----------|---------|---------|
| Greifaufgaben (Grasping) | SAC + HER | Google Roboter lernt in 4 Stunden autonomes Greifen |
| Laufroboter | PPO, SAC | ANYmal läuft über unbekanntes Terrain |
| Drohnenfliegen | Model-Based RL | Autonomes Fliegen durch Windturbulenzen |
| Chirurgische Roboter | Imitation + RL | Davinchi-Roboter lernt Naht-Technik |
| Autonomes Fahren | Multi-Agent RL | Waymo nutzt RL für Spurwechsel-Entscheidungen |

### Konkretes Beispiel: Roboterarme greifen lernen

```
Setup:
  Roboterarm mit 6 Gelenken
  Kamera (RGB + Tiefe)
  Aufgabe: Greife beliebige Objekte aus einem Behälter

Zustand:   Kamerabild + Gelenkwinkel + Greifer-Zustand
Aktion:    6 Gelenkwinkelgeschwindigkeiten + Greifer öffnen/schließen
Reward:    +10 Objekt erfolgreich gegriffen und angehoben
           -1 pro Zeitschritt (Effizienz)
           -5 Kollision mit anderen Objekten

Herausforderung (Sim-to-Real):
  Training in Simulation (MuJoCo/Isaac Gym): 10 Millionen Schritte
  Übertragung auf echten Roboter:
    → Domain Randomization: variierende Reibung, Lichtbedingungen, Objektgewichte
    → Real-World Fine-Tuning: einige hundert echte Greifversuche
```

### OpenAI Dactyl: Rubik's Cube lösen mit einer Hand

```
2019: OpenAI trainierte einen Roboterhand-Controller in Simulation
      → überträgt auf echten Roboter (Shadow Hand)
      → löst Rubik's Cube einhändig

Schlüssel:
  1. Domain Randomization: 1000+ variierende Simulationsparameter
  2. Automatic Domain Randomization (ADR): Schwierigkeit automatisch steigern
  3. Transfer: Zero-Shot auf reale Hand (keine Echtzeit-Feinabstimmung)
```

### Boston Dynamics + RL

```
Spot (Laufroboter):
  Klassischer Controller: handcodiert für bekannten Untergrund
  RL-Erweiterung: lernt aus Simulation auf unebenem Gelände zu laufen
  → Treppe, Schnee, Baustellen — ohne manuelle Anpassung

Atlas (humanoider Roboter):
  Parkour-Bewegungen durch RL in Simulation
  → Transfer auf reale Hardware
```

### Empfohlene Videos & Ressourcen

- **OpenAI Solving Rubik's Cube with a Robot Hand**  
  YouTube: `https://www.youtube.com/watch?v=x4O8pojMF0w`  
  *(OpenAI Official Channel — verifiziert)*

- **DeepMind — Locomotion with RL**  
  YouTube-Suche: `"DeepMind locomotion reinforcement learning emergence"`

- **Boston Dynamics Spot & Atlas**  
  YouTube-Suche: `"Boston Dynamics Atlas parkour reinforcement learning"`

- **Google Robotics — RT-2**  
  YouTube-Suche: `"Google RT-2 robot vision language action"`

- **Yannic Kilcher — Robotics RL Paper Reviews**  
  YouTube-Suche: `"Yannic Kilcher reinforcement learning robotics"`

- **Paper:** Andrychowicz et al. (2020) — *Learning Dexterous In-Hand Manipulation*  
  *(Google Scholar: "OpenAI Dactyl learning dexterous manipulation")*

---

## UE 4: Energieoptimierung in Gebäuden und Fabriken

### Überblick

Gebäude verursachen ca. **40 % des globalen Energieverbrauchs** — Heizung, Kühlung, Beleuchtung, Aufzüge. Industrielle Produktionsanlagen verbrauchen zusätzlich enorme Mengen.

Klassische Steuerung: regelbasiert, statisch (z. B. Heizung an wenn T < 20°C).  
RL: lernt, **wann** und **wie viel** Energie optimal eingesetzt wird — vorausschauend, kontextbewusst.

### RL-Formulierung: HVAC-Steuerung

HVAC = Heating, Ventilation, Air Conditioning

```
Zustand s:
  - Innentemperatur (je Raum / Zone)
  - Außentemperatur + Wettervorhersage
  - Belegung (Personen im Gebäude)
  - Aktuelle Strompreise (Spotmarkt)
  - Tageszeit, Wochentag
  - CO₂-Konzentration

Aktion a:
  - Heizleistung je Zone (0–100 %)
  - Lüftungsrate
  - Kühlaggregat ein/aus
  - Zeitpunkt für Vorheizung/Vorkühlung

Reward r:
  - Komfort: Temperatur im Zielbereich (20–22°C)
  - Effizienz: minimaler Energieverbrauch
  - Kosten: Energie zu günstigen Zeiten beziehen
  - Strafe: Zonen zu heiß/kalt
```

### Google DeepMind: Rechenzentrum-Kühlung

```
Ausgangslage: Google Rechenzentrum, riesiger Kühlbedarf
Problem:      Komplexes System mit tausenden Sensoren
              Nicht-lineare Wechselwirkungen

RL-Lösung (2016):
  Training: historische Sensordaten (Supervised Pretraining)
  Fine-Tuning: Online RL im realen Betrieb
  
Ergebnis:
  - 40 % Reduktion des PUE (Power Usage Effectiveness) für Kühlung
  - 15 % Gesamtenergieeinsparung des Rechenzentrums
  - Seit 2018: vollständig autonom in Betrieb
```

### Industrielle Anwendung: Fabrikhalle

```
Produktionshalle mit:
  - 20 CNC-Maschinen (variabler Energiebedarf)
  - Klimaanlage
  - Druckluftsystem
  - Lastspitzen vermeiden (teures Peak-Pricing)

RL-Agent:
  Zustand:  Produktionsplan + aktuelle Last + Strompreise
  Aktion:   Startzeiten der Maschinen verschieben (innerhalb Toleranz)
  Reward:   -Energiekosten + Produktionsplan einhalten

Ergebnis (Siemens-Pilotprojekt):
  20–25 % Kostensenkung durch Lastverschiebung in günstige Stunden
```

### Smart Grid: RL für das Stromnetz

```
Herausforderung Stromnetz:
  Erneuerbare Energie (Solar, Wind) = nicht planbar
  Lastspitzen = teuer und netzgefährdend
  Batteriespeicher = begrenzt

RL für Batterie-Management:
  Zustand:  Netzlast + Wettervorhersage + Batteriestand + Strompreis
  Aktion:   Batterie laden / entladen / halten
  Reward:   Netzstabilität + minimale Kosten + Batterieverschleiß minimieren

→ RL-Policy lernt: "Lade tagsüber bei niedrigen Preisen,
                    entlade abends bei Spitzenlast"
```

### Empfohlene Videos & Ressourcen

- **Google DeepMind — AI for Data Centre Cooling**  
  YouTube-Suche: `"DeepMind AI data center cooling 40 percent reduction"`  
  *(Originalvideo auf dem Google DeepMind YouTube-Kanal)*

- **Reinforcement Learning for HVAC Control (Practical Talk)**  
  YouTube-Suche: `"reinforcement learning HVAC building energy optimization"`

- **Smart Grid RL — Tesla/Powerwall Optimization**  
  YouTube-Suche: `"reinforcement learning smart grid battery energy storage"`

- **Paper:** Zhang et al. (2019) — *Building HVAC Scheduling Using RL*  
  *(Google Scholar: "reinforcement learning HVAC scheduling building control")*

- **Paper:** Evans & Gao (2016) — *DeepMind AI Reduces Google Data Centre Cooling Bill*  
  *(DeepMind Blog, auch auf Google Scholar auffindbar)*

---

## UE 5: Finanzportfolio-Management

### Überblick

Finanzmärkte sind eines der komplexesten dynamischen Systeme — Millionen von Akteuren, globale Wechselwirkungen, unvollständige Information, nicht-stationäre Statistiken.

RL modelliert Portfolio-Management als sequentiellen Entscheidungsprozess:

```
Jeden Handelstag:
  Beobachte Markt → Entscheide Allokation → Erhalte Rendite → Wiederhole

Ziel: Maximiere risikoadjustierte Rendite über Zeit
```

### RL-Formulierung: Portfolio-Optimierung

```
Zustand s:
  - Aktuelle Portfolio-Gewichtung [w₁, w₂, ..., wₙ]
  - Preishistorie der letzten k Tage (Kursänderungen, Volumen)
  - Technische Indikatoren (RSI, MACD, Bollinger Bands)
  - Makroökonomische Signale (Zinssatz, VIX)
  - Verbleibende Handelskosten (Transaktionsgebühren)

Aktion a:
  - Neue Portfolio-Gewichtung [w₁', w₂', ..., wₙ']
  - (kontinuierlich oder diskret: kaufen / halten / verkaufen)

Reward r:
  Logarithmische Portfolio-Rendite
  - Transaktionskosten
  - Risiko-Penalty (z. B. Drawdown-Strafe)
```

### Sharpe Ratio als Reward

```
Standard Reward:   r_t = (Portfolio-Wert_t - Portfolio-Wert_{t-1}) / Portfolio-Wert_{t-1}

Problem: Ignoriert Risiko — hohe Rendite mit hoher Volatilität ≠ gut

Besser: Sharpe Ratio als Reward:
  Sharpe = (E[R] - R_f) / σ(R)
         = (erwartete Überrendite) / (Standardabweichung der Rendite)

→ RL-Agent lernt: "Maximiere Rendite pro eingegangenes Risiko-Einheit"
```

### Konkretes Beispiel: Crypto-Portfolio

```
Universum: BTC, ETH, SOL, stablecoin USDT
Episode:   30 Handelstage
Zustand:   7-Tage-Kurshistorie (OHLCV) für alle 4 Assets
Aktion:    Gewichtungsvektor [w_BTC, w_ETH, w_SOL, w_USDT] mit Σw=1

RL-Strategie (gelernt):
  - Crash-Signale erkannt → große USDT-Position (defensiv)
  - Breakout erkannt → Konzentration auf führende Asset
  - Seitwärtsmarkt → gleichmäßige Diversifikation

Vergleich (Backtesting):
  Buy & Hold BTC:        +40 % (Musterperiode)
  Equal-Weight:          +28 %
  PPO-Agent:             +61 % (gleiche Periode)
  → Achtung: Overfitting-Gefahr! Backtesting ≠ Live-Performance
```

### Warnung: Overfitting und Look-Ahead Bias

```
Was oft schiefläuft:

1. Look-Ahead Bias:
   Training auf Daten t=0..T, Test auf t=T+1..T+30
   FEHLER: Normalisierung über den gesamten Zeitraum → Zukunft fließt ein

2. Transaktionskosten unterschätzt:
   1 % Transaktionskosten bei täglichem Rebalancing = 250 % p.a. Kosten!

3. Markteinfluss ignoriert:
   Große Orders bewegen den Markt (besonders bei Krypto)

4. Nicht-stationäre Märkte:
   2015 gelernte Strategie funktioniert 2021 nicht mehr
```

### High-Frequency Trading (HFT) mit RL

```
Zeitrahmen:  Millisekunden bis Sekunden
Zustand:     Order Book (Bid/Ask + Mengen), 
             jüngste Preisbewegungen
Aktion:      Market Order / Limit Order / Cancel
Reward:      Realized P&L - Market Impact

Anwender:    Citadel, Jane Street, Two Sigma
→ RL ergänzt klassische quantitative Strategien
```

### Empfohlene Videos & Ressourcen

- **Two Sigma / Quantitative Finance & ML**  
  YouTube-Suche: `"reinforcement learning stock trading portfolio management tutorial"`

- **RL for Trading — Practical Implementation (Python)**  
  YouTube-Suche: `"reinforcement learning trading Python OpenAI gym stock"`

- **Yannic Kilcher — Financial RL Papers**  
  YouTube-Suche: `"Yannic Kilcher reinforcement learning finance"`

- **FinRL Library (Open Source)**  
  GitHub: `AI4Finance-Foundation/FinRL`  
  YouTube-Suche: `"FinRL tutorial reinforcement learning finance"`

- **Paper:** Jiang et al. (2017) — *A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem*  
  *(Google Scholar: "deep reinforcement learning portfolio management Jiang 2017")*

- **Paper:** Liu et al. (2021) — *FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading*  
  *(Google Scholar: "FinRL deep reinforcement learning stock trading 2021")*

---

## UE 6: Weitere Use Cases + Ausblick

### 6.1 Gesundheitswesen

#### Personalisierte Medikamentendosierung

```
Zustand:  Patientenzustand (Laborwerte, Vitalparameter, Krankengeschichte)
Aktion:   Medikamentendosis (Art, Menge, Zeitpunkt)
Reward:   Gesundheitsverbesserung - Nebenwirkungen

Beispiel: Sepsis-Behandlung auf der Intensivstation
  Problem:  Jeder Patient reagiert anders auf Vasopressoren + Flüssigkeit
  RL:       Lernt aus historischen Patientendaten (Offline RL)
  Ergebnis: Klinische Studie zeigt RL-Policy reduziert 28-Tage-Sterblichkeit um ~3 %
  
Wichtig: RL in der Medizin = Decision Support, kein autonomes System
```

#### Strahlentherapie-Planung

```
Aufgabe:  Bestrahlungswinkel und -intensität optimieren
Ziel:     Tumor maximal bestrahlen, gesundes Gewebe schonen
RL:       Formuliert als sequentielle Entscheidung über Bestrahlungsfelder
Ergebnis: Bessere Tumor-Abdeckung bei reduzierter Nebenwirkung
```

### 6.2 Spieleindustrie & Game AI

```
Warum Spiele ideal für RL?
  ✅ Klarer Reward (Score, Sieg/Niederlage)
  ✅ Schnelle Simulation (10.000× Echtzeit)
  ✅ Einfaches Reset
  ✅ Gut definierter Zustandsraum

Meilensteine:
  1997: Deep Blue (Schach) — noch kein RL, aber Suche
  2013: DQN spielt 49 Atari-Spiele auf menschlichem Niveau
  2016: AlphaGo besiegt Lee Sedol (Go)
  2017: AlphaZero — lernt Schach/Go/Shogi in Stunden durch Self-Play
  2019: OpenAI Five besiegt Weltmeister in Dota 2
  2019: AlphaStar: Grandmaster-Level in StarCraft II
  2022: DeepMind CICERO: Diplomacy (Sprache + Strategie)
```

### 6.3 Autonomes Fahren

```
Aktuelle RL-Anwendungen (Waymo, Tesla, Cruise):
  ✅ Spurwechsel-Entscheidungen (MARL: andere Fahrzeuge als Agenten)
  ✅ Kreuzungsverhalten bei unklarer Vorfahrt
  ✅ Simulation-to-Real für seltene Szenarien (Geisterfahrer, Falschabbieger)
  ✅ Reward Shaping für komfortables Fahrverhalten

Nicht (vollständig) mit RL:
  ❌ Primäres Bewegungscontrolling (klassische Regelungstechnik zuverlässiger)
  ❌ Objekterkennung (supervised learning)

Hybridansatz:
  Perception (CV) → Zustandsschätzung → RL-Entscheidung → Regelung
```

### 6.4 Telekommunikation & Netzwerke

```
RL für 5G/6G-Netzwerkmanagement:
  - Ressourcenallokation: Welcher Nutzer bekommt welches Frequenzband?
  - Handover-Optimierung: Wann wechselt Smartphone die Basisstation?
  - Traffic-Routing: Welcher Pfad hat aktuell die geringste Latenz?

Zustand:   Netzlast, Signalqualität, Nutzerdaten
Aktion:    Frequenzband / Sendeleistung / Route
Reward:    Latenz + Throughput + Energieverbrauch

Praxis: Nokia, Ericsson nutzen RL-basierte Netzwerkoptimierung in Produktionsnetzen
```

### 6.5 Content-Empfehlungssysteme

```
Netflix, YouTube, TikTok — alle nutzen RL-ähnliche Systeme:

Problem:   Nächstes Video empfehlen
Zustand:   Nutzerhistorie + aktueller Kontext + Inhaltsmerkmale
Aktion:    Welches Video/Produkt anzeigen?
Reward:    Klick + Watchtime + Langzeit-Retention

Herausforderung "Filter Bubble":
  RL optimiert auf Klicks → lernt Empörung/Extreminhalte zu bevorzugen
  → Langzeitschaden für Nutzer (schlecht für langfristigen Reward)
  
Lösung: Multi-Objective RL (kurzfristige Clicks + langfristige Zufriedenheit)
```

### 6.6 Wissenschaft & Forschung

| Domäne | Anwendung | Ergebnis |
|--------|-----------|---------|
| **Proteinfaltung** | AlphaFold 2 (nicht RL, aber verwandt) | Revolutioniert Biologie |
| **Kernfusion** | DeepMind + Tokamak Plasma-Kontrolle | RL steuert Plasma erstmals direkt |
| **Quantencomputing** | RL für Quantum Circuit Design | Findet kürzere Quantenschaltkreise |
| **Materialwissenschaft** | RL-gesteuerte Synthesepfade | Neues Batteriematerial KI-entdeckt |
| **Mathematik** | AlphaTensor / FunSearch | Neue Matrixmultiplikations-Algorithmen |

### Ausblick: Wo geht RL hin?

```
2026 und darüber hinaus:

1. Foundation Models + RL:
   LLMs als universelle Planer → RL für präzise Ausführung
   (SayCan, RT-2 als Vorboten)

2. Real-World Deployment:
   Mehr RL in Produktionssystemen (Energienetze, Logistik)
   → Sicherheit, Erklärbarkeit, Regularisierung werden kritisch

3. Multi-Agent RL in der Wirtschaft:
   Mehrere autonome Systeme interagieren (Preisagenten, Lieferkettenagenten)
   → Emergente Marktdynamiken, Regulierungsbedarf

4. RL + Nachhaltigkeit:
   CO₂-minimierendes Routing, Kreislaufwirtschaft-Optimierung
   → RL als Werkzeug für Klimaziele

5. Personalisierte KI-Assistenten:
   Langzeit-RL für persönliche Präferenzen (nicht nur Session-basiert)
```

### Empfohlene Videos & Ressourcen für UE 6

- **AlphaGo — The Movie (Full Documentary)**  
  YouTube: `https://www.youtube.com/watch?v=WXuK6gekU1Y`  
  *(DeepMind Official — verifiziert)*

- **DeepMind AlphaStar: Mastering StarCraft II**  
  YouTube-Suche: `"DeepMind AlphaStar grandmaster StarCraft II"`

- **OpenAI Five — Dota 2 Weltmeister**  
  YouTube-Suche: `"OpenAI Five Dota 2 world champions"`

- **DeepMind Tokamak Plasma Control with RL**  
  YouTube-Suche: `"DeepMind magnetic confinement fusion reinforcement learning"`

- **Two Minute Papers — RL Highlights**  
  YouTube-Kanal: `https://www.youtube.com/@TwoMinutePapers`  
  *(Wöchentliche Zusammenfassungen von RL-Papieren mit Visualisierungen)*

- **Lex Fridman — Interviews mit RL-Forschern**  
  YouTube-Kanal: `https://www.youtube.com/@lexfridman`  
  YouTube-Suche: `"Lex Fridman reinforcement learning Sutton Silver Mnih"`

- **David Silver — Introduction to Reinforcement Learning (UCL/DeepMind)**  
  YouTube-Suche: `"David Silver reinforcement learning lecture UCL DeepMind"`  
  *(10-teilige Vorlesungsreihe — Gold-Standard für RL-Grundlagen)*

---

## Gesamtüberblick: Use Cases nach RL-Schwierigkeit

```
Einfacher einzusetzen:                          Schwieriger einzusetzen:
──────────────────────────────────────────────────────────────────────
Spiele          Empfehlungssysteme    Robotik    Medizin    Autonomes Fahren
  ↑                     ↑               ↑           ↑              ↑
Simulation      Simul. + Online      Sim2Real    Offline RL    Safety-kritisch
klarer Reward   impliziter Reward    Sicherheit  Ethik          Regulation
```

## Checkliste: RL für eigene Projekte

Bevor man RL einsetzt, prüfe:

```
✅ Ist das Problem sequenziell? (Entscheidungen beeinflussen Zukunft)
✅ Kann man eine Simulation / Umgebung bauen?
✅ Ist ein sinnvoller Reward definierbar?
✅ Reicht der Datensatz / die Interaktionszeit?
✅ Ist das Problem wirklich zu komplex für Heuristiken?

❌ Ist ein einzelnes Modell / eine Lookup-Table ausreichend?
❌ Ist der Zustandsraum zu groß ohne gute Simulation?
❌ Sind die Konsequenzen falscher Aktionen zu gravierend für Online-Lernen?

→ Wenn viele ✅: RL lohnt sich
→ Wenn viele ❌: Klassische Optimierung / ML zuerst versuchen
```

---

## Abschluss-Quiz (alle 6 UE)

1. Nenne drei Gründe, warum RL für Lieferkettenoptimierung besser geeignet ist als klassische lineare Programmierung.
2. Was ist "Surge Pricing" und wie ist es als RL-Problem formulierbar?
3. Was ist "Sim-to-Real Transfer" in der Robotik und warum ist er schwierig?
4. Welche Ergebnis erzielte DeepMind mit RL für Google-Rechenzentrum-Kühlung?
5. Was ist "Look-Ahead Bias" beim Backtesting von Finanz-RL-Agenten?
6. Warum sind Spiele ideale Testumgebungen für RL-Algorithmen?
7. Nenne zwei ethische Herausforderungen beim Einsatz von RL in der Medizin.
8. Was sind die wichtigsten Fragen der "Checkliste" vor dem RL-Einsatz?
9. Was hat AlphaGo mit RL zu tun, und warum war es ein Meilenstein?
10. Beschreibe einen Use Case aus deinem eigenen Berufsfeld, der als RL-Problem formuliert werden könnte.

---

*Quellen & weiterführende Literatur:*  
*Mnih et al. (2015) — DQN; Silver et al. (2016) — AlphaGo; Vinyals et al. (2019) — AlphaStar; Evans & Gao (2016) — DeepMind Cooling; Jiang et al. (2017) — Portfolio RL; Oroojlooyjadid et al. (2021) — Supply Chain RL; Killian et al. (2020) — Sepsis Treatment RL; Degrave et al. (2022) — Tokamak Plasma Control; Andrychowicz et al. (2020) — Dactyl*

```

**Hinweis zu den Links:** Alle Video-URLs sind bekannte, öffentliche Quellen. Die mit `YouTube-Suche:` markierten Einträge sind als Suchbegriffe angegeben — bitte die aktuellen Links vor dem Kurs verifizieren, da YouTube-URLs sich gelegentlich ändern. Die zwei expliziten URLs (AlphaGo Documentary, Two Minute Papers, Lex Fridman) sind etablierte Kanäle und sollten stabil sein.
