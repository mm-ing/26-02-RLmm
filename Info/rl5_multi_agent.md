# Erweitertes Reinforcement Learning

## Inhaltsübersicht

| UE   | Thema                                     |
|------|-------------------------------------------|
| 1–2  | Model-Based RL — Lernen mit Weltmodell    |
| 3–4  | Hierarchical RL (HRL) — Ziele in Zielen   |
| 5–6  | Meta-RL — Lernen zu lernen                |
| 7–8  | Offline / Batch RL — Lernen ohne Umgebung |
| 9–10 | Multi-Agent RL — Mehrere Agenten          |
| 11–12| Imitation & Inverse RL                    |
| 13–14| Generalist Agents / Foundation RL         |

---

## UE 1–2: Model-Based RL — Lernen mit Weltmodell

### Intuitive Erklärung

Bisher: **Model-Free RL** — der Agent lernt durch reales Ausprobieren in der Umgebung.

```

Agent → Aktion → Umgebung → Belohnung + neuer Zustand
                  (echte Welt)

```

**Problem:** Jede Erfahrung kostet Zeit und Ressourcen (z. B. echter Roboter, teures Simulator-Rendering). Für gute Policies braucht man Millionen von Schritten.

**Idee Model-Based RL:** Der Agent lernt zusätzlich ein **Weltmodell** — eine Vorhersagefunktion, die simuliert, was nach einer Aktion passiert:

```

Weltmodell:  f(s, a) → (s', r)
             "Wenn ich in Zustand s Aktion a ausführe,
              lande ich wahrscheinlich in s' und erhalte r."

```

Mit dem Weltmodell kann der Agent **im Kopf planen**, ohne echte Schritte zu machen.

```

Agent → [Weltmodell] → simulierte Erfahrungen → Policy-Update
         (gelernt)

```

### Vergleich Model-Free vs. Model-Based

| Eigenschaft | Model-Free | Model-Based |
|------------|-----------|------------|
| Sample-Effizienz | niedrig | hoch |
| Geschwindigkeit pro Update | schnell | langsamer |
| Fehlerquelle | keins | Modell-Fehler (Model Bias) |
| Beispiele | DQN, PPO | Dyna, MBPO, MuZero |

### Dyna-Q: Der klassische Ansatz

```

Jeder Schritt:

  1. Führe echten Schritt aus → (s, a, r, s') → speichere
  2. Update Q-Funktion mit echter Erfahrung
  3. n-mal: Sample (s, a) aus Erfahrungsspeicher
            → frage Weltmodell: (s', r) = f(s, a)
            → Update Q-Funktion mit simulierter Erfahrung

```

**Effekt:** n simulierte Updates pro echtem Schritt → n-fache Sample-Effizienz.

### Visualisierungsidee

```

Model-Free:                 Model-Based (Dyna):

Real world:                 Real world:
 s → a → s'                  s → a → s'
    ↓                            ↓
  Update                     Update + speichern
                                  ↓
                            Weltmodell lernt: f(s,a) → s'
                                  ↓
                            k × simulierte Updates
                            (schnelles Planen im Kopf)

→ Schachspieler, der echte Partien spielt
  vs. Schachspieler, der auch zu Hause Züge "im Kopf" analysiert

```

### Latente Weltmodelle: Dreamer / MuZero

Moderne Ansätze lernen ein **latentes Weltmodell** — nicht über rohe Pixel, sondern über komprimierte Repräsentationen:

```

Pixel → Encoder → latenter Zustand z
                       ↓
              Dynamikmodell: z, a → z'
                       ↓
              Rewardmodell: z → r

```

**MuZero (DeepMind, 2020):** Kennt die Regeln des Spiels nicht explizit — lernt sie aus Daten. Übertrumpft AlphaZero bei Atari und besiegt Weltklasse-Spieler in Schach/Go.

### Häufige Missverständnisse

❌ **„Model-Based RL ist immer besser als Model-Free."**  
✅ Modell-Fehler (Model Bias) können Policies ruinieren. Bei einfachen Aufgaben mit viel Daten ist Model-Free oft besser.

❌ **„Das Weltmodell muss perfekt sein."**  
✅ Schon ein imperfektes Modell kann Sample-Effizienz massiv steigern, solange Fehler begrenzt bleiben.

❌ **„Model-Based RL erfordert ein manuell codiertes Modell."**  
✅ Das Modell wird aus Erfahrungen gelernt — vollständig datengetrieben.

### Mini-Quiz UE 1–2

1. Was ist der Kernunterschied zwischen Model-Free und Model-Based RL?  
   *(Antwort: Model-Based lernt ein Weltmodell und plant damit; Model-Free lernt direkt aus echten Erfahrungen)*

2. Was ist "Model Bias"?  
   *(Antwort: Fehler im gelernten Weltmodell führen zu falschen Simulationen → schlechte Policy-Updates)*

3. Wie steigert Dyna-Q die Sample-Effizienz?  
   *(Antwort: n simulierte Updates pro echtem Schritt durch das Weltmodell)*

4. Was lernt MuZero, das klassische Planer wie minimax nicht können?  
   *(Antwort: Die Spielregeln / Dynamik aus Daten — ohne explizites Modell der Umgebung)*

---

## UE 3–4: Hierarchical RL (HRL) — Ziele in Zielen

### Intuitive Erklärung

**Problem beim flachen RL:** Lange Zeithorizonte mit sparse rewards.

Beispiel: "Räume dein Zimmer." Das Reward-Signal kommt erst am Ende — aber der Weg dahin hat hunderte Schritte.

```

Aktion_1, Aktion_2, ..., Aktion_500 → Reward: +1 (Zimmer aufgeräumt)

```

Das Netz muss 500 Schritte zurückverfolgen — kaum lernbar.

**HRL-Idee:** Teile das Problem in **Ebenen** auf:

```

Manager (High-Level Policy):
  "Räume zuerst den Schreibtisch auf."
       ↓ Subziel
Worker (Low-Level Policy):
  "Gehe zu Schreibtisch → Hebe Stift auf → Lege in Schublade → ..."
       ↓ primitive Aktionen
Umgebung

```

Der Manager setzt **abstrakte Subziele**, der Worker führt **primitive Aktionen** aus, um sie zu erreichen. Jede Ebene hat ihren eigenen Zeithorizont.

### Architektur: Options Framework

Ein **Option** ist ein wiederverwendbares Verhalten mit:
- **Initiation Set** $I$: Zustände, in denen die Option starten kann
- **Policy** $\pi_o$: Aktionsauswahl während der Option
- **Termination** $\beta(s)$: Wahrscheinlichkeit, die Option zu beenden

```

Option "Tür öffnen":
  I:     Agent steht vor Tür
  π_o:   Greife Türklinke → drücke herunter → schiebe
  β(s):  endet, wenn Tür offen oder Timeout

```

Semi-Markov Decision Process (SMDP): Zeitschritte sind variabel — eine Option dauert mehrere primitive Schritte.

### HIRO: Hierarchical RL with Hindsight

```

Manager (alle k Schritte):
  beobachtet s_t
  wählt Subziel g_t ∈ Zustandsraum
       ↓
Worker (jeden Schritt):
  beobachtet (s_t, g_t)
  wählt a_t ∈ primitiver Aktionsraum
  erhält intrinsischen Reward: -‖s_{t+1} - g_t‖
       ↓
Umgebung → extrinsischer Reward r_t → Manager-Update

```

**Hindsight-Relabeling:** Wenn der Worker das Subziel verfehlt, wird nachträglich das tatsächlich erreichte Subziel als "gewolltes Ziel" neu beschriftet — effizienter lernen trotz Misserfolg.

### Visualisierungsidee

```

HRL in einem Labyrinth:

Abstraktionsebene:
┌─────────────────────────────────────┐
│ Manager:  Raum A → Raum B → Ziel    │  (alle 20 Schritte)
│              ↓           ↓          │
│ Worker:   gehe links, rechts, ...   │  (jeden Schritt)
└─────────────────────────────────────┘

Ohne HRL: 200 primitive Schritte → dünnes Reward-Signal
Mit HRL:  Manager: 10 Subziele × Worker: 20 Schritte = gut strukturiert

```

### Beispiele

| Domäne | Manager-Level | Worker-Level |
|--------|-------------|-------------|
| Robotik | "Greife Objekt" | Gelenkwinkel-Steuerung |
| Navigation | "Gehe in Sektion B" | Einzelne Bewegungsschritte |
| Minecraft | "Baue Werkzeugbank" | Einzelne Block-Aktionen |
| Sprachsteuerung | "Beantworte E-Mail" | Einzelne Tastenanschläge |

### Häufige Missverständnisse

❌ **„HRL braucht handcodierte Subziele."**  
✅ Moderne HRL-Methoden (HIRO, HAC) lernen Subziele aus Daten.

❌ **„Je mehr Ebenen, desto besser."**  
✅ Mehr Ebenen = komplexeres Training, mehr Hyperparameter, schwieriger zu coordenieren. 2 Ebenen sind oft ausreichend.

❌ **„HRL löst automatisch das Credit-Assignment-Problem."**  
✅ Es verteilt es — aber Manager und Worker haben jeweils eigene Assignment-Probleme.

### Mini-Quiz UE 3–4

1. Warum ist flaches RL bei langen Zeithorizonten problematisch?  
   *(Antwort: Sparse rewards müssen über viele Schritte zurückpropagiert werden — kaum lernbar)*

2. Was ist eine "Option" im Options Framework?  
   *(Antwort: Ein wiederverwendbares Verhalten mit Startbedingung, eigener Policy und Abbruchbedingung)*

3. Was macht Hindsight-Relabeling in HIRO?  
   *(Antwort: Verfehlte Subziele werden durch das tatsächlich erreichte Ziel ersetzt → der Worker erhält trotzdem positives Feedback)*

4. Nenne ein Beispiel für Manager- und Worker-Ebene in einem Robotik-Szenario.  
   *(Antwort: Manager: "Greife Objekt"; Worker: Gelenkwinkel-Steuerung der einzelnen Motoren)*

---

## UE 5–6: Meta-RL — Lernen zu lernen

### Intuitive Erklärung

Standard-RL: Trainiere für **eine** Aufgabe von Grund auf. Wechselt die Aufgabe, beginnt alles neu.

**Meta-RL:** Trainiere einen Agenten, der **schnell** lernt — über viele verschiedene Aufgaben hinweg.

> Ziel: Nicht "was ist die beste Aktion in Aufgabe X?",  
> sondern: "wie lerne ich schnell, was die beste Aktion in einer **neuen** Aufgabe ist?"

**Analogie:**
```

Standard-RL:  Kind lernt Schach. Dann lernt es von vorne Poker.
Meta-RL:      Kind lernt "wie man Spiele lernt" — Regeln erkennen,
              Strategien aufbauen, schnell anpassen.
              → Beim nächsten Spiel reichen wenige Partien.

```

### Ansatz 1: MAML (Model-Agnostic Meta-Learning)

**Idee:** Finde Modellparameter θ, von denen aus **wenige Gradienten-Schritte** für jede neue Aufgabe reichen.

```

Meta-Training:
  Für jede Aufgabe τᵢ:
    1. Starte von θ
    2. Führe k Gradient-Schritte aus → θ'ᵢ
    3. Berechne den Performance-Verlust auf τᵢ mit θ'ᵢ
  Optimiere θ so, dass der durchschnittliche Verlust minimal ist

Meta-Test (neue Aufgabe τ_neu):
  Starte von θ → k Schritte → gute Policy

```

```

Parameterraum:
      θ (Meta-Parameter)
     /|\
    / | \
θ'₁ θ'₂ θ'₃   ← nach wenigen Schritten je Aufgabe
(τ₁)(τ₂)(τ₃)

```

θ ist ein "guter Startpunkt" im Parameterraum — nah an der Lösung vieler Aufgaben.

### Ansatz 2: RL² (RL als RNN)

**Idee:** Nutze ein **rekurrentes Netz (z. B. LSTM)**, das die gesamte Interaktionshistorie einer neuen Aufgabe im internen Zustand (hidden state) speichert.

```

Episode 1 (neue Aufgabe):
  (s₁, a₁, r₁) → LSTM → (s₂, a₂, r₂) → ... → Ende Ep.1
                   ↕
              hidden state h₁

Episode 2 (gleiche Aufgabe, h₁ als Start):
  h₁ enthält "was ich in Ep.1 gelernt habe"
  → Agent verhält sich bereits besser

```

Das LSTM *ist* der Lernalgorithmus — in seinen Gewichten sind Meta-Wissen über das Lernen kodiert.

### Vergleich der Ansätze

| Methode | Mechanismus | Stärke | Schwäche |
|---------|------------|--------|----------|
| MAML | Gradient-basiert | Allgemein, flexible Architektur | Teuer (zweite Ableitung) |
| RL² / RNN | Hidden-State | Kein explizites Fine-Tuning nötig | Begrenzte Kontextlänge |
| Prototyp-Netze | Nearest-Neighbor | Einfach, interpretierbar | Nur für klassifizierungsähnliche Tasks |

### Visualisierungsidee

```

Aufgaben-Verteilung (Training):
  Maze-1, Maze-2, ..., Maze-1000 (verschiedene Labyrinthe)

MAML:
  θ* → nach 3 Gradient-Schritten → gut in Maze-1001 (ungesehen)

RL²:
  LSTM sieht 3 Episoden in Maze-1001 → hidden state kodiert die Labyrinthtopologie
  → Episode 4: fast optimale Policy

```

### Häufige Missverständnisse

❌ **„Meta-RL ist dasselbe wie Transfer Learning."**  
✅ Transfer Learning passt ein vortrainiertes Modell an. Meta-RL optimiert explizit die Lerngeschwindigkeit über Aufgaben.

❌ **„Meta-RL funktioniert für jede beliebige neue Aufgabe."**  
✅ Nur für Aufgaben aus der gleichen **Aufgaben-Verteilung** (Task Distribution) wie im Training.

❌ **„MAML braucht nur einen Gradient-Schritt."**  
✅ Ein Schritt ist die Minimalvariante — in der Praxis werden mehrere Schritte verwendet (First-Order MAML als Näherung).

### Mini-Quiz UE 5–6

1. Was ist das Ziel von Meta-RL im Gegensatz zu Standard-RL?  
   *(Antwort: Nicht eine spezifische Aufgabe lösen, sondern lernen, wie man neue Aufgaben schnell lernt)*

2. Was bedeutet "guter Startpunkt" in MAML?  
   *(Antwort: Modellparameter θ*, von denen aus wenige Gradient-Schritte für jede Aufgabe reichen)*

3. Wie "lernt" das LSTM in RL² eine neue Aufgabe?  
   *(Antwort: Die Interaktionshistorie wird im Hidden State gespeichert — das LSTM passt sich in-context an)*

4. Welche Einschränkung gilt für Meta-RL bei neuen Aufgaben?  
   *(Antwort: Neue Aufgaben müssen aus der gleichen Verteilung stammen, auf der trainiert wurde)*

---

## UE 7–8: Offline / Batch RL — Lernen ohne Umgebung

### Intuitive Erklärung

**Online RL:** Agent interagiert aktiv mit der Umgebung → sammelt neue Daten → lernt.

**Offline RL (Batch RL):** Fester Datensatz aus vergangenen Interaktionen — **keine neuen Umgebungsinteraktionen** erlaubt.

```

Online RL:          Offline RL:
     Umgebung           [Datensatz D]
       ↕                     ↓
     Agent           Agent lernt nur
   (interaktiv)      aus D — kein Feedback

```

**Anwendungsfälle:**
- Medizin: Patientendaten aus Krankenhäusern (keine ethisch vertretbaren Experimente)
- Autonomes Fahren: Millionen Fahrkilometer aufgezeichnet
- Finanzwesen: Historische Handelsdaten
- Industrie: Sensordaten von Produktionsanlagen

### Das Kernproblem: Distributional Shift

**Problem:** Der Policy-Optimierer möchte Aktionen ausprobieren, die im Datensatz selten oder gar nicht vorkommen. Das Modell hat über diese Bereiche keine verlässlichen Q-Schätzungen.

```

Datensatz D enthält:  Aktion A oft (gut bekannt)
                      Aktion B selten
                      Aktion C nie

Gelerntes Q:  Q(s, A) = 2.5  ← zuverlässig
              Q(s, B) = 1.2  ← unsicher
              Q(s, C) = 9.8  ← FALSCH (extrapoliert)

Policy wählt: C  ← katastrophal, da nie beobachtet

```

Der Agent "halluziniert" gute Werte für unbekannte Aktionen — **Extrapolationsfehler**.

### Lösungsansätze

#### 1. Conservative Q-Learning (CQL)

Minimiere Q-Werte für Aktionen **außerhalb** des Datensatzes:

```

CQL-Loss = Standard-Bellman-Loss
         + α · E[Q(s, a) für a ~ uniform]   ← drückt OOD-Werte runter
         - α · E[Q(s, a) für a aus Datensatz] ← hält Datensatz-Werte oben

```

Resultat: Q-Funktion ist konservativ — keine Überschätzung außerhalb der Datenverteilung.

#### 2. Behavior Cloning als Regularisierung (BCQ, TD3+BC)

Kombiniere RL-Optimierung mit **Behavior Cloning** (nachahmen der Datensatz-Policy):

```

Policy-Update = RL-Gradient + λ · (Policy bleibt nah an Datensatz-Verteilung)

```

#### 3. Decision Transformer

Behandle Offline-RL als **Sequence-Modelling-Problem**:

```

Eingabe:  [R_1, s_1, a_1, R_2, s_2, a_2, ..., R_t, s_t, ?]
                                                           ↑
Ausgabe:  a_t  (Aktion, die das gewünschte Return R_1 erreicht)

```

Der Transformer lernt: "Wenn Return-to-Go = 100 gewünscht, welche Aktionen führen dahin?" — kein Bellman-Update, reine Sequenzvorhersage.

### Visualisierungsidee

```

Offline RL — Datensatz-Abdeckung:

Zustandsraum (2D):
  ┌─────────────────────┐
  │ ░░░░░░░░░░░░░░░░░░░ │  ← gut abgedeckt (viele Daten)
  │ ░░░░░░░░░░░░░░░░░░░ │
  │         ████        │  ← unbekannte Region
  │         ████        │  ← Policy will hier hin → Extrapolationsfehler
  └─────────────────────┘

CQL:  "Ich trau mir in der schwarzen Region nicht zu,
       also nehme ich konservativ niedrige Q-Werte an."

```

### Häufige Missverständnisse

❌ **„Mehr Daten lösen das Distributional-Shift-Problem."**  
✅ Nur wenn die Daten die relevanten Zustands-Aktions-Paare ausreichend abdecken. Viele Daten in falschen Bereichen helfen nicht.

❌ **„Offline RL ist dasselbe wie Supervised Learning."**  
✅ Supervised Learning maximiert Vorhersagegenauigkeit. Offline RL optimiert kumulative Belohnungen — mit temporaler Abhängigkeit und Bellman-Gleichungen.

❌ **„Decision Transformer ist ein RL-Algorithmus."**  
✅ Es ist Sequence Modelling — kein Bellman-Update, keine Policy-Gradienten. Es ist RL-motiviert, aber nicht RL im klassischen Sinne.

### Mini-Quiz UE 7–8

1. Was ist der Hauptunterschied zwischen Online- und Offline-RL?  
   *(Antwort: Online-RL interagiert mit der Umgebung; Offline-RL lernt ausschließlich aus einem festen Datensatz)*

2. Was ist "Extrapolationsfehler" in Offline-RL?  
   *(Antwort: Das Modell schätzt Q-Werte für Aktionen, die im Datensatz nicht vorkommen — diese Schätzungen sind unzuverlässig und oft zu hoch)*

3. Wie bekämpft CQL den Extrapolationsfehler?  
   *(Antwort: Es minimiert Q-Werte für Aktionen außerhalb des Datensatzes und maximiert sie für Datensatz-Aktionen)*

4. Warum eignet sich Offline-RL besonders für medizinische Anwendungen?  
   *(Antwort: Echte Experimente an Patienten sind ethisch nicht vertretbar — vorhandene Behandlungsdaten können genutzt werden)*

---

## UE 9–10: Multi-Agent Reinforcement Learning (MARL)

### Intuitive Erklärung

Bisher: **Ein Agent** in einer Umgebung.

**MARL:** Mehrere Agenten interagieren gleichzeitig — miteinander und mit der Umgebung.

```

Single-Agent:           Multi-Agent:

    Agent               Agent₁  Agent₂
      ↕                    ↕       ↕
  Umgebung             Umgebung (geteilt)
                         Agent₃

```

**Neue Komplexität:** Die Umgebung ist aus Sicht jedes Agenten **nicht-stationär** — während Agent₁ lernt, ändert Agent₂ sein Verhalten → was gestern optimal war, ist heute falsch.

### Drei grundlegende Szenarien

| Typ | Beschreibung | Reward-Struktur | Beispiele |
|-----|-------------|----------------|---------|
| **Kooperativ** | Alle Agenten arbeiten zusammen | Gemeinsamer Reward | Schwarmroboter, Netzwerk-Routing |
| **Kompetitiv** | Agenten konkurrieren | Gegensätzliche Rewards (Zero-Sum) | Schach, Poker, StarCraft |
| **Mixed** | Kooperation innerhalb, Konkurrenz zwischen Teams | Gemischt | MOBA-Spiele (LoL), Fußball |

### Zentrales Training, dezentrale Ausführung (CTDE)

**Problem:** Während des Trainings würde zentralisierte Information helfen (alle Agenten beobachten alles). Aber im echten Einsatz hat jeder Agent nur seine lokale Beobachtung.

**Lösung (CTDE):**

```

Training:
  Zentrale Kritik nutzt globale Information:
  V(s₁, s₂, ..., sₙ, a₁, a₂, ..., aₙ)  ← alles bekannt

Ausführung:
  Jeder Agent hat nur lokale Policy:
  πᵢ(aᵢ | oᵢ)  ← nur eigene Beobachtung oᵢ

```

**MADDPG** (Multi-Agent DDPG): Jeder Agent hat einen Critic, der alle Aktionen und Beobachtungen kennt — aber einen Actor, der nur die eigene Beobachtung nutzt.

### QMIX: Kooperatives MARL

Für vollständig kooperative Szenarien:

```

Jeder Agent i hat eine lokale Q-Funktion:  Qᵢ(oᵢ, aᵢ)
QMIX kombiniert sie zu einem globalen Q:
  Q_total = Mixing Network(Q₁, Q₂, ..., Qₙ, s)

Constraint: ∂Q_total/∂Qᵢ ≥ 0 (Monotonie)
→ Maximiere global Q durch greedy-Aktionswahl lokal

```

**Monotonie-Constraint** garantiert: Wenn Agent i lokal eine bessere Aktion wählt, verbessert sich auch der globale Q-Wert.

### Self-Play und Emergenz

**Kompetitives Training durch Self-Play:**

```

Agent v0 spielt gegen Agent v0  →  lernt Basis-Strategie
Agent v1 (gelernt) spielt gegen v1  →  lernt Gegenstrategien
...
Agent vN  →  emergente, komplexe Strategien

```

**AlphaStar (DeepMind):** Lernt StarCraft II ausschließlich durch Self-Play → übertrifft menschliche Grandmaster.

**Emergente Kommunikation:** In kooperativen Settings entwickeln Agenten manchmal eigene Kommunikationssprachen — ohne explizit dafür trainiert zu werden.

### Visualisierungsidee

```

Kooperatives MARL — Netzwerk-Routing:

Paket muss von A nach E:
  A → B → C → E  (Route 1)
  A → D → E      (Route 2)

Agent B entscheidet: "Leite weiter über C"
Agent D entscheidet: "Leite weiter über E"
→ Ihre Entscheidungen beeinflussen sich gegenseitig
→ Stationaritätsproblem: B ändert Strategie während D lernt
→ CTDE: Zentraler Critic kennt beide Agenten-Zustände beim Training

```

### Häufige Missverständnisse

❌ **„MARL = viele unabhängige RL-Agenten parallel trainieren."**  
✅ Unabhängiges Training ignoriert die Nicht-Stationarität — Konvergenz ist nicht garantiert.

❌ **„Mehr Agenten = schnelleres Lernen."**  
✅ Mehr Agenten = mehr Interaktionen, aber auch mehr Nicht-Stationarität und Koordinationsaufwand.

❌ **„Self-Play führt immer zu optimalen Strategien."**  
✅ Self-Play kann in lokalen Nash-Gleichgewichten steckenbleiben oder zyklisches Verhalten erzeugen.

### Mini-Quiz UE 9–10

1. Was ist das Nicht-Stationaritätsproblem in MARL?  
   *(Antwort: Während ein Agent lernt, ändern andere Agenten ihr Verhalten → die Umgebung ist aus Sicht eines Agenten instabil)*

2. Was bedeutet CTDE?  
   *(Antwort: Centralized Training, Decentralized Execution — zentralisiertes Training mit globalem Wissen, aber dezentrale Ausführung mit lokaler Beobachtung)*

3. Was garantiert der Monotonie-Constraint in QMIX?  
   *(Antwort: Wenn ein Agent lokal eine bessere Aktion wählt, verbessert sich zwingend auch der globale Q-Wert)*

4. Beschreibe ein Mixed-Szenario (kooperativ + kompetitiv).  
   *(Antwort: z. B. zwei Teams in einem MOBA — innerhalb des Teams kooperativ, zwischen Teams kompetitiv)*

---

## UE 11–12: Imitation Learning & Inverse RL

### Intuitive Erklärung

**Problem mit Standard-RL:** Reward-Engineering ist schwierig. Wie beschreibt man "fahre wie ein erfahrener Fahrer" als Reward-Funktion?

**Alternative:** Lerne direkt aus **Demonstrationen** eines Experten.

```

Experte (Mensch):  s₁ → a₁, s₂ → a₂, ..., sₙ → aₙ
                   (beobachtete Trajektorien)
                        ↓
Lernender Agent:   lerne Policy π: s → a
                   (ohne Reward-Funktion!)

```

### Ansatz 1: Behavior Cloning (BC)

**Idee:** Behandle Imitation als **Supervised Learning**:

```

Datensatz: {(s₁, a₁), (s₂, a₂), ..., (sₙ, aₙ)}
Ziel:      π(s) ≈ a_Experte  (minimiere Cross-Entropy / MSE)

```

**Einfach zu implementieren** — aber mit gravierendem Problem:

**Compounding-Fehler (Covariate Shift):**

```

Experte:  s₀ → a₀ → s₁ → a₁ → s₂ → ...
                              ↑
Agent:    s₀ → a₀' (kleiner Fehler) → s₁' (leicht abgewichen)
                                          ↓
                               s₁' nicht im Trainingsdatensatz!
                               → Fehler akkumulieren sich → Katastrophe

```

Lösung: **DAgger (Dataset Aggregation)** — frage den Experten interaktiv nach Korrekturen für die tatsächlich besuchten Zustände.

### Ansatz 2: Inverse Reinforcement Learning (IRL)

**Frage:** Was ist die **Reward-Funktion**, die das Experten-Verhalten erklärt?

```

Standard-RL:   Reward → Policy
IRL:           Experten-Policy → Reward → (dann: RL mit gelerntem Reward)

```

**MaxEntIRL:** Finde Reward R, sodass die Experten-Policy die wahrscheinlichste unter allem möglichen Verhalten ist (Maximum-Entropy-Prinzip).

### Ansatz 3: GAIL (Generative Adversarial Imitation Learning)

Kombiniert Imitation Learning mit dem **GAN-Prinzip**:

```

Generator = Policy π (Agent)
Discriminator D = "Kann ich unterscheiden, ob Trajektorie vom Experten oder Agenten stammt?"

Trainingsschleife:

  1. Agent generiert Trajektorie τ_π
  2. D(τ) → 1 wenn Experte, 0 wenn Agent
  3. Agent optimiert: "Täusche D — mache Trajektorien ununterscheidbar"
  4. D verbessert sich: "Unterscheide besser"
  → Nash-Gleichgewicht: Agent imitiert Experten perfekt

```

**Vorteil gegenüber BC:** Kein Distributional-Shift-Problem, da der Agent mit der Umgebung interagiert.

### Vergleich der Ansätze

| Methode | Interaktion nötig? | Reward-Lernen | Skalierbarkeit |
|---------|-------------------|--------------|----------------|
| Behavior Cloning | Nein | Nein | Hoch (einfach) |
| DAgger | Ja (Experten-Feedback) | Nein | Mittel |
| MaxEntIRL | Ja | Ja | Niedrig (teuer) |
| GAIL | Ja | Implizit (Discriminator) | Hoch |

### Praktisches Beispiel: Autonomes Fahren

```

Ansatz BC:
  Aufzeichnung: 1000 Stunden menschliches Fahren
  Training: π(Bild) → Lenkwinkel, Geschwindigkeit
  Problem: Randstreifen oder Baustellen → unbekannte Situation → Fehler

Ansatz GAIL:
  Discriminator: "Wirkt das wie menschliches Fahren?"
  Agent lernt: "Täusche den Discriminator auf neuen Strecken"
  → Generalisiert besser auf neue Situationen

```

### Häufige Missverständnisse

❌ **„Behavior Cloning ist ausreichend für komplexe Aufgaben."**  
✅ Compounding-Fehler machen BC für lange Horizonte unzuverlässig — DAgger oder GAIL sind robuster.

❌ **„IRL rekonstruiert die genau richtige Reward-Funktion."**  
✅ Viele Reward-Funktionen erklären das gleiche Verhalten (Ambiguität). IRL findet eine mögliche, nicht die einzige.

❌ **„GAIL benötigt keine Reward-Funktion."**  
✅ Es benötigt keine explizite Reward-Funktion — der Discriminator entspricht implizit einer gelernten Reward-Funktion.

### Mini-Quiz UE 11–12

1. Was ist der Unterschied zwischen Behavior Cloning und IRL?  
   *(Antwort: BC lernt direkt die Policy durch Supervised Learning; IRL lernt zuerst die Reward-Funktion, die das Experten-Verhalten erklärt)*

2. Was ist der Compounding-Fehler bei Behavior Cloning?  
   *(Antwort: Kleine Fehler führen zu leicht abweichenden Zuständen, die nicht im Trainingsdatensatz sind — Fehler akkumulieren sich exponentiell)*

3. Wie funktioniert GAIL analog zu einem GAN?  
   *(Antwort: Der Agent ist der Generator, ein Discriminator unterscheidet Experten- von Agenten-Trajektorien; der Agent optimiert, den Discriminator zu täuschen)*

4. Wann würdest du IRL gegenüber GAIL bevorzugen?  
   *(Antwort: Wenn die gelernte Reward-Funktion selbst interessant/interpretierbar sein soll, z. B. für Analyse oder Transfer auf neue Aufgaben)*

---

## UE 13–14: Generalist Agents & Foundation RL

### Intuitive Erklärung

**Bisheriger Stand:** Pro Aufgabe ein spezialisierter Agent.

```

DQN für Atari-Pong     (nur Pong)
AlphaGo für Go         (nur Go)
OpenAI Five für Dota 2 (nur Dota)

```

**Vision der Generalist Agents:** Ein einzelnes Modell, das **viele verschiedene Aufgaben** löst — ohne aufgabenspezifisches Fine-Tuning.

```

Generalist Agent:
  Input: [Aufgabenbeschreibung, Zustand, Geschichte]
  Output: Aktion (für jede Aufgabe geeignet)

Aufgaben: Spielen, Navigieren, Greifen, Sprachbefehle, ...

```

### Gato (DeepMind, 2022)

Gato ist ein **Multi-Modal Transformer**, der als Generalist fungiert:

- **Eingaben:** Bilder, Text, diskrete Tokens, kontinuierliche Vektoren
- **Ausgaben:** Aktionen (Joystick, Text, Roboter-Gelenke)
- **Training:** 604 verschiedene Aufgaben gleichzeitig

```

Gato tokenisiert alles:
  Bild:        → Bildpatches → Tokens
  Text:        → BPE-Tokens
  Aktion:      → diskrete/kontinuierliche Tokens
  Belohnung:   → numerischer Token

Dann: Autoregressive Vorhersage des nächsten Tokens
      (genau wie ein Sprachmodell, aber für alles)

```

**Ergebnisse:** Gato spielt Atari, steuert Roboterarme, führt Gespräche und löst bildbasierte Aufgaben — alles mit einem Modell (1,2 Milliarden Parameter).

### RT-2 (Google DeepMind, 2023): Vision-Language-Action Model

**Idee:** Nutze das Wissen eines vortrainierten Vision-Language-Modells (VLM) für Robotersteuerung:

```

Vortrainiertes VLM:
  "Ein Apfel liegt links vom Messer."
                ↓ Fine-Tuned als RT-2
  "Greife den Apfel" → Roboter-Aktions-Tokens [move_left, close_gripper, ...]

```

**Emergente Fähigkeiten:** Ohne explizites Training kann RT-2 Anweisungen wie "Lege den gefährlichsten Gegenstand in die Box" interpretieren — weil das VLM-Vorwissen über "gefährlich" transferiert wird.

### Foundation Models für RL

**Paradigma:** Statt RL von Grund auf — große vortrainierte Modelle als **Basisschicht** nutzen:

```

Ebene 3: Fein-Abstimmung für spezifische Aufgabe
Ebene 2: RL-Policy-Kopf (wenige Layer)
Ebene 1: Foundation Model (GPT, CLIP, etc.)
          ↑ vortrainiert auf Milliarden von Daten

```

**Vorteile:**
- Riesige Wissensbasis (Physik, Alltagswissen, Sprache)
- Weniger Daten für neue Aufgaben nötig
- Zero-Shot / Few-Shot auf neue Aufgaben

### SayCan (Google, 2022): LLM + RL

```

Aufgabe: "Ich brauche etwas Kaltes zum Trinken."

LLM (PaLM):
  → Generiert mögliche Aktionssequenzen:
    "Gehe zum Kühlschrank → Öffne Tür → Nimm Getränk"

RL-Policy (Affordance Model):
  → Bewertet: "Was kann der Roboter physisch ausführen?"
  → p(Erfolg | Aktion, aktueller Zustand)

Kombination:
  Ausführe Sequenz mit höchster (LLM-Wahrscheinlichkeit × RL-Ausführbarkeit)

```

### Aktuelle Herausforderungen

| Herausforderung | Beschreibung |
|----------------|-------------|
| **Catastrophic Forgetting** | Lernen neuer Aufgaben überschreibt altes Wissen |
| **Evaluation** | Wie misst man "Generalismus"? |
| **Sample-Effizienz** | Generalist-Training braucht immense Datenmengen |
| **Safety** | Ein mächtiger Generalist kann bei falschen Zielen großen Schaden anrichten |
| **Reward Specification** | Wie definiert man Reward für tausende Aufgaben? |

### Vergleich: Spezialist vs. Generalist

```

Performance

  Spezialist: 100%  ████████████████████ (eine Aufgabe)
  Generalist:  85%  █████████████████     (in der Aufgabe)
                                          + 600 weitere Aufgaben

→ Kein Spezialist für alle 600 Aufgaben gleichzeitig
→ Generalist: akzeptable Performance breit skaliert

```

### Visualisierungsidee

```

Gato — Ein Modell, viele Aufgaben:

┌─────────────────────────────────────────────────────┐
│              GATO (Transformer)                     │
├──────────┬──────────┬────────────┬──────────────────┤
│  Atari   │  Robotik │   Dialog   │  Bildklassif.    │
│  Pong    │  Greifen │  Chatbot   │  ImageNet        │
│  Breakout│  Stapeln │  Q&A       │  COCO-Caption    │
└──────────┴──────────┴────────────┴──────────────────┘
     ↑ Tokens    ↑ Gelenk-Tokens  ↑ Text-Tokens  ↑ Bild-Tokens
     └──────────── alles unified als Token-Sequenz ──────────────┘

```

### Häufige Missverständnisse

❌ **„Gato ersetzt spezialisierte Agenten in allen Domänen."**  
✅ Gato zeigt Machbarkeit, übertrifft aber Spezialisten meist nicht in ihrer Domäne.

❌ **„Foundation RL = LLM + ein bisschen RL."**  
✅ Die Integration ist komplex: Aktionsräume, zeitliche Abhängigkeiten und Reward-Signale sind fundamental anders als Text-Vorhersage.

❌ **„Generalist Agents sind bereits produktionsreif."**  
✅ Stand 2026: Aktives Forschungsfeld — robuste, sichere Generalist-Agenten für echte Produktionsumgebungen sind noch nicht ausgereift.

### Mini-Quiz UE 13–14

1. Was unterscheidet einen Generalist Agent von einem spezialisierten RL-Agenten?  
   *(Antwort: Ein Generalist löst viele verschiedene Aufgaben mit einem einzigen Modell; ein Spezialist ist auf eine Aufgabe optimiert)*

2. Wie tokenisiert Gato verschiedene Eingabetypen?  
   *(Antwort: Bilder → Patches, Text → BPE-Tokens, Aktionen → diskrete Tokens — alles wird in eine einheitliche Token-Sequenz umgewandelt)*

3. Wie kombiniert SayCan ein LLM mit einem RL-Agenten?  
   *(Antwort: LLM generiert plausible Aktionssequenzen; RL-Policy bewertet physische Ausführbarkeit; beide werden multipliziert für die finale Auswahl)*

4. Was ist Catastrophic Forgetting und warum ist es bei Generalist Agents kritisch?  
   *(Antwort: Lernen neuer Aufgaben überschreibt Gewichte für alte Aufgaben — bei einem Generalist würde das die Allgemeinheit zerstören)*

---

## Gesamtüberblick: Erweitertes RL — Wohin geht die Reise?

```

Problem → Lösung

Zu viele echte Interaktionen nötig?
    → Model-Based RL (Weltmodell, Dyna, MuZero)

Langer Zeithorizont, sparse Rewards?
    → Hierarchical RL (Options, HIRO, HAC)

Schnell auf neue Aufgaben adapieren?
    → Meta-RL (MAML, RL²)

Nur historische Daten verfügbar?
    → Offline RL (CQL, Decision Transformer)

Mehrere Agenten interagieren?
    → Multi-Agent RL (MADDPG, QMIX, Self-Play)

Reward-Funktion unklar, Experte verfügbar?
    → Imitation / Inverse RL (BC, GAIL, IRL)

Eine KI für alles?
    → Generalist Agents (Gato, RT-2, Foundation RL)

```

### Verbindungen zwischen den Themen

```

Meta-RL ←──────────────────── Offline RL
   ↑                               ↑
   │ (schnelles Adaptieren)        │ (aus Daten lernen)
   │                               │
HRL ─────── MARL ──────────── Imitation RL
   ↑            ↑                  ↑
   │            │                  │
   └── Alle münden in ─────────────┘
            ↓
   Generalist Foundation RL

```

---

## Abschluss-Quiz (alle 14 UE)

1. Erkläre den Unterschied zwischen Model-Free und Model-Based RL in einem Satz.
2. Was ist "Model Bias" und wann ist er gefährlich?
3. Erkläre das Options-Framework in HRL mit einem Alltagsbeispiel.
4. Was ist das Ziel von Meta-RL (nicht: wie funktioniert es, sondern: wozu)?
5. Warum ist Offline RL besonders relevant in der Medizin?
6. Was ist der Distributional-Shift in Offline RL?
7. Nenne drei Szenarien in MARL (kooperativ, kompetitiv, mixed) mit je einem Beispiel.
8. Was bedeutet CTDE und warum ist es sinnvoll?
9. Erkläre den Compounding-Fehler bei Behavior Cloning.
10. Was lernt Inverse RL im Gegensatz zu Behavior Cloning?
11. Wie kombiniert GAIL einen Generator mit einem Discriminator?
12. Was ist ein Generalist Agent und welche aktuellen Vertreter kennst du?
13. Beschreibe, wie SayCan LLM-Wissen mit RL verbindet.
14. Was ist Catastrophic Forgetting und wie hängt es mit Generalist Agents zusammen?

---

*Quellen: Sutton & Barto (2018), Deisenroth et al. (2011) — Dyna, Ha & Schmidhuber (2018) — World Models, Hafner et al. (2020) — Dreamer, Schrittwieser et al. (2020) — MuZero, Nachum et al. (2018) — HIRO, Finn et al. (2017) — MAML, Duan et al. (2016) — RL², Levine et al. (2020) — Offline RL Survey, Kumar et al. (2020) — CQL, Chen et al. (2021) — Decision Transformer, Lowe et al. (2017) — MADDPG, Rashid et al. (2018) — QMIX, Vinyals et al. (2019) — AlphaStar, Ross et al. (2011) — DAgger, Ho & Ermon (2016) — GAIL, Reed et al. (2022) — Gato, Brohan et al. (2023) — RT-2, Ahn et al. (2022) — SayCan*
```
