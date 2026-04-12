# Deep Q-Learning — Kursmaterial (14 × 45 Min.)

## Inhaltsübersicht

| UE | Thema |
|----|-------|
| 1–2 | Vom Q-Learning zum Problem der Skalierbarkeit |
| 3–4 | Neuronale Netze als Funktionsapproximatoren — Baseline DQN |
| 5 | Instabilitäten & warum DQN zwei Tricks braucht |
| 6–7 | Double DQN — Q-Überschätzung eliminieren |
| 8–9 | Dueling DQN — V(s) und A(s,a) trennen |
| 10–11 | Prioritized Experience Replay (PER) |
| 12 | Noisy Nets — Exploration durch Stochastik |
| 13 | Rainbow — alle 6 Komponenten kombiniert |
| 14 | Explorations-Strategien im Überblick & situationsgerecht wählen |

---

## UE 1–2: Vom Q-Learning zur Notwendigkeit von DQN

### Intuitive Erklärung

Q-Learning speichert für jedes **(Zustand, Aktion)**-Paar einen Wert in einer Tabelle:

```
Q-Tabelle:
         links  rechts  springen
Zustand A: 0.1    0.9     0.0
Zustand B: 0.3    0.2     0.8
...
```

Das funktioniert wunderbar, wenn es **wenige Zustände** gibt — z. B. ein 4×4-Gitter (16 Zustände × 4 Aktionen = 64 Einträge).

**Das Problem:** Echte Umgebungen explodieren in der Zustandsanzahl.

| Umgebung | Zustandsraum |
|----------|-------------|
| 4×4-Gitter | 16 Zustände |
| Schachbrett | ~10^43 Stellungen |
| Atari Pong (Bildpixel) | ~10^33,000 mögliche Frames |

Eine Tabelle für Atari wäre unmöglich zu speichern, zu befüllen oder zu lernen. Wir brauchen eine **Funktion**, die Q-Werte schätzt — auch für Zustände, die wir noch nie gesehen haben.

> **Kernidee:** Statt einer Tabelle trainieren wir ein neuronales Netz:  
> Q(s, a; θ) ≈ Q*(s, a)

### Visualisierungsidee

```
Tabellarisches Q-Learning:          Deep Q-Network:

Zustand → Tabellenzeile → Wert      Zustand → [Netz] → Q-Werte für alle Aktionen
                                               θ₁,θ₂,...
"Lookup"                            "Generalisierung"
```

Zeige einen konkreten Atari-Frame (z. B. Pong) und frage: „Wie viele Pixel hat ein Frame? Wie viele mögliche Frames gibt es?"

### Wiederholung: Q-Learning-Grundformel

Q(s,a) ← Q(s,a) + α [ r + γ · max_a' Q(s',a') - Q(s,a) ]
                       └─────────── TD-Fehler ───────────┘

- r: erhaltene Belohnung
- γ: Diskontfaktor (Wichtigkeit der Zukunft)
- α: Lernrate
- TD-Fehler: Wie weit lag unsere Schätzung daneben?

### Häufige Missverständnisse

❌ **„Q-Learning ist veraltet und DQN ersetzt es komplett."**  
✅ DQN *ist* Q-Learning — nur mit einem Netz statt einer Tabelle. Die Update-Formel ist dieselbe.

❌ **„Je mehr Zustände, desto besser lernt Q-Learning."**  
✅ Mehr Zustände = mehr unbesuchte Felder = Q-Tabelle bleibt leer. DQN generalisiert über ungesehene Zustände.

### Mini-Quiz UE 1–2

1. Ein Roboter navigiert in einem Lager mit 10.000 Zellen, kann 8 Aktionen ausführen. Wie viele Einträge bräuchte die Q-Tabelle?  
   *(Antwort: 80.000)*

2. Was ist der TD-Fehler intuitiv?  
   *(Antwort: Die Differenz zwischen der aktuellen Schätzung des Q-Werts und dem, was wir nach einem Schritt besser wissen)*

3. Warum reicht eine Q-Tabelle für Atari nicht?  
   *(Antwort: Der Zustandsraum aus Pixeln ist astronomisch groß; jeder Frame wäre ein einzigartiger Zustand)*

---

## UE 3–4: Baseline DQN — Neuronales Netz als Q-Funktion

### Intuitive Erklärung

**Architektur (Mnih et al., 2015 — DeepMind):**

```
Eingabe: 4 gestapelte Graustufenframes (84×84 px)
    ↓
Conv-Layer 1: 32 Filter, 8×8, Stride 4  → lokale Merkmale
    ↓
Conv-Layer 2: 64 Filter, 4×4, Stride 2  → komplexere Muster
    ↓
Conv-Layer 3: 64 Filter, 3×3, Stride 1  → feine Details
    ↓
Fully Connected: 512 Neuronen
    ↓
Ausgabe: Q(s, a) für jede mögliche Aktion (z. B. 18 bei Atari)
```

Das Netz lernt gleichzeitig: **Merkmale aus Bildern extrahieren** *und* **Q-Werte schätzen**.

### Die zwei entscheidenden Tricks

Ohne diese Tricks divergiert das Training:

#### Trick 1: Experience Replay

**Problem ohne Replay:** Aufeinanderfolgende Erfahrungen sind stark korreliert. Das Netz überanpasst sich an die aktuelle Situation.

**Lösung:** Speichere Erfahrungen (s, a, r, s') in einem **Replay Buffer** (z. B. 1 Million Einträge). Beim Training: ziehe zufällige Mini-Batches.

```
Replay Buffer:
[Erfahrung_1042, Erfahrung_7, Erfahrung_999, Erfahrung_234, ...]
         ↑ zufällig gemischt → kein zeitlicher Zusammenhang
```

#### Trick 2: Target Network

**Problem:** Target und Vorhersage nutzen dieselben Gewichte θ → Target bewegt sich mit → Training divergiert.

**Lösung:** Separates **Target Network** mit eingefrorenen Gewichten θ⁻, das nur alle C Schritte aktualisiert wird:

$$Target = r + γ · max_a' Q(s', a'; θ⁻)$$

### Visualisierungsidee

```
Online Network θ         Target Network θ⁻
[lernt ständig]          [wird alle 1000 Schritte 
                          von θ kopiert]
        ↕                       ↕
   Q-Schätzung              stabile Ziele

→ Wie eine Lehrerin (θ⁻), die dem Schüler (θ) stabile Aufgaben stellt.
```

### Trainingsschleife DQN

```python
for Schritt in Trainingsschleife:
    1. Wähle Aktion mit ε-Greedy aus Q(s, ·; θ)
    2. Führe Aktion aus → erhalte (s, a, r, s')
    3. Speichere (s, a, r, s') im Replay Buffer
    4. Ziehe zufälligen Mini-Batch aus Buffer
    5. Berechne Target: y = r + γ · max_a' Q(s', a'; θ⁻)
    6. Update θ: minimiere (y - Q(s,a;θ))²
    7. Alle C Schritte: θ⁻ ← θ
```

### Häufige Missverständnisse

❌ **„Ohne Experience Replay könnte man immer noch lernen, nur langsamer."**  
✅ Nein — stark korrelierte Samples führen aktiv zu Divergenz, nicht nur zu langsamem Lernen.

❌ **„Das Target Network ist ein zweites, unabhängig trainiertes Netz."**  
✅ Es hat dieselbe Architektur und wird regelmäßig aus dem Online-Netz kopiert — es lernt nicht selbst.

### Mini-Quiz UE 3–4

1. Warum werden 4 Frames gestapelt statt nur 1 Frame genommen?  
   *(Antwort: Bewegungsrichtung und -geschwindigkeit sind in einem einzelnen Frame nicht sichtbar)*

2. Wie oft wird das Target Network üblicherweise aktualisiert?  
   *(Antwort: z. B. alle 1.000 oder 10.000 Trainingsschritte)*

3. Was passiert ohne Experience Replay, wenn der Agent eine Wand trifft?  
   *(Antwort: Er lernt intensiv, Wände zu meiden — überschreibt dabei das Wissen über andere Situationen)*

---

## UE 5: Instabilitäten im DQN-Training verstehen

### Intuitive Erklärung

| Problem | Ursache | Lösung ... |
|---------|---------|-------------------|
| Q-Überschätzung | max-Operator wählt höchsten Q-Wert, auch verrauschte | Double DQN  |
| Schlechte Generalisierung | Netz kann V(s) und A(s,a) nicht trennen | Dueling DQN  |
| Uniform Sampling ineffizient | Seltene, wichtige Erfahrungen kaum genutzt | PER  |
| Schlechte Exploration | Parametrisierter Zufall statt netzbasierter Neugier | Noisy Nets  |

---

## UE 6–7: Double DQN — Q-Überschätzung eliminieren

### Intuitive Erklärung

**Das Problem:** Im Standard-DQN-Target benutzt `max` denselben Netz-Pass für zwei Aufgaben:

1. **Welche** Aktion ist die beste? (Auswahl)
2. **Wie gut** ist diese Aktion? (Bewertung)

$$y_DQN = r + γ · max_a' Q(s', a'; θ⁻)   ← Auswahl UND Bewertung durch θ⁻$$

Wenn das Netz eine Aktion zufällig überschätzt, wählt `max` genau diese Aktion — und bewertet sie ebenfalls hoch. Zwei Fehler verstärken sich gegenseitig.

**Lösung (Double DQN):** Trenne die beiden Aufgaben:

$$y_DDQN = r + γ · Q(s', argmax_a' Q(s',a';θ); θ⁻)$$
$$                          └── Auswahl durch θ ──┘  └─ Bewertung durch θ⁻ ─┘$$

### Visualisierungsidee

```
Standard DQN:
Q-Werte:  [2.1, 2.0, 8.7 ← verrauscht, 2.3]
           └──── max wählt 8.7 aus UND bewertet mit 8.7 ────┘
           → Target zu hoch!

Double DQN:
Online-Netz wählt: max([2.1, 2.0, 8.7, 2.3]) → Aktion 3
Target-Netz bewertet Aktion 3: Q(s', a₃; θ⁻) = 2.5
           → Realistischerer Target!
```

### Implementierung (minimale Änderung!)

```python
# Standard DQN Target:
target = r + gamma * target_net(s_).max()

# Double DQN Target:
best_action = online_net(s_).argmax()           # Online-Netz wählt
target = r + gamma * target_net(s_)[best_action]  # Target-Netz bewertet
```

### Häufige Missverständnisse

❌ **„Double DQN trainiert zwei völlig separate Netze."**  
✅ Es nutzt das bereits vorhandene Online-Netz und Target-Netz mit vertauschten Rollen.

❌ **„Überschätzung ist ein kleines Problem."**  
✅ Systematische Überschätzung akkumuliert sich über Millionen von Updates und kann zum Kollaps führen.

### Mini-Quiz UE 6–7

1. In welcher Situation führt Standard-DQN zu einer Überschätzung?  
   *(Antwort: Wenn eine Aktion zufällig einen hohen Q-Wert erhält, wählt max diese — und das gleiche verrauschte Netz bewertet sie hoch)*

2. Warum ist Double DQN nur eine kleine Code-Änderung?  
   *(Antwort: Online-Netz und Target-Netz existieren bereits; nur die Rollen werden getauscht)*

3. Was bedeutet „Dekopplung von Auswahl und Bewertung"?  
   *(Antwort: Die beste Aktion wird vom einen Netz ausgewählt, ihr Wert aber vom anderen bewertet)*

---

## UE 8–9: Dueling DQN — V(s) und A(s,a) trennen

### Intuitive Erklärung

Q(s,a) = V(s) + A(s,a)

- **V(s)** — Value Function: „Wie gut ist dieser Zustand?" (aktionsunabhängig)
- **A(s,a)** — Advantage Function: „Wie viel besser/schlechter ist Aktion a relativ zum Durchschnitt?"

### Architektur

```
Eingabe: s
    ↓
Gemeinsamer Feature-Extraktor (Conv-Layer)
    ↙                    ↘
Value-Stream           Advantage-Stream
[FC → V(s)]            [FC → A(s,a₁), A(s,a₂), ..., A(s,aₙ)]
    ↘                    ↙
         Aggregation:
   Q(s,a) = V(s) + A(s,a) - mean_a'[A(s,a')]
         ↓
   Q-Werte für alle Aktionen
```

**Warum `- mean`?** Zur Identifizierbarkeit: Ohne Normalisierung könnten V(s) und A(s,a) beliebig verschoben sein.

### Visualisierungsidee

```
Atari: Breakout

Zustand: Ball fliegt auf Paddel zu
→ V(s) hoch (gute Situation)
→ A(links): +0.3  (Ball-Aufprallpunkt leicht links)
→ A(rechts): -0.1
→ A(nix): -0.2

Zustand: Ball weit weg, viele Blöcke übrig
→ V(s) mittel
→ A(links) ≈ A(rechts) ≈ A(nix) ≈ 0  (kurzfristig egal)
```

### Häufige Missverständnisse

❌ **„Dueling DQN ist eine andere Lernregel."**  
✅ Gleiches Bellman-Update, nur die Netzarchitektur ändert sich.

❌ **„V(s) wird explizit aus den Daten berechnet."**  
✅ V(s) und A(s,a) sind **gelernte** interne Repräsentationen, nicht direkt beobachtbar.

### Mini-Quiz UE 8–9

1. Was repräsentieren V(s) und A(s,a)?  
   *(Antwort: V(s) = genereller Wert des Zustands; A(s,a) = relative Güte einer Aktion im Vergleich zum Durchschnitt)*

2. Warum wird der Mittelwert der Advantage-Werte abgezogen?  
   *(Antwort: Zur Identifizierbarkeit der Zerlegung — ohne Normalisierung ist V und A nicht eindeutig trennbar)*

3. In welchem Zustand ist der Advantage-Stream besonders wichtig?  
   *(Antwort: Wenn es kritische Aktionsunterschiede gibt, z. B. direkt vor einer Falle)*

---

## UE 10–11: Prioritized Experience Replay (PER)

### Intuitive Erklärung

Standard Replay Buffer: **Uniform Sampling** — jede Erfahrung wird gleich häufig gezogen.

**Problem:** Viele Erfahrungen sind „langweilig" (kleiner TD-Fehler). Wenige sind „überraschend" (großer TD-Fehler) und enthalten viel Lernpotenzial.

**Analogie:** Ein Schüler übt für eine Prüfung:

- A) Je 10 Min. pro Kapitel (uniform)
- B) Mehr Zeit bei Kapiteln, die er noch nicht versteht (prioritized) → effizienter

**PER-Idee:** Priorisiere Erfahrungen mit **großem TD-Fehler**:

$$p_i = |δ_i| + ε
P(i) = p_i^α / Σ_k p_k^α$$

- δ_i: TD-Fehler der Erfahrung i
- ε: kleine Konstante gegen Division durch 0
- α: Stärke der Priorisierung (0 = uniform, 1 = vollständig priorisiert)

### Importance Sampling Korrektur

Priorisiertes Sampling verändert die Datenverteilung → verzerrter Gradient.

**Fix:** IS-Gewichte korrigieren den Bias:

$$w_i = (1 / (N · P(i)))^β$$

β wird während des Trainings von 0 auf 1 angehoben (Annealing).

### Visualisierungsidee

```
Replay Buffer (Größe 10):
[δ=0.1, δ=5.2, δ=0.0, δ=3.1, δ=0.2, δ=8.4, δ=0.1, δ=0.3, δ=4.7, δ=0.0]

Uniform:  jede Erfahrung mit 10% Wahrscheinlichkeit
PER:      Erfahrung mit δ=8.4 wird ~40-mal öfter gezogen als δ=0.0
```

### Implementierung: SumTree

Effizientes PER verwendet eine **SumTree**-Datenstruktur für O(log n) Einfügen und Samplen:

```
       22 (Summe)
      /    \
    14      8
   / \     / \
  5   9   3   5
```

### Häufige Missverständnisse

❌ **„PER garantiert, dass wichtige Erfahrungen immer verwendet werden."**  
✅ PER erhöht die Wahrscheinlichkeit — auch Erfahrungen mit niedrigem δ werden noch gezogen.

❌ **„PER braucht keine IS-Korrektur."**  
✅ Ohne IS-Korrektur konvergiert das Training zu einem verzerrten Policy.

### Mini-Quiz UE 10–11

1. Was ist der TD-Fehler, und warum eignet er sich als Prioritätskriterium?  
   *(Antwort: TD-Fehler = Differenz zwischen Schätzung und verbessertem Ziel; groß = noch viel zu lernen)*

2. Warum werden IS-Gewichte in PER benötigt?  
   *(Antwort: Priorisiertes Sampling verändert die Datenverteilung; IS-Gewichte korrigieren diesen Bias)*

3. Was ist der Vorteil der SumTree-Datenstruktur?  
   *(Antwort: O(log n) statt O(n) für Einfügen und Samplen — skaliert auf große Buffer)*

---

## UE 12: Noisy Nets — Exploration durch Stochastische Gewichte

### Intuitive Erklärung

**Standard ε-Greedy:**

```
mit Wahrscheinlichkeit ε:   zufällige Aktion
mit Wahrscheinlichkeit 1-ε: greedy Aktion
```

**Probleme mit ε-Greedy:**

- ε ist ein globaler Parameter — das Netz weiß nicht, wo sein Wissen lückenhaft ist
- In bekannten Bereichen zu viel Zufall, in unbekannten zu wenig

**Noisy Nets-Idee (Fortunato et al., 2017):** Füge **lernbare Rauschterme** zu den Gewichten hinzu:

$$y = (μ^w + σ^w * ε^w) x + (μ^b + σ^b * ε^b)$$

- μ: gelernte mittlere Gewichte (wie normal)
- σ: gelernte Rausch-Skalierung (wie viel Exploration?)
- ε: zufällig gezogenes Rauschen (Gauß-verteilt)

**Entscheidend:** Das Netz lernt selbst, wie viel Rauschen es braucht:

- Bekannte Situation → σ → 0 → deterministische Entscheidung
- Unbekannte Situation → σ bleibt groß → explorative Entscheidung

### Visualisierungsidee

```
ε-Greedy:                      Noisy Net:

Exploration                    Exploration
   ▲                              ▲
   │ ε                            │ σ
   │──────────────────────        │   ╲
   │                              │    ╲
   └──────────────────── Zeit     └─────╲────── Zeit
        (manuell abgesenkt)           (gelernt — sinkt wo nötig)
```

### Häufige Missverständnisse

❌ **„ε-Greedy und Noisy Nets werden kombiniert."**  
✅ Noisy Nets *ersetzen* ε-Greedy. Mit Noisy Nets wird ε auf 0 gesetzt.

❌ **„Das Netz wird instabiler durch Rauschen."**  
✅ Da σ gelernt wird, kann das Netz Rauschen reduzieren, wo es schadet.

❌ **„Rauschen wird nur im Training genutzt."**  
✅ Noisy Net-Rauschen wird auch beim Handeln (Inference) gezogen.

### Mini-Quiz UE 12

1. Was ist der zentrale Unterschied zwischen ε-Greedy und Noisy Nets?  
   *(Antwort: ε ist ein globaler, manueller Parameter; σ in Noisy Nets wird pro Gewicht gelernt)*

2. Wie lernt das Netz, in bekannten Zuständen weniger zu explorieren?  
   *(Antwort: σ wird durch Backpropagation gelernt — niedriges σ = confident)*

3. Warum ist Factorised Gaussian effizienter als Independent Gaussian?  
   *(Antwort: Rauschparameter skalieren mit p+q statt p·q)*

---

## UE 13: Rainbow — Alle 6 Komponenten kombiniert

### Überblick

| # | Komponente | Löst |
|---|-----------|------|
| 1 | **Double DQN** | Q-Überschätzung |
| 2 | **Dueling Networks** | Ineffiziente V/A-Trennung |
| 3 | **Prioritized Replay** | Ineffizientes Sampling |
| 4 | **Noisy Nets** | Schlechte Exploration |
| 5 | **Multi-step Returns** | Langsame Reward-Propagation |
| 6 | **Distributional RL (C51)** | Nur Erwartungswert statt Verteilung |

### Multi-step Returns

Standard DQN (1-step):

$$G_t(1) = r_t + γ · max_a' Q(s_{t+1}, a')$$

Multi-step (n Schritte):

$$G_t(n) = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γ^(n-1)·r_{t+n-1} + γ^n · max_a' Q(s_{t+n}, a')$$

**Vorteil:** Belohnungen propagieren schneller zurück — besonders bei sparse rewards.

### Distributional RL (C51)

Standard DQN schätzt **einen Wert** Q(s,a) = E[G].

C51 schätzt die **vollständige Verteilung** der Returns:

```
Standard Q:      [2.5]          ← Erwartungswert

C51:       ▐▐▐▐▐▐             ← Verteilung über mögliche Returns
       -10  0  +10  +20
```

### Rainbow-Performance auf Atari

```
Median Human-Normalized Score (57 Atari Spiele):

DQN (2015):   ████████████ ~80%
Double DQN:   ██████████████ ~100%
PER:          ██████████████████ ~125%
Rainbow:      ████████████████████████████ ~230%

→ Rainbow übersteigt menschliche Leistung in 40 von 57 Spielen
  (nach nur 7% der Trainingszeit von ursprünglichem DQN)
```

### Ablationsstudie

```
Rainbow ohne PER:           großer Einbruch
Rainbow ohne Multi-step:    großer Einbruch
Rainbow ohne Distributional: mittel
Rainbow ohne Noisy Nets:    klein
Rainbow ohne Dueling:       klein
Rainbow ohne Double DQN:    klein

→ PER und Multi-step sind die wichtigsten Einzelkomponenten
```

### Häufige Missverständnisse

❌ **„Rainbow ist immer die beste Wahl."**  
✅ Rainbow ist schwerer zu implementieren und zu tunen. Für einfache Aufgaben reicht DQN + Double + PER.

❌ **„Die 6 Komponenten sind unabhängig voneinander."**  
✅ Sie interagieren — z. B. Multi-step Returns verändern die TD-Fehler, die PER nutzt.

❌ **„Rainbow ist das neueste/beste RL-Verfahren."**  
✅ Rainbow (2018) wurde seitdem von Agent57, MuZero u. a. übertroffen. Es ist aber ein wichtiger Meilenstein.

### Mini-Quiz UE 13

1. Nenne die 6 Komponenten von Rainbow.  
   *(Antwort: Double DQN, Dueling Networks, PER, Noisy Nets, Multi-step Returns, Distributional RL)*

2. Was ist der Unterschied zwischen Q(s,a) in DQN und in C51?  
   *(Antwort: DQN schätzt einen Erwartungswert; C51 schätzt die vollständige Wahrscheinlichkeitsverteilung der Returns)*

3. Welche zwei Komponenten tragen laut Ablationsstudie am meisten bei?  
   *(Antwort: Prioritized Experience Replay und Multi-step Returns)*

---

## UE 14: Explorations-Strategien im Vergleich

### Überblick aller Strategien

| Strategie | Typ | Wann sinnvoll | Wann problematisch |
|-----------|-----|--------------|-------------------|
| **ε-Greedy** | Zufällig | Einfache Umgebungen, Baseline | Skaliert schlecht, blind |
| **ε-Greedy mit Decay** | Adaptiv | Standard für DQN | ε-Schedule manuell wählen |
| **Boltzmann/Softmax** | Probabilistisch | Q-Werte gut kalibriert | Temperatur-Hyperparameter |
| **UCB** | Optimismus | Multi-Armed Bandit | Skaliert schlecht |
| **Noisy Nets** | Parametrisch | DQN-Varianten, Rainbow | Komplexere Implementierung |
| **ICM (Curiosity)** | Modellbasiert | Sparse Rewards, große Räume | Noisy TV Problem |
| **Count-Based** | Statistisch | Diskrete Räume | Skaliert nicht |

### ε-Greedy mit Decay

```python
epsilon = epsilon_start  # z.B. 1.0

for episode in training:
    if random() < epsilon:
        action = random_action()
    else:
        action = argmax(Q(state))

    epsilon = max(epsilon_min, epsilon * decay_rate)
    # z.B. epsilon_min=0.01, decay=0.995
```

### Boltzmann-Exploration

$$π(a|s) = e^(Q(s,a)/τ) / Σ_a' e^(Q(s,a')/τ)$$

- τ hoch: fast uniform (viel Exploration)
- τ niedrig: fast greedy

### Intrinsic Curiosity Module (ICM)

$$r_gesamt = r_extrinsic + η · r_intrinsic$$
$$r_intrinsic = ‖ ŝ_{t+1} - s_{t+1} ‖²$$

Das ICM-Netz versucht, den nächsten Zustand vorherzusagen. Großer Vorhersagefehler = unbekannte Situation = hohe Curiosity.

```
Agent sieht zum ersten Mal eine Tür:
→ ICM: "Ich kann nicht vorhersagen, was dahinter ist!"
→ Hoher Vorhersagefehler → Hohe Curiosity-Belohnung → Agent öffnet Tür

Nächstes Mal:
→ ICM hat gelernt, was hinter der Tür ist
→ Kleiner Vorhersagefehler → Agent exploriert woanders
```

**Noisy TV Problem:** Ein TV mit zufälligem Rauschen ist immer unvorhersagbar → unendliche Curiosity. Lösung: RND (Random Network Distillation).

### Situationsgerechte Wahl

```
Kleiner, bekannter State-Space?
    └─ JA  → UCB oder Count-Based
    └─ NEIN → Kontinuierlich / hochdimensional?
                └─ JA → DQN-Variante?
                            └─ JA  → Noisy Nets
                            └─ NEIN → ε-Greedy with Decay
                └─ NEIN → Sparse Rewards?
                            └─ JA  → ICM / RND
                            └─ NEIN → ε-Greedy with Decay
```

| Projekt-Typ | Empfehlung |
|------------|-----------|
| Spieleprototyp / Lernen | ε-Greedy mit Decay |
| Produktions-DQN | Noisy Nets |
| Robotik, Sparse Rewards | ICM oder RND |
| Multi-Armed Bandit | UCB |
| Viele kalibrierte Q-Werte | Boltzmann |

### Häufige Missverständnisse

❌ **„Mehr Exploration ist immer besser."**  
✅ Zu viel Exploration verhindert Exploitation des Gelernten — Balance ist entscheidend.

❌ **„Curiosity löst alle Explorationsprobleme."**  
✅ Curiosity leidet am Noisy TV Problem und ist rechenintensiv.

❌ **„ε-Greedy ist veraltet."**  
✅ Für viele Aufgaben ist es robust und ausreichend.

### Mini-Quiz UE 14

1. Was ist der Hauptnachteil von ε-Greedy im Vergleich zu Noisy Nets?  
   *(Antwort: ε ist global — das Netz lernt nicht selbst, wann/wo Exploration nötig ist)*

2. Was ist das Noisy TV Problem?  
   *(Antwort: Zufällige, unvorhersagbare Ereignisse erzeugen permanente Curiosity-Belohnung ohne echten Lernerfolg)*

3. Für welchen Anwendungsfall ist UCB geeignet, und warum nicht für Atari?  
   *(Antwort: Multi-Armed Bandit / kleiner diskreter Zustandsraum; in hochdimensionalen Räumen ist Zählen nicht praktikabel)*

4. Was ist der Unterschied zwischen extrinsischer und intrinsischer Belohnung?  
   *(Antwort: Extrinsisch = aus der Umgebung; Intrinsisch = intern erzeugt durch das Modell)*

---

## Gesamtübersicht: DQN-Evolutionspfad

```
Q-Learning (tabellarisch)
    ↓ + neuronales Netz + Experience Replay + Target Network
DQN (2015)
    ↓ + Dekopplung von Auswahl und Bewertung
Double DQN (2015)
    ↓ + V(s) / A(s,a) Architektur
Dueling DQN (2016)
    ↓ + TD-fehlerbasiertes Sampling
Prioritized Experience Replay (2015)
    ↓ + lernbare Rauschgewichte
Noisy Nets (2017)
    ↓ + Multi-step Returns + Distributional RL (C51)
Rainbow (2018) ← SOTA auf Atari zum Erscheinungszeitpunkt
    ↓
Agent57, MuZero, ... (neuere Entwicklungen)
```

---

## Abschluss-Quiz (alle 14 UE)

1. Warum reicht eine Q-Tabelle nicht für Atari?
2. Was sind die zwei zentralen Tricks des Basis-DQN?
3. Wozu dient das Target Network?
4. Wie unterscheidet sich die Target-Berechnung von Double DQN gegenüber Standard DQN?
5. Was repräsentieren V(s) und A(s,a) in Dueling DQN?
6. Welche Prioritätsmetrik verwendet PER?
7. Warum braucht PER Importance Sampling Gewichte?
8. Was lernt ein Noisy Net zusätzlich zu normalen Gewichten?
9. Nenne alle 6 Rainbow-Komponenten.
10. Was ist der Unterschied zwischen DQN und C51 in ihrer Ausgabe?
11. Beschreibe das Noisy TV Problem.
12. Für welche Aufgabe empfiehlst du UCB und warum nicht für Atari?

---

*Quellen: Mnih et al. (2015), van Hasselt et al. (2015), Wang et al. (2016), Schaul et al. (2015), Fortunato et al. (2017), Bellemare et al. (2017), Hessel et al. (2018)*

```

Den Inhalt kannst du mit **Ctrl+A → Ctrl+C** kopieren und in eine neue Datei, z. B. `rl3-Deep_Q_Learning_Kurs.md`, einfügen und speichern. Alternativ kannst du im Editor-Modus (statt Ask-Modus) arbeiten, damit ich die Datei direkt anlegen kann.Den Inhalt kannst du mit **Ctrl+A → Ctrl+C** kopieren und in eine neue Datei, z. B. `rl3-Deep_Q_Learning_Kurs.md`, einfügen und speichern. Alternativ kannst du im Editor-Modus (statt Ask-Modus) arbeiten, damit ich die Datei direkt anlegen kann.
