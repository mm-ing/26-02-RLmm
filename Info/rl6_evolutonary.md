# Alternatives Learning: Evolutionäre Methoden

## Inhaltsübersicht

| UE   | Thema                                          |
|------|---------------------------------------------------------|
| 1–2  | Genetische Algorithmen (GA) — Evolution als Optimierung |
| 3–4  | Evolution Strategies (ES) — Gradienten ohne Backprop    |
| 5–6  | Neuroevolution — Neuronale Netze evolvieren             |
| 7–8  | Genetische Programmierung (GP) — Code evolvieren        |
| 9–10 | Evolutionary Learning im Vergleich & Ausblick           |

---

## UE 1–2: Genetische Algorithmen — Evolution als Optimierung

### Intuitive Erklärung

Die Natur optimiert seit Milliarden von Jahren — ohne Gradientenabstieg, ohne Backpropagation. Durch **Selektion, Kreuzung und Mutation** entstehen immer besser angepasste Organismen.

Genetische Algorithmen (GA) übertragen dieses Prinzip auf Optimierungsprobleme:

```

Population (Lösungskandidaten)
      ↓ Bewertung (Fitness-Funktion)
Selektion der Besten
      ↓ Kreuzung (Crossover)
Neue Generation
      ↓ Mutation
Population (verbessert)
      ↓ wiederholen...

```

> **Kernidee:** Eine Population von Lösungen entwickelt sich über Generationen. Bessere Lösungen überleben und vererben ihre Eigenschaften.

### Die vier Grundoperationen

#### 1. Repräsentation (Kodierung)

Jede Lösung wird als **Chromosom** kodiert — meistens als Bitstring, Array oder Vektor:

```

Problem: Optimiere 8 Parameter (je 0–255)
Chromosom: [42, 187, 93, 255, 0, 128, 64, 201]
           [p₁,  p₂,  p₃,  p₄, p₅,  p₆, p₇,  p₈]

```

#### 2. Fitness-Funktion

Bewertet, wie "gut" ein Chromosom ist:

```python
def fitness(chromosome):
    # Beispiel: Policy-Reward in einer RL-Umgebung
    total_reward = evaluate_in_environment(chromosome)
    return total_reward
```

#### 3. Selektion

Wähle Eltern für die nächste Generation — bessere Lösungen sollen häufiger ausgewählt werden:

```
Roulette-Selektion (fitness-proportional):
  Individuum A: fitness=10 → 50% Wahrscheinlichkeit
  Individuum B: fitness=6  → 30%
  Individuum C: fitness=4  → 20%

Turnier-Selektion:
  Wähle k zufällige Individuen → bestes gewinnt
  (robuster gegen extreme Fitness-Unterschiede)
```

#### 4. Kreuzung (Crossover) und Mutation

```
Eltern:
  A: [42, 187, 93, 255 | 0, 128, 64, 201]
  B: [10,  55, 200,  88 | 33,  77, 190,  5]
                    ↑ Kreuzungspunkt

Kind 1: [42, 187, 93, 255, 33,  77, 190,  5]
Kind 2: [10,  55, 200,  88,  0, 128,  64, 201]

Mutation (kleine zufällige Änderung):
  Kind 1: [42, 187, 93, 255, 33, 77, 191, 5]
                                        ↑ zufällig verändert
```

### GA-Loop in Pseudocode

```python
# Initialisierung
population = [random_chromosome() for _ in range(POP_SIZE)]

for generation in range(MAX_GENERATIONS):
    # Bewertung
    fitnesses = [fitness(c) for c in population]
    
    # Neue Generation
    new_population = []
    while len(new_population) < POP_SIZE:
        parent_a = tournament_select(population, fitnesses)
        parent_b = tournament_select(population, fitnesses)
        child = crossover(parent_a, parent_b)
        child = mutate(child, mutation_rate=0.01)
        new_population.append(child)
    
    population = new_population

best = population[argmax(fitnesses)]
```

### Visualisierungsidee

```
Generation 0:        Generation 5:        Generation 20:
● ● ● ● ●            ● ● ●                  ★
● ● ● ● ●            ● ●   ★               ★★★
● ●   ● ●              ★★                ★★★★★
(zufällig verteilt)  (Clustering)        (konvergiert zum Optimum)

Fitness-Landschaft:
    ▲ Fitness
    │         ★
    │      ★★★★★★
    │   ★★★★★★★★★★★
    │ ★★★★★★★★★★★★★★★
    └─────────────────────→ Parameter
      GA sucht Gipfel ohne Gradient
```

### Vergleich mit Gradientenabstieg

| Eigenschaft | Gradientenabstieg | Genetischer Algorithmus |
|-------------|------------------|------------------------|
| Ableitung nötig? | Ja | Nein |
| Mehrere Optima | bleibt oft im lokalen Optimum | exploriert parallel viele Bereiche |
| Kontinuierliche Räume | sehr effizient | möglich, aber weniger effizient |
| Diskrete/kombinatorische Räume | schwierig | natürlich geeignet |
| Parallelisierbar | begrenzt | sehr gut (jedes Individuum unabhängig) |

### Beispiele

- **Stundenplanung:** Schulstundenplan als Chromosom — Fitness = minimale Kollisionen
- **Antennenentwurf:** NASA nutzte GA für Satellitenantennenentwurf (2006) — nicht-intuitives, hochoptimales Design
- **Spielstrategie:** Evolviere Parameter einer Regelstrategie für Atari-Spiele
- **Handelsstrategien:** Optimiere Parameter technischer Indikatoren

### Häufige Missverständnisse

❌ **„GA konvergiert immer zur globalen Lösung."**  
✅ GA kann in lokalen Optima stecken bleiben. Diversität (Mutation, Populations-Größe) ist entscheidend.

❌ **„GA ist langsam und veraltet."**  
✅ Mit Parallelisierung (GPU-Cluster, verteilte Systeme) skalieren GA sehr gut — OpenAI nutzte ES/GA auf tausenden CPU-Kernen.

❌ **„Je mehr Mutation, desto besser."**  
✅ Zu hohe Mutation = zufällige Suche ohne Konvergenz. Balance zwischen Exploration (Mutation) und Exploitation (Selektion) ist entscheidend.

### Mini-Quiz UE 1–2

1. Nenne die vier Grundoperationen eines Genetischen Algorithmus.  
   *(Antwort: Repräsentation/Kodierung, Fitness-Bewertung, Selektion, Kreuzung + Mutation)*

2. Was ist der Vorteil von Turnier-Selektion gegenüber Roulette-Selektion?  
   *(Antwort: Robuster bei extremen Fitness-Unterschieden; ein einziges Individuum mit riesiger Fitness dominiert nicht die ganze Selektion)*

3. Wann eignet sich GA besonders im Vergleich zu Gradientenabstieg?  
   *(Antwort: Bei diskreten oder kombinatorischen Suchräumen, nicht-differenzierbaren Fitness-Funktionen, multimodalen Landschaften)*

4. Was passiert bei zu niedriger Mutationsrate?  
   *(Antwort: Vorzeitige Konvergenz — die Population verliert Diversität und bleibt in einem suboptimalen Bereich)*

---

## UE 3–4: Evolution Strategies (ES) — Gradienten ohne Backprop

### Intuitive Erklärung

Evolution Strategies (ES) sind eine engere Verwandte der GA, speziell für **kontinuierliche Parametervektoren** (z. B. neuronale Netzgewichte) entwickelt.

**Grundidee (1+1)-ES:**

```
Starte mit einem Individuum θ
Wiederhole:
  θ' = θ + ε  (ε ~ N(0, σ²))  ← zufällige Perturbation
  Wenn fitness(θ') > fitness(θ):  θ = θ'
```

Das klingt simpel — aber moderne ES (insbesondere **OpenAI-ES**) skaliert auf Millionen von Parametern.

### OpenAI-ES (Salimans et al., 2017)

**Idee:** Statt eines Individuums — eine **Population von Perturbationen** um den aktuellen Parameter-Vektor:

```
θ (aktueller Vektor)
  ↓
Erstelle N Perturbationen: εᵢ ~ N(0, I), i = 1..N
  ↓
Evaluiere: F(θ + σεᵢ) für alle i  (vollständig parallelisierbar!)
  ↓
Update:
  θ ← θ + α/(Nσ) · Σᵢ Fᵢ · εᵢ
```

**Das ist ein Gradienten-Schätzung!** Der Update entspricht näherungsweise einem Gradienten-Schritt von E[F(θ + σε)] bezüglich θ.

```
Intuitiv: "Perturbiere θ in alle Richtungen εᵢ.
           Wenn eine Richtung zu höherem Reward führt,
           bewege θ in diese Richtung — gewichtet mit dem Reward."
```

### ES vs. RL (Policy Gradient)

| Eigenschaft | Policy Gradient (z. B. PPO) | OpenAI ES |
|------------|----------------------------|-----------|
| Gradientenberechnung | Backpropagation durch Netz | Nur Forward-Pass |
| Parallelisierbarkeit | Begrenzt (sequentielle Episoden) | Perfekt (N unabhängige Evaluierungen) |
| Speicherbedarf | Hoch (Aktivierungen für Backprop) | Niedrig |
| Long-horizon Credit Assignment | Schwierig | Implizit (Episode-Return direkt) |
| Sensoren/Black-Box-Systeme | Neta muss differenzierbar sein | Vollständige Black Box |
| 1000 CPUs: Training Ant | ~10 Stunden | ~10 Minuten |

### CMA-ES: Covariance Matrix Adaptation

Die mächtigste ES-Variante für hochdimensionale kontinuierliche Optimierung:

```
Statt σ·I (kugelförmige Perturbationen):
  CMA-ES lernt die Kovarianzmatrix C der Suchverteilung

Intuition:
  Anfang:   ●●●●●   (kugelförmig)
              ●●●
  
  Angepasst: ●●●
            ●●●●●   (ellipsoid entlang der Fortschrittsrichtung)
              ●●●
              
→ Suchverteilung passt sich der Fitness-Landschaft an
```

CMA-ES gilt als **State-of-the-Art für Black-Box-Optimierung** mit wenigen tausend Parametern.

### Visualisierungsidee

```
OpenAI-ES auf 1000 CPUs:

CPU₁:  θ + σε₁  →  evaluate → F₁
CPU₂:  θ + σε₂  →  evaluate → F₂
...
CPU₁₀₀₀: θ + σε₁₀₀₀ → evaluate → F₁₀₀₀
                ↓ (alle parallel, synchron)
Update: θ ← θ + α/(Nσ) · Σ Fᵢεᵢ
                ↓
        nächste Generation (Minuten statt Stunden)

→ Wie: 1000 Roboter probieren gleichzeitig kleine Variationen aus
       und berichten zurück, was funktioniert hat
```

### Praktisches Beispiel: MuJoCo Locomotion

OpenAI-ES trainierte in 10 Minuten auf 1440 CPUs einen Laufalgorithmus für den MuJoCo-Ant-Roboter — vergleichbare Performance zu A3C, das Stunden auf einer GPU benötigt.

```
Warum geht das so schnell?
  - Keine Backpropagation → nur Forward-Passes
  - Jede CPU evaluiert eine Episode unabhängig
  - Synchonisation: nur ein Vektor θ + N Skalare Fᵢ übertragen
  - Kein geteilter Speicher zwischen CPUs nötig
```

### Häufige Missverständnisse

❌ **„ES ist primitiver als RL, weil es keinen Gradienten nutzt."**  
✅ ES *schätzt* einen Gradienten durch Sampling — bei Black-Box-Systemen oder sehr langen Episoden kann das überlegen sein.

❌ **„ES kann nur einfache Probleme lösen."**  
✅ OpenAI-ES überstand MuJoCo-Benchmarks, die zu den schwierigsten kontinuierlichen Steuerungsaufgaben gehören.

❌ **„CMA-ES skaliert auf beliebig viele Parameter."**  
✅ CMA-ES skaliert auf ~1000 Parameter gut. Für Millionen Parameter ist OpenAI-ES oder NES besser geeignet.

### Mini-Quiz UE 3–4

1. Was ist das Grundprinzip von OpenAI-ES in einem Satz?  
   *(Antwort: Perturbiere den Parameter-Vektor in N Richtungen, evaluiere parallel und bewege den Vektor in Richtung höherer Rewards)*

2. Warum ist ES auf vielen CPUs oft schneller als Policy Gradient?  
   *(Antwort: Jede Evaluation braucht nur einen Forward-Pass ohne Backpropagation; alle N Evaluierungen sind vollständig unabhängig parallelisierbar)*

3. Was schätzt der ES-Update-Schritt näherungsweise?  
   *(Antwort: Den Gradienten des erwarteten Returns bezüglich der Parameter — ohne Backpropagation)*

4. Wann sollte man CMA-ES statt OpenAI-ES wählen?  
   *(Antwort: Bei kleinen bis mittleren Parameterräumen (~1000 Parameter), wenn keine massiv verteilte Infrastruktur verfügbar ist)*

---

## UE 5–6: Neuroevolution — Neuronale Netze evolvieren

### Intuitive Erklärung

**Neuroevolution** kombiniert neuronale Netze mit evolutionären Algorithmen: Statt Backpropagation werden **Netzgewichte** (und manchmal auch die **Netzstruktur selbst**) durch Evolution optimiert.

```
Population von neuronalen Netzen:
  Netz₁: [w₁=0.3, w₂=-1.2, w₃=0.7, ...]
  Netz₂: [w₁=0.5, w₂=-0.8, w₃=1.1, ...]
  ...
  Netz_N: [w₁=0.1, w₂=0.3, w₃=-0.5, ...]
       ↓
  Fitness = kumulierter Reward in Umgebung
       ↓
  Selektion, Kreuzung, Mutation
       ↓
  Nächste Generation (bessere Netze)
```

**Warum interessant?**

- Kein Backpropagation → funktioniert für nicht-differenzierbare Systeme
- Direkte Optimierung des Episode-Returns (kein Credit Assignment nötig)
- Kann diskrete Entscheidungen, Sprünge, nicht-stetige Systeme handhaben

### NEAT: NeuroEvolution of Augmenting Topologies

NEAT (Stanley & Miikkulainen, 2002) ist der bekannteste Neuroevolutions-Algorithmus — er evolviert gleichzeitig **Gewichte** und **Topologie** (Struktur) des Netzes.

**Das Problem bei Kreuzung von Netzen unterschiedlicher Topologie:**

```
Netz A (3 Knoten):  1→3, 2→3
Netz B (4 Knoten):  1→3, 1→4, 2→4, 3→4

Kreuzung → welche Verbindung von A entspricht welcher in B?
→ Das "competing conventions problem"
```

**NEAT's Lösung: Innovation Numbers**

Jede neue Verbindung und jeder neue Knoten bekommt eine globale **Innovations-Nummer**:

```
Gen 1:  1→3 (Nr.1), 2→3 (Nr.2)
Gen 2:  füge 1→4 hinzu (Nr.3), 2→4 (Nr.4)
Gen 3:  füge 3→4 hinzu (Nr.5)

Kreuzung: Verbindungen mit gleicher Nr. → von besserem Elternteil
          Verbindungen nur in einem Elternteil → zufällig übernehmen
```

**NEAT-Entwicklungspfad:**

```
Generation 0:       Generation 5:       Generation 20:
IN₁ → OUT          IN₁ → H₁ → OUT     IN₁ → H₁ → H₃
IN₂ → OUT          IN₂ → H₁            IN₂ → H₁ → H₄ → OUT
(minimstes Netz)   IN₁ → OUT           IN₁ → H₂ → H₄
                   IN₂ → H₂ → OUT      (komplexe Topologie)
                   (wächst bedarfsgerecht)
```

Netze starten minimal und **wachsen nur, wenn es hilft** — keine überdimensionierten Architekturen.

### HyperNEAT: Geometrische Regulierung

**Idee:** Statt einzelne Gewichte zu evolvieren — evolviere eine Funktion, die Gewichte erzeugt:

```
CPPN (Compositional Pattern Producing Network):
  Eingabe: (x₁, y₁, x₂, y₂)  ← Positionen zweier Neuronen
  Ausgabe: w                  ← Verbindungsgewicht zwischen ihnen

Vorteile:
  - Reguläre, geometrisch strukturierte Netze
  - Enormes Netz aus kleiner CPPN beschreibbar
  - Biologisch plausibler (Gehirn hat geometrische Struktur)
```

### Gewichts-Agnostische Neuronale Netze (WANN)

**Überraschende Erkenntnis (Gaier & Ha, 2019):** Manchmal ist die **Topologie** wichtiger als die **Gewichte**:

```
WANN-Suche:
  1. Suche nach Netz-Topologien, die bei einem EINZIGEN Gewichtwert w
     (gleich für alle Verbindungen!) eine Aufgabe lösen können.
  2. Teste w ∈ {-2, -1, -0.5, 0.5, 1, 2}

Ergebnis:
  Manche Topologien lösen CartPole oder BipedalWalker
  mit einem einzigen gemeinsamen Gewicht → Struktur trägt Information
```

### Visualisierungsidee

```
NEAT-Evolution in einem Labyrinth:

t=0:  Alle Netze zufällig → laufen gegen Wände
t=50: Einige lernen, der Wand auszuweichen
t=200: Netz mit H₁ (Wanddetektor) entsteht
t=500: Netz mit H₁, H₂, H₃ meistert das Labyrinth

Netz-Visualisierung (Größe wächst organisch):
●─────●          ●────●─────●
                      │
●─────●          ●────●─────●
               ←  neue Verbindung entstand durch Mutation
```

### Beispiele

| Anwendung | Methode | Ergebnis |
|-----------|---------|---------|
| Atari-Spiele | OpenAI-ES + Netzgewichte | Wettbewerbsfähig mit DQN |
| CartPole, MuJoCo | NEAT | Löst mit minimstem Netz |
| Robotik-Morphologie | HyperNEAT | Evolviert Körper + Controller zusammen |
| Spielstrategie Doom | Direct ES | Übertrifft A3C in manche Szenarien |
| Architektur-Suche | NEAT-basiert | Vorbild für Neural Architecture Search (NAS) |

### Häufige Missverständnisse

❌ **„Neuroevolution ist veraltet seit Deep Learning."**  
✅ Neuroevolution erlebt seit 2017 (OpenAI-ES, WANN) eine Renaissance — besonders bei Black-Box-Systemen und Robotik.

❌ **„NEAT findet immer die minimale optimale Topologie."**  
✅ NEAT tendiert zu Minimalität, aber Konvergenz zum globalen Optimum ist nicht garantiert.

❌ **„Neuroevolution braucht keine Hyperparameter."**  
✅ Mutations-Rate, Populations-Größe, Selektion und Speziation sind kritische Hyperparameter.

### Mini-Quiz UE 5–6

1. Was evolviert NEAT im Gegensatz zu einfacher Weight-Evolution?  
   *(Antwort: Gleichzeitig Gewichte und Netz-Topologie — Struktur und Parameter gemeinsam)*

2. Was ist das "competing conventions problem" und wie löst NEAT es?  
   *(Antwort: Verbindungen in verschiedenen Topologien sind nicht direkt vergleichbar; NEAT weist jeder neuen Verbindung eine globale Innovations-Nummer zu)*

3. Was ist die überraschende Erkenntnis von WANN?  
   *(Antwort: Manche Netz-Topologien lösen Aufgaben mit einem einzigen gemeinsamen Gewicht — die Struktur trägt entscheidende Information)*

4. Was beschreibt eine CPPN in HyperNEAT?  
   *(Antwort: Eine Funktion, die aus den geometrischen Positionen zweier Neuronen deren Verbindungsgewicht berechnet)*

---

## UE 7–8: Genetische Programmierung — Code evolvieren

### Intuitive Erklärung

Genetische Algorithmen evolvieren **Parameter** (Zahlen).  
Genetische Programmierung (GP) evolviert **Programme** (Code, Ausdrücke, Bäume).

```
GA:  [0.3, -1.2, 0.7, ...]  ← Zahlen-Array evolvieren
GP:  if sensor > 0.5:       ← Programm-Struktur evolvieren
       turn_left()
     else:
       move_forward()
```

**Programme als Bäume:**

```python
# Ausdruck: (x + y) * sin(x)

        *
       / \
      +   sin
     / \   \
    x   y   x
```

Jeder Knoten ist eine Funktion (Operator), jedes Blatt ist ein Terminal (Variable, Konstante). Der Baum kann durch Kreuzung und Mutation verändert werden.

### GP-Operationen

#### Baum-Kreuzung (Subtree Crossover)

```
Elternteil A:           Elternteil B:
        +                    *
       / \                  / \
      *   z               sin   y
     / \                   \
    x   y                   x

Kreuzungspunkte: Teilbaum von A = (*, x, y); aus B = (sin, x)

Kind:
        +
       / \
     sin   z
      \
       x
```

#### Mutation

```
Original:   if sensor > 0.5: left()   →   Zufaller Teilbaum ersetzt
Mutiert:    if sensor > 0.3: right()
            (Konstante 0.5 → 0.3, Funktion left → right)
```

### Symbolic Regression: GP als Wissenschaftler

**Aufgabe:** Finde eine mathematische Formel, die Datenpunkte beschreibt.

```
Daten:  [(1, 2), (2, 5), (3, 10), (4, 17)]
        ↑ klassifiziert durch Schüler als: y = x² + 1

GP-Suche:
  Kandidat 1: x + 2          → Fitness schlecht
  Kandidat 2: x² + x         → Fitness mittel
  Kandidat 3: x² + 1         → Fitness perfekt! ✓

→ GP entdeckt die Formel ohne Vorkenntnisse
```

**Reales Beispiel:** Eureqa (Schmidt & Lipson, 2009) nutzte GP, um physikalische Gesetze direkt aus Messdaten zu entdecken — darunter das Pendel-Gesetz.

### GP für Regelbasierte Agenten

```python
# Evolviertes GP-Programm für einen Labyrinth-Agenten:

if wall_front > 0.8:
    if wall_right < 0.3:
        turn_right()
    else:
        turn_left()
else:
    if goal_distance > 2.0:
        move_forward()
    else:
        turn_toward_goal()
```

**Vorteil gegenüber neuronalem Netz:**

- Vollständig interpretierbar
- Explizite Regeln können überprüft und manuell angepasst werden
- Kein Black-Box-Problem

### Bloat-Problem

**Wachstumsproblem in GP:**

```
Generation 1:  (x + 1) * 2        ← 5 Knoten, Fitness=0.8
Generation 50: ((x + 0 + 0) * 1 + (y*0)) * 2 + (z - z)
               ← 20 Knoten, Fitness=0.81 (kaum besser!)
```

Programme wachsen durch neutrale Mutationen (Code ohne Effekt) — **Bloat**.

**Lösungen:**

- Maximale Baumtiefe begrenzen
- Längen-Penalty in der Fitness: `fitness = reward - λ * tree_size`
- Parsimony Pressure (Occams Razor-Prinzip)

### Visualisierungsidee

```
GP-Evolution für symbolische Regression (y ≈ x²):

Gen 0:   zufällige Bäume:
         (x+y), sin(x), x*3, log(x)...

Gen 20:  vielversprechende Teilstrukturen:
         x*x, x²-1, x*x+0.3...

Gen 100: konvergiert:
         x*x+1.0  ← fast exakt

Baum-Visualisierung Gen 100:
       +
      / \
     *   1.0
    / \
   x   x
```

### GP vs. klassisches Programmieren vs. ML

| | Klassisch | ML (Netz) | GP |
|--|-----------|-----------|-----|
| **Lösung** | Handcode | Black-Box-Gewichte | Interpretierbarer Code |
| **Flexibel** | Nein | Ja | Ja |
| **Interpretierbar** | Ja | Nein | Ja |
| **Aufwand** | Manuell | Daten + Training | Fitness + Evolution |
| **Worst Case** | Bugs | Halluzinationen | Bloat |

### Häufige Missverständnisse

❌ **„GP erzeugt unlesbaren Code."**  
✅ GP kann sehr lesbaren Code erzeugen — mit Bloat-Kontrolle entstehen oft elegante, kompakte Ausdrücke.

❌ **„GP kann nur einfache Ausdrücke finden."**  
✅ Eureqa und moderne GP-Systeme entdeckten physikalische Gesetze, die nicht in Lehrbüchern standen.

❌ **„GP und GA sind dasselbe."**  
✅ GA evolviert feste Vektoren; GP evolviert variable baumförmige Strukturen — die Kreuzungsoperatoren sind grundlegend verschieden.

### Mini-Quiz UE 7–8

1. Was ist der fundamentale Unterschied zwischen GA und GP?  
   *(Antwort: GA evolviert Zahlenvektoren (Parameter); GP evolviert Programmstrukturen (Bäume aus Operatoren und Terminalen))*

2. Was ist Symbolic Regression?  
   *(Antwort: GP sucht eine mathematische Formel, die Datenpunkte beschreibt — keine vorgegebene Formelstruktur)*

3. Was ist "Bloat" in GP und wie kann er bekämpft werden?  
   *(Antwort: Programme wachsen durch neutrale Mutationen ohne Fitness-Verbesserung; Gegenmaßnahmen: Tiefenbegrenzung, Längen-Penalty)*

4. Wann würdest du GP einem neuronalen Netz vorziehen?  
   *(Antwort: Wenn Interpretierbarkeit wichtig ist, wenn die Lösung als explizite Regelstruktur benötigt wird, oder bei kleinen Datenmengen)*

---

## UE 9–10: Evolutionary Learning im Vergleich & Ausblick

### Überblick: Evolutionäre Algorithmen im RL-Kontext

```
Problemtyp → Empfohlene Methode:

Diskrete/kombinatorische Parameter?
    → Genetischer Algorithmus (GA)

Kontinuierliche Parameter, viele CPUs verfügbar?
    → OpenAI-ES / CMA-ES

Neuronale Netze, unbekannte Architektur?
    → NEAT / HyperNEAT

Code/Regeln als Lösung gewünscht?
    → Genetische Programmierung (GP)

Exploit bestehender Sprachmodell-Strukturen?
    → LLM-basierte evolutionäre Suche (aktuell)
```

### Vergleich aller Methoden

| Methode | Was wird evolviert? | Stärke | Schwäche |
|---------|-------------------|--------|---------|
| **GA** | Parameter-Vektoren | Diskrete Räume, einfach | Skaliert schlecht bei hoher Dimension |
| **ES** | Kontinuierliche Vektoren | Skaliert, parallelisierbar | Kein Struktur-Wachstum |
| **CMA-ES** | Vektoren + Kovarianz | Black-Box State-of-the-Art | Nur ~1000 Parameter |
| **NEAT** | Gewichte + Topologie | Wächst organisch | Komplex zu implementieren |
| **HyperNEAT** | Geometrische Gewichtsfunktion | Große, reguläre Netze | Abstraktionsschicht komplex |
| **GP** | Programmbäume | Interpretierbar, Symbolic Regression | Bloat, langsam |

### Evolutionary vs. Gradient-based RL

```
Wann Evolutionär?                   Wann Gradient-basiert (DRL)?

✅ Black-Box-Systeme                ✅ Differenzierbare Systeme
✅ Nicht-differenzierbare Rewards   ✅ Schnelles Lernen bei klarem Reward
✅ Sehr lange Episoden              ✅ Dichte Reward-Signale
✅ Massiv parallelisierbare Infra   ✅ Einzelne GPU ausreichend
✅ Interpretierbare Lösung (GP)     ✅ Komplexe wahrnehmungsbasierte Tasks
✅ Kombinatorische Suchräume        ✅ Standard RL-Benchmarks
```

### Quality Diversity (QD): Evolution + Diversität

Moderner Ansatz: Nicht nur das **beste** Individuum suchen, sondern eine **Bibliothek** unterschiedlicher guter Lösungen.

**MAP-Elites (Mouret & Clune, 2015):**

```
Fitness-Landschaft als 2D-Grid:
  Achse 1: Körperhöhe des Roboters (30–90 cm)
  Achse 2: Ganggeschwindigkeit (0.5–2.0 m/s)
  
  Jede Zelle: bestes Individuum mit dieser Eigenschaftskombination

┌──────────────────────────────┐
│ beste langsame, kleine Lösung │   ← Zelle (30cm, 0.5m/s)
│ beste langsame, große Lösung  │   ← Zelle (90cm, 0.5m/s)  
│ beste schnelle, kleine Lösung │   ← Zelle (30cm, 2.0m/s)
│ ...                           │
└──────────────────────────────┘
```

**Vorteil:** Bei Beinverlust kann der Roboter schnell auf eine andere Verhaltens-Nische zurückgreifen — Robustheit durch Diversität.

### EvoJAX und moderne Neuroevolution

**EvoJAX (Tang & Ha, 2022):** Hardware-beschleunigte ES auf TPU/GPU:

```
Traditionell: 1000 CPUs, langsame Kommunikation
EvoJAX:       Alle Individuen gleichzeitig auf einer TPU
              Kein Kommunikations-Overhead
              → 100× schneller als CPU-basierte ES
```

### LLMs als evolutionäre Operatoren (2024–2026)

**Neueste Entwicklung:** Nutze LLMs (GPT-4, Gemini) als intelligente Mutations- und Kreuzungsoperatoren:

```
Standard-Mutation: zufälliger Bit-Flip
LLM-Mutation:      "Hier ist das aktuelle Python-Programm.
                    Verändere es auf eine sinnvolle Weise,
                    die die Performance verbessern könnte."

→ FunSearch (DeepMind, 2024): LLM + GP → entdeckte neue mathematische
  Konstruktionen für das Cap-Set-Problem (über 50 Jahre ungelöst)
```

### Evolutionäres RL: Hybride Ansätze

**ERL (Evolutionary Reinforcement Learning):**

```
Population (GA):         Einzelner RL-Agent (z. B. TD3):
  Netz₁, Netz₂, ..., NetzN    π_RL (gradient-basiert)
       ↓ GA-Fitness             ↓ periodisch eingebettet
       ↓←──────────────────────↓
  Bester RL-Agent wird in Population injiziert
  Gute GA-Individuen helfen RL-Exploration
  
→ GA exploriert global, RL optimiert lokal — beste aus beiden Welten
```

### Visualisierungsidee: Gesamtüberblick

```
Evolutionäre Algorithmen — Stammbaum:

Biologische Evolution
         ↓
Genetische Algorithmen (Holland, 1975)
    ├── Genetische Programmierung (Koza, 1992)
    │       └── Symbolic Regression, Code-Evolution
    │
    └── Evolution Strategies (Rechenberg, 1973)
            ├── CMA-ES (Hansen, 2001)
            ├── OpenAI-ES (Salimans, 2017)
            └── Neuroevolution
                    ├── NEAT (Stanley, 2002)
                    ├── HyperNEAT (Stanley, 2007)
                    ├── WANN (Gaier, 2019)
                    └── EvoJAX (Tang, 2022)

Aktuelle Frontiers:
    ├── Quality Diversity (MAP-Elites, 2015)
    ├── ERL - Hybrid mit RL (2018)
    └── LLM + GP (FunSearch, 2024)
```

### Häufige Missverständnisse (Gesamtkapitel)

❌ **„Evolutionäre Methoden sind durch Deep Learning überholt."**  
✅ Sie lösen andere Probleme: Black-Box-Systeme, diskrete Räume, Interpretierbarkeit, Architektur-Suche.

❌ **„Evolution braucht immer eine sehr große Population."**  
✅ (1+1)-ES oder CMA-ES arbeiten mit sehr kleinen Populationen effektiv.

❌ **„Evolutionäres RL und Deep RL schließen sich aus."**  
✅ ERL und ähnliche hybride Methoden kombinieren beide — oft mit Synergieeffekten.

❌ **„GP kann echte Programme für Produktionssysteme erzeugen."**  
✅ GP-Code ist schwer zu testen, Debug zu betreiben und zu maintainen — meist als Prototyping-Tool oder für Formelfindung genutzt.

### Mini-Quiz UE 9–10

1. Wann würdest du ES gegenüber einem GA bevorzugen?  
   *(Antwort: Bei kontinuierlichen Parameterräumen (z. B. Netzgewichte), bei massiver Parallelisierung, wenn kein Gradient verfügbar ist)*

2. Was ist Quality Diversity und was unterscheidet es von Standard-Evolution?  
   *(Antwort: QD sucht eine Bibliothek unterschiedlicher, guter Lösungen statt nur das eine Optimum — Diversität als Ziel neben Fitness)*

3. Was hat FunSearch gezeigt?  
   *(Antwort: LLMs können als intelligente GP-Operatoren dienen und neue mathematische Entdeckungen machen, die jahrzehntelange offene Probleme lösen)*

4. Erkläre das ERL-Prinzip in zwei Sätzen.  
   *(Antwort: Eine GA-Population exploriert den Lösungsraum global; ein RL-Agent optimiert lokal durch Gradientenabstieg. Beide tauschen periodisch Individuen aus — globale und lokale Optimierung ergänzen sich)*

5. Nenne drei Szenarien, in denen evolutionäre Methoden Gradient-basiertem Deep RL überlegen sind.  
   *(Antwort: Black-Box-Systeme ohne differenzierbaren Reward; sehr lange Episoden ohne dichtes Reward-Signal; diskrete/kombinatorische Suchräume)*

---

## Gesamtüberblick: Wann welche Methode?

```
Aufgabe / Einschränkung                Empfehlung
──────────────────────────────────────────────────────────────
Differenzierbar, dichter Reward        → Deep RL (PPO, SAC, DQN)
Black-Box, keine Ableitung             → ES (OpenAI-ES, CMA-ES)
Diskret, kombinatorisch                → GA
Netz-Architektur unbekannt            → NEAT / NAS
Lösung soll Code/Regeln sein          → Genetische Programmierung
Formel aus Daten                       → Symbolic Regression (GP)
Robustheit durch Verhaltens-Diversität → MAP-Elites (QD)
Schnelle Parallelisierung auf TPU/GPU  → EvoJAX
Hybrid (Exploration + Exploitation)    → ERL
LLM-gestützte Code-Optimierung         → FunSearch-Paradigma
```

---

## Abschluss-Quiz (alle 10 UE)

1. Nenne die vier Grundoperationen eines Genetischen Algorithmus.
2. Was unterscheidet Turnier-Selektion von Roulette-Selektion?
3. Wie schätzt OpenAI-ES einen Gradienten ohne Backpropagation?
4. Was ist der Vorteil von ES bei Nutzung von 1000 CPUs gegenüber Policy Gradient?
5. Was macht NEAT anders als reines Gewichts-Evolution?
6. Was ist das "competing conventions problem" und NEATs Lösung?
7. Was evolviert Genetische Programmierung im Gegensatz zu GA?
8. Was ist Symbolic Regression und wie setzt GP sie um?
9. Was ist Bloat in GP und wie bekämpft man ihn?
10. Was ist Quality Diversity und was ist der Vorteil gegenüber Standard-Evolution?
11. Erkläre das Prinzip von ERL in einem Satz.
12. Wann würdest du einen evolutionären Ansatz einem Gradient-basierten RL-Verfahren vorziehen?
13. Was hat FunSearch (DeepMind, 2024) gezeigt?
14. Erkläre den Unterschied zwischen CMA-ES und OpenAI-ES.

---

*Quellen: Holland (1975) — Genetic Algorithms; Rechenberg (1973) — Evolution Strategies; Koza (1992) — Genetic Programming; Stanley & Miikkulainen (2002) — NEAT; Hansen (2001) — CMA-ES; Salimans et al. (2017) — OpenAI ES; Gaier & Ha (2019) — WANN; Mouret & Clune (2015) — MAP-Elites; Such et al. (2017) — Deep Neuroevolution; Tang & Ha (2022) — EvoJAX; Romera-Paredes et al. (2024) — FunSearch; Schmidt & Lipson (2009) — Eureqa*

```

Da ich im Ask-Modus bin, kann ich die Datei nicht direkt speichern. Kopiere den Inhalt oben und speichere ihn als neue `.md`-Datei — z. B. `rl5_evolutionary_kurs.md`. Im Editor-Modus kann ich das direkt für dich anlegen.Da ich im Ask-Modus bin, kann ich die Datei nicht direkt speichern. Kopiere den Inhalt oben und speichere ihn als neue `.md`-Datei — z. B. `rl5_evolutionary_kurs.md`. Im Editor-Modus kann ich das direkt für dich anlegen.
