# Policy Gradient – Vom Instinkt zur Strategie

### Reinforcement Learning, Lektion 4 · 14 × 45 Min

---

## Lernziele dieser Lektion

Nach dieser Einheit kannst du:

- erklären, warum Policy-Gradient-Methoden entstanden und wo Value-Based RL an Grenzen stößt
- den Policy-Gradient-Theorem mathematisch herleiten und intuitiv verstehen
- REINFORCE implementieren und seine Schwächen benennen
- Actor-Critic-Architekturen beschreiben und den Advantage-Begriff verwenden
- PPO als Industriestandard einordnen und seine Schlüsselidee erklären
- typische Fehler diagnostizieren und beheben

---

## Unterrichtsplan (Übersicht)

| Einheit | Thema | Schwerpunkt |
|---------|-------|-------------|
| 1–2 | Motivation & Intuition | Warum Policy Gradient? |
| 3–4 | Mathematische Grundlagen | Policy-Gradient-Theorem |
| 5–6 | REINFORCE | Implementierung & Varianz |
| 7–8 | Actor-Critic | Architektur & Advantage |
| 9–10 | Fortgeschrittene Varianten | PPO, TRPO, SAC |
| 11 | Stabilitätsmechanismen | Regularisierung & Tricks |
| 12 | Implementierungsdetails | Code-Praxis |
| 13 | Fehlerbilder & Debugging | Diagnose-Strategien |
| 14 | Praxisblock & Zusammenfassung | Übungen & Cheat-Sheet |

---

---

# Einheit 1–2 · Motivation & Intuition

## Warum Policy Gradient?

### Das Problem mit Value-Based RL

In den vorherigen Lektionen hast du Q-Learning und DQN kennengelernt: Der Agent lernt eine **Q-Funktion** Q(s, a), die den erwarteten Gesamtreturn für jedes Zustands-Aktions-Paar bewertet. Die Policy ist dann implizit:

```
π(s) = argmax_a Q(s, a)
```

Das funktioniert gut – aber hat strukturelle Schwächen:

| Problem | Ursache | Beispiel |
|---------|---------|---------|
| **Diskrete Aktionsräume erforderlich** | argmax über kontinuierliche Mengen ist nicht berechenbar | Roboterarm: Winkel ∈ [−180°, 180°] |
| **Instabilität** | Q-Funktion und Policy beeinflussen sich gegenseitig | DQN-Training divergiert ohne Tricks (Target Network, Replay Buffer) |
| **Suboptimale stochastische Policies** | Deterministische Greedy-Politik lässt sich nicht lernen | Poker: Man muss zufällig blenden, um nicht lesbar zu sein |
| **Keine Gradienten durch argmax** | argmax ist nicht differenzierbar | Kein End-to-End-Training |

### Die Policy-Gradient-Idee

Statt die Policy indirekt über eine Wertfunktion zu lernen, **optimieren wir sie direkt**:

```
Finde θ*, sodass J(θ) = E[G_t | π_θ] maximiert wird
```

Die Policy `π_θ(a|s)` ist ein parametrisiertes **neuronales Netz**, das eine Wahrscheinlichkeitsverteilung über Aktionen ausgibt. Wir optimieren θ mit Gradientenaufstieg.

---

### Intuition: Der Agent als Schauspieler

> **Analogie:** Stell dir einen Schauspieler vor, der ein Theaterstück probt.  
> Er wählt verschiedene Interpretationen (Aktionen) und beobachtet die Reaktion des Publikums (Reward).  
> Er verstärkt Interpretationen, die gut ankommen – und schwächt jene, die schlecht ankommen.  
> Er braucht kein Regelbuch (Modell) – er lernt direkt aus Applaus und Buhrufen.

```
Policy π_θ(a|s)
     │
     ▼ Wählt Aktion a
Umgebung → Reward r
     │
     ▼ Wie gut war die Aktion?
Gradient → θ wird angepasst
```

---

### Visualisierungsidee: Kontinuierlicher Aktionsraum

```
Value-Based (diskret):        Policy-Gradient (kontinuierlich):
                                
Q-Tabelle / Netz:             Policy-Netz:
┌────┬────┬────┐              ┌──────────────────────────┐
│ ↑  │ ↓  │ → │              │ Input: Zustand s          │
├────┼────┼────┤              │ Output: μ = 0.3, σ = 0.1 │
│0.4 │0.7 │0.2 │              │ → Aktion ~ N(0.3, 0.1)   │
└────┴────┴────┘              └──────────────────────────┘

argmax → Aktion ↓ (diskret)   Sample → Kraft = 0.28 N (kontinuierlich)
```

---

### Häufige Missverständnisse

> ❌ **„Policy Gradient ist immer besser als DQN"**  
> ✅ Es kommt auf den Anwendungsfall an. DQN ist effektiver bei kleinen diskreten Räumen und dateneffizienter. Policy-Gradient-Methoden glänzen bei kontinuierlichen Aktionsräumen und stochastischen Policies.

> ❌ **„Policy Gradient braucht kein neuronales Netz"**  
> ✅ Prinzipiell könnte jede differenzierbare Parametrisierung verwendet werden, aber in der Praxis sind neuronale Netze der Standard.

> ❌ **„Wir lernen direkt die optimale Aktion"**  
> ✅ Wir lernen eine **Wahrscheinlichkeitsverteilung** über Aktionen. Die Aktion wird dann gesampelt.

---

### Mini-Quiz 1

**Frage 1:** Warum kann DQN nicht direkt auf kontinuierliche Aktionsräume angewendet werden?  
> *Antwort:* Das `argmax` über eine unendliche (kontinuierliche) Menge ist nicht effizient berechenbar. DQN bräuchte unendlich viele Ausgabeneuronen oder müsste die Optimierung anders lösen.

**Frage 2:** In welchem Szenario ist eine stochastische Policy notwendig?  
> *Antwort:* Wenn die optimale Strategie Zufälligkeit erfordert – z.B. in Spielen mit unvollständiger Information (Poker, Stein-Schere-Papier), wo eine deterministische Policy vom Gegner ausgenutzt werden kann.

**Frage 3:** Was optimiert Policy Gradient direkt, was DQN indirekt?  
> *Antwort:* Policy Gradient optimiert direkt die Policy-Parameter θ. DQN optimiert eine Wertfunktion Q(s,a) und leitet die Policy daraus implizit ab.

---

---

# Einheit 3–4 · Mathematische Grundlagen

## Die Policy als Wahrscheinlichkeitsverteilung

Eine **parametrisierte Policy** ordnet jedem Zustand s eine Verteilung über Aktionen zu:

$$π_θ(a|s) = P(Aktion = a | Zustand = s, Parameter θ)$$

**Diskret (Softmax-Policy):**

$$π_θ(a|s) = exp(f_θ(s,a)) / Σ_a' exp(f_θ(s,a'))$$

**Kontinuierlich (Gaussian Policy):**

$$π_θ(a|s) = N(μ_θ(s), σ_θ(s))$$

Das Netz gibt Mittelwert μ und Standardabweichung σ aus, die Aktion wird gesampelt.

---

## Das Optimierungsziel J(θ)

Wir wollen den **erwarteten kumulierten Return** maximieren:

$$J(θ) = E_π[G_0] = E_π[Σ_{t=0}^{T} γ^t · r_t]$$

Dabei ist:

- `G_t` der Return ab Zeitschritt t
- `γ` der Diskontfaktor
- Die Erwartung hängt von der Policy π_θ ab

---

## Der Policy-Gradient-Theorem

### Das Problem

Wir wollen `∇_θ J(θ)` berechnen. Aber J(θ) hängt von der Verteilung der Trajektorien ab – und diese Verteilung ändert sich mit θ. Wie leiten wir durch die Umgebungsdynamik ab?

### Der Log-Likelihood-Trick

Für jede Funktion p(x;θ) gilt:

$$∇_θ p(x;θ) = p(x;θ) · ∇_θ log p(x;θ)$$

Dieser Trick erlaubt es uns, den Gradienten als **Erwartungswert** zu schreiben – ohne die Umgebungsdynamik ableiten zu müssen.

### Das Policy-Gradient-Theorem (Kern)

$$∇_θ J(θ) = E_π[ ∇_θ log π_θ(a|s) · Q^π(s,a) ]$$

**Intuition:**

- `∇_θ log π_θ(a|s)` → „In welche Richtung ändert sich log π_θ, wenn ich θ anpasse?"
- `Q^π(s,a)` → „Wie gut war diese Aktion langfristig?"
- Produkt: „Passe θ in die Richtung an, die gute Aktionen wahrscheinlicher macht."

---

### Visualisierungsidee: Gradient als Wahrscheinlichkeitsverschiebung

```
Vor dem Update:          Nach dem Update (gute Aktion):
                         
P(a|s)                   P(a|s)
  ┃                         ┃
  ┃  ■ ■                    ┃    ■ ■ ■
  ┃■ ■ ■ ■                  ┃  ■ ■ ■ ■ ■
  ┗━━━━━━━━▶ a              ┗━━━━━━━━━━▶ a
        ↑                           ↑
   gewählte Aktion            mehr Wahrsch. für diese Aktion
```

Wenn eine Aktion einen hohen Return hatte → erhöhe ihre Wahrscheinlichkeit.  
Wenn eine Aktion einen niedrigen Return hatte → senke ihre Wahrscheinlichkeit.

---

### Warum brauchen wir keine Ableitung durch die Umgebung?

$$J(θ) = Σ_τ P(τ|θ) · R(τ)$$

$$∇_θ J(θ) = Σ_τ ∇_θ P(τ|θ) · R(τ)
           = Σ_τ P(τ|θ) · ∇_θ log P(τ|θ) · R(τ)
           = E_π[ ∇_θ log P(τ|θ) · R(τ) ]$$

Die Trajektorienwahrscheinlichkeit P(τ|θ) enthält die Umgebungsdynamik P(s'|s,a):

$$log P(τ|θ) = Σ_t log π_θ(a_t|s_t) + Σ_t log P(s_{t+1}|s_t, a_t)$$

Der zweite Term hängt **nicht von θ ab** – sein Gradient ist null. Es bleibt:

$$∇_θ log P(τ|θ) = Σ_t ∇_θ log π_θ(a_t|s_t)$$

> **Die Umgebungsdynamik fällt heraus.** Wir brauchen kein Modell der Welt.

---

### Stochastisch vs. deterministisch

| Eigenschaft | Stochastische Policy | Deterministische Policy |
|-------------|---------------------|------------------------|
| Ausgabe | Verteilung P(a\|s) | Direkte Aktion μ(s) |
| Exploration | Intrinsisch | Explizit (Noise nötig) |
| On-Policy | Ja (Standard) | Oft Off-Policy (DDPG) |
| Algorithmen | REINFORCE, A2C, PPO | DDPG, TD3 |
| Continuous Control | ✓ | ✓ (effizienter) |

---

### Häufige Missverständnisse

> ❌ **„Wir leiten durch die Umgebung ab"**  
> ✅ Der Log-Likelihood-Trick eliminiert die Umgebungsdynamik aus dem Gradienten. Policy Gradient ist modellfreie Optimierung.

> ❌ **„Q^π(s,a) muss exakt bekannt sein"**  
> ✅ In der Praxis wird Q^π geschätzt – entweder durch Monte-Carlo-Returns (REINFORCE) oder durch einen Critic (Actor-Critic).

> ❌ **„Höherer Return → immer stärkeres Update"**  
> ✅ Es geht um **relative** Güte. Eine Baseline (z.B. V(s)) macht den Gradienten stabiler – darauf kommen wir in REINFORCE.

---

### Mini-Quiz 2

**Frage 1:** Was bedeutet `∇_θ log π_θ(a|s)` anschaulich?  
> *Antwort:* Es ist die Richtung im Parameterraum, in die die Wahrscheinlichkeit der Aktion a im Zustand s am stärksten steigt.

**Frage 2:** Warum fällt die Umgebungsdynamik P(s'|s,a) aus dem Policy-Gradienten heraus?  
> *Antwort:* Weil P(s'|s,a) nicht von θ abhängt. Beim Ableiten des Log-Terms nach θ verschwindet dieser Summand.

**Frage 3:** Welchen Vorteil hat eine Gaussian Policy bei kontinuierlichen Aktionen?  
> *Antwort:* Das Netz gibt μ und σ aus, die Aktion wird gesampelt. Damit ist die Policy differenzierbar und kann direkt mit Gradientenaufstieg optimiert werden.

---

---

# Einheit 5–6 · REINFORCE – Der Einstieg

## Monte-Carlo Policy Gradient

REINFORCE ist der einfachste Policy-Gradient-Algorithmus. Die Idee:

1. Führe eine **vollständige Episode** durch
2. Berechne den Return G_t für jeden Zeitschritt
3. Update θ in Richtung des Policy-Gradienten

### Update-Regel

$$θ ← θ + α · Σ_t ∇_θ log π_θ(a_t|s_t) · G_t$$

Dabei ist:

- `G_t = Σ_{k=t}^{T} γ^{k-t} · r_k` der diskontierte Return ab t
- `α` die Lernrate

---

### Algorithmus: REINFORCE

```
Initialisiere θ zufällig
Für jede Episode:
  1. Erzeuge Trajektorie τ = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T)
     mit Policy π_θ
  2. Für jeden Zeitschritt t = 0, 1, ..., T:
     a) Berechne §§G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...$$
     b) $$∇_θ ← ∇_θ log π_θ(a_t|s_t) · G_t$$
  3. $$θ ← θ + α · Σ_t ∇_θ$$
```

---

### Visualisierungsidee: Episodenbasiertes Lernen

```
Episode 1:  s₀→a₀→r₁→s₁→a₁→r₂→s₂→ENDE    G₀=5, G₁=3, G₂=2
Episode 2:  s₀→a₀→r₋₁→s₁→a₁→r₋₂→ENDE     G₀=−3, G₁=−2
Episode 3:  s₀→a₀→r₃→s₁→a₁→r₄→ENDE       G₀=7, G₁=4

Nach jeder Episode: Update der Policy-Parameter
↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓

Gute Aktionen (G > 0) → höhere Wahrscheinlichkeit
Schlechte Aktionen (G < 0) → niedrigere Wahrscheinlichkeit
```

---

## Das Problem: Hohe Varianz

REINFORCE hat strukturell **hohe Varianz** im Gradienten:

```
Zwei identische (s, a)-Paare → unterschiedliche G_t
weil zufällige zukünftige Rewards (Rauschen) eingeflossen sind
```

### Ursachen der Varianz

1. **Zufällige Umgebung:** Gleiche Aktion kann verschiedene Rewards liefern
2. **Lange Episoden:** G_t ist Summe vieler Zufallsvariablen → Varianz addiert sich
3. **Monte-Carlo-Schätzung:** Kein Bootstrapping → hohe Varianz, kein Bias

### Konsequenzen

- Sehr langsames Lernen (viele Episoden nötig)
- Große Schwankungen in den Updates
- Training kann instabil werden

---

## Baseline zur Varianzreduktion

Eine **Baseline** b(s) wird vom Return subtrahiert:

$$θ ← θ + α · ∇_θ log π_θ(a|s) · (G_t - b(s))$$

Die optimale Baseline ist die **Wertfunktion V^π(s)**:

$$b(s) = V^π(s)$$
   →   "Wie gut ist dieser Zustand im Durchschnitt?"

Dann wird `G_t - V(s)` zum **Advantage**:

```
A(s,a) ≈ G_t - V(s)   ("War diese Aktion besser oder schlechter als erwartet?")
```

### Warum reduziert die Baseline die Varianz?

```
Ohne Baseline: Update proportional zu G_t (absolut, hohes Rauschen)
Mit Baseline:  Update proportional zu A(s,a) = G_t - V(s) (relativ, geringes Rauschen)

Beispiel:
  $$G_t$$ = 100    →  starkes positives Update  (aber war es wirklich gut?)
  $$G_t$$ = 90     →  schwaches positives Update
  
  Mit V(s) = 95:
  A = 100 - 95 = +5   →  leicht positiv (besser als Durchschnitt)
  A = 90 - 95  = -5   →  leicht negativ (schlechter als Durchschnitt)
```

### Wichtig: Die Baseline beeinflusst nicht den erwarteten Gradienten

$$E_π[∇_θ log π_θ(a|s) · b(s)] = b(s) · E_π[∇_θ log π_θ(a|s)] = b(s) · 0 = 0$$

Die Baseline verändert die Varianz, aber **nicht den Erwartungswert** des Gradienten.

---

### Didaktische Analogie: Bergsteigen im Nebel

> REINFORCE ist wie das Messen der Steigung eines Berges aus verrauschten GPS-Punkten. Jeder Messpunkt ist ungenau. Eine Baseline ist wie ein lokal gemitteltes Höhenmodell – du weißt ungefähr, wo du stehst, und kannst relative Unterschiede besser erkennen.

---

### Beispiel: CartPole mit REINFORCE

**Aufgabe:** Halte einen Stab auf einem Wagen im Gleichgewicht.  
**Zustand:** Position, Geschwindigkeit, Winkel, Winkelgeschwindigkeit (4 Werte)  
**Aktionen:** Links oder rechts (diskret, 2 Aktionen)

```python
# Policy-Netz (Softmax-Ausgabe für diskrete Aktionen)
import torch
import torch.nn as nn

class PolicyNet(nn.Module):
    def __init__(self, state_dim=4, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state)  # Gibt P(a|s) zurück

# Update-Schritt
def reinforce_update(log_probs, returns, optimizer):
    loss = -torch.stack(
        [lp * G for lp, G in zip(log_probs, returns)]
    ).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

### Visualisierungsidee: Trainingskurve REINFORCE vs. REINFORCE+Baseline

```
Return (Episode)
200 ┃     ·                · REINFORCE + Baseline
    ┃   ·   ·               ·  ·
150 ┃ ·       · ·          ·     ·
    ┃·          · ·      ·         ·  ·
100 ┃             ·    ·              ·  ·    ← REINFORCE (roh)
    ┃              ····                 ···
 50 ┃
    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━▶ Episoden
                50        100       150
```

---

### Häufige Missverständnisse

> ❌ **„REINFORCE ist unbiased, also optimal"**  
> ✅ Unbiased bedeutet: der Erwartungswert des Gradienten ist korrekt. Aber hohe Varianz macht das Lernen extrem langsam. In der Praxis braucht man Varianzreduktion.

> ❌ **„Eine Baseline verändert die Richtung des Lernens"**  
> ✅ Nein. Eine Baseline beeinflusst nicht den Erwartungswert des Gradienten – sie reduziert nur die Varianz.

> ❌ **„Monte-Carlo-Returns sind exakt"**  
> ✅ Sie sind unbiased, aber haben hohe Varianz. Zudem brauchen sie vollständige Episoden – bei langen/endlosen Episoden ein Problem.

---

### Mini-Quiz 3

**Frage 1:** Warum wartet REINFORCE immer bis zum Episodenende?  
> *Antwort:* Weil G_t die Summe aller zukünftigen Rewards ist – die erst nach dem Episodenende vollständig bekannt ist.

**Frage 2:** Welche Baseline reduziert die Varianz am stärksten?  
> *Antwort:* Die optimale Baseline ist die State-Value-Funktion V^π(s), weil sie den erwarteten Return aus diesem Zustand schätzt und damit die Fluktuation um den Erwartungswert minimiert.

**Frage 3:** Was ist der Unterschied zwischen Return G_t und Advantage A(s,a)?  
> *Antwort:* G_t ist der absolute Monte-Carlo-Return. A(s,a) = G_t - V(s) misst, wie viel besser die Aktion a war als der Durchschnitt in Zustand s (relative Güte).

---

---

# Einheit 7–8 · Actor-Critic-Methoden

## Motivation: Warum REINFORCE nicht reicht

| Problem REINFORCE | Ursache |
|-------------------|---------|
| Hohe Varianz | Vollständinge Monte-Carlo-Returns |
| Langsames Lernen | Episodenbasiert, kein Online-Update |
| Schlechte Dateneffizienz | Viele Episoden für signifikantes Update |

**Lösung:** Kombiniere Policy Gradient mit einer gelernten Wertfunktion.

---

## Architektur: Actor + Critic

```
       ┌─────────────────────────────┐
       │           Zustand s          │
       └───────────┬─────────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
   ┌──────────┐        ┌──────────┐
   │  ACTOR   │        │  CRITIC  │
   │ $$π_θ(a|s)$$ │        │  $$V_φ(s)$$  │
   └──────────┘        └──────────┘
   Wählt Aktion a      Bewertet Zustand s
         │                   │
         ▼                   ▼
    Aktion ausführen    Advantage schätzen:
    a ~ π_θ(a|s)        A(s,a) = r + γV(s') - V(s)
         │                   │
         └─────── Update ─────┘
         Actor: maximiere A(s,a)
         Critic: minimiere (r + γV(s') - V(s))²
```

---

## Die Advantage-Funktion

```
A(s, a) = Q(s, a) - V(s)
```

**Interpretation:**

- A > 0: Aktion a war **besser** als der Durchschnitt in s
- A < 0: Aktion a war **schlechter** als der Durchschnitt in s
- A = 0: Aktion war exakt so gut wie der Durchschnitt

**In der Praxis** schätzt man A über den **TD-Fehler** (One-Step-Advantage):

$$A(s,a) ≈ δ = r + γ·V(s') - V(s)$$

Das ist eine **biased, aber niedrig-variante** Schätzung – der Trade-off gegenüber REINFORCE.

---

## TD-Learning im Critic

Der Critic lernt V_φ(s) durch **Temporal Difference**:

Critic-Loss: $$L(φ) = (r + γ·V_φ(s') - V_φ(s))²$$

Update: $$φ ← φ - α_critic · ∇_φ L(φ)$$
```

Ein Schritt nach dem anderen – kein Warten auf das Episodenende.

## Actor-Update

```
Actor-Loss: $$L(θ) = -log π_θ(a|s) · A(s,a)$$

Update: $$θ ← θ + α_actor · ∇_θ log π_θ(a|s) · A(s,a)$$
```

---

### Bias-Varianz-Trade-off

```
                   REINFORCE           Actor-Critic (TD)
                   (Monte-Carlo)       (Bootstrapping)
Schätzung          Unbiased            Biased
Varianz            Hoch                Niedrig
Update-Frequenz    Episodenende        Jeden Schritt
Dateneffizienz     Gering              Höher
```

Visualisierung: Pfeil-Diagramm

```
             Hohe Varianz ←→ Niedriger Bias
Monte-Carlo: ●───────────────────────────────○ TD(0)
             (REINFORCE)                   (1-Step AC)

TD(n) mit n → ∞ nähert sich Monte-Carlo an.
TD(λ) interpoliert mit einem Parameter λ ∈ [0,1].
```

---

## On-Policy vs. Off-Policy Actor-Critic

| Eigenschaft | On-Policy (A2C) | Off-Policy (DDPG, SAC) |
|-------------|-----------------|------------------------|
| Datenquelle | Aktuelle Policy | Replay Buffer |
| Dateneffizienz | Niedrig | Hoch |
| Stabilität | Höher | Geringer (ohne Korrekturen) |
| Korrektur | Keine nötig | Importance Sampling |

---

### Visualisierungsidee: Actor-Critic Online-Update

```
Zeitverlauf:
t=0: s₀ →[Actor]→ a₀ → r₀, s₁
         [Critic] → A(s₀,a₀) = r₀ + γ·V(s₁) - V(s₀)
         [Update Actor + Critic] ← sofort, kein Warten

t=1: s₁ →[Actor]→ a₁ → r₁, s₂
         [Critic] → A(s₁,a₁)
         [Update]

...  Kontinuierliches Lernen, auch in langen Episoden
```

---

### Didaktische Analogie: Trainer und Sportler

> Der **Actor** ist ein Sportler, der Bewegungen ausführt.  
> Der **Critic** ist ein Trainer, der jede Bewegung bewertet ohne das Ende des Spiels abzuwarten.  
> Der Sportler verbessert seine Technik basierend auf dem Feedback des Trainers.  
> Der Trainer korrigiert auch sein eigenes Bewertungsmodell laufend.

---

### Häufige Missverständnisse

> ❌ **„Actor und Critic müssen separate Netze sein"**  
> ✅ In der Praxis teilen Actor und Critic oft die unteren Schichten (Shared Backbone) und haben nur getrennte Ausgabeköpfe. Das spart Parameter und verbessert das Lernen gemeinsamer Repräsentationen.

> ❌ **„Der TD-Fehler ist eine gute Schätzung von Q(s,a)"**  
> ✅ Der TD-Fehler δ = r + γV(s') - V(s) ist eine Schätzung der **Advantage-Funktion**, nicht von Q direkt. Er ist biased (weil V(s') geschätzt ist), aber niedrig-variantz.

> ❌ **„Ein perfekter Critic würde reichen"**  
> ✅ Auch mit perfektem Critic bleibt der Actor-Update verrauscht (Sampling-Varianz). Beide müssen gemeinsam lernen.

---

### Mini-Quiz 4

**Frage 1:** Was liefert der Critic als Ausgabe?  
> *Antwort:* Den geschätzten State-Value V_φ(s) – den erwarteten Return aus Zustand s unter der aktuellen Policy.

**Frage 2:** Warum kann Actor-Critic online lernen, REINFORCE aber nicht (ohne Änderungen)?  
> *Antwort:* Actor-Critic verwendet TD-Bootstrapping: A ≈ r + γV(s') - V(s) ist sofort nach jedem Schritt berechenbar. REINFORCE braucht G_t, das erst am Episodenende bekannt ist.

**Frage 3:** Was ist der Trade-off bei Bootstrapping im Critic?  
> *Antwort:* Bootstrapping reduziert die Varianz erheblich, führt aber einen Bias ein (weil V(s') selbst eine Schätzung ist). Man tauscht Unbiasedness gegen niedrige Varianz.

---

---

# Einheit 9–10 · Fortgeschrittene Varianten

## A2C und A3C

### A2C – Advantage Actor-Critic (synchron)

A2C ist ein synchrones Actor-Critic-Verfahren mit mehreren parallelen Umgebungen:

```
Environment 1 ──┐
Environment 2 ──┼──▶  Zentraler Actor-Critic  ──▶  Update
Environment 3 ──┤     (wartet auf alle Envs)
Environment n ──┘
```

- **Synchrone Updates:** Alle Worker sammeln Daten gleichzeitig, dann Update
- **Stabilere Gradienten** durch Mittelung über mehrere Umgebungen
- **Einfach zu implementieren** (kein Locking-Problem)

### A3C – Asynchronous Advantage Actor-Critic

```
Global Network θ, φ
  │    │    │
  ▼    ▼    ▼
Worker Worker Worker  (jeder mit eigener Env-Kopie)
  ↑    ↑    ↑
  └────┴────┘
  Asynchrone Gradient-Updates (kein Warten)
```

| | A2C | A3C |
|-|-----|-----|
| Update-Typ | Synchron | Asynchron |
| Stabilität | Höher | Geringer (race conditions) |
| Geschwindigkeit | Etwas langsamer | Schneller auf CPU |
| CPU vs. GPU | GPU-freundlicher | CPU-parallel |
| Empfehlung | Standard heute | Historisch wichtig |

---

## TRPO – Trust Region Policy Optimization

### Das Problem: Zu große Policy-Updates

Beim normalen Policy Gradient kann ein Update die Policy **zu stark verändern** – das alte Verhalten wird überschrieben, ohne dass neue Erfahrungen gesammelt wurden.

```
Gute Policy π₀         Schlechter Update         Katastrophaler Verfall
     ■                      ■     ■                      ■
    ■■■         →          ■■■■■■■■■      →        Policy vergisst alles
     ■                      ■     ■                (kein Recovery)
```

### Die Lösung: Trust Region

TRPO begrenzt, wie weit sich die Policy pro Update verändern darf – gemessen durch die **KL-Divergenz**:

```
Maximiere: $$E[π_θ_new(a|s)/π_θ_old(a|s) · A(s,a)]$$
unter der Nebenbedingung:
    $$KL(π_θ_old || π_θ_new) ≤ δ$$
```

**Intuition:** Der Roboter darf seinen Stil verbessern, aber nicht komplett umtrainieren.

```
Policy-Raum:
        │
        │      ┌────────────────┐
        │      │  Trust Region  │
        │      │  (KL ≤ δ)     │
        │      │       ●─────▶● │   erlaubter neuer Parameter
        │      │    θ_alt      │
        │      └────────────────┘
        └─────────────────────────────▶
```

### Nachteil: Aufwändig

TRPO benötigt **konjugiertes Gradientenverfahren** und Liniensuche zur Nebenbedingungsoptimierung – komplex zu implementieren.

---

## PPO – Proximal Policy Optimization

PPO vereinfacht TRPO erheblich und ist heute der **Industriestandard**.

### Die Idee: Clipped Objective

Statt einer harten KL-Nebenbedingung verwendet PPO einen **geclippen Verlust**:

```
Probability Ratio: $$r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)$$

PPO-Clipped Objective:
$$L^CLIP(θ) = E[ min( r_t(θ)·A_t,  clip(r_t(θ), 1-ε, 1+ε)·A_t ) ]$$
```

Mit typischerweise ε = 0.2.

### Visualisierungsidee: Clipping-Mechanismus

```
Objective   │
            │          (1+ε)·A     ←  Deckel: verhindert zu starkes Update
            │        ╔════════════
            │      ╔═╝
            │    ╔═╝    ← unkritischer Bereich
            │  ╔═╝
────────────╬═╝─────────────────────────── r_t
            │  ╚═╗
            │    ╚═╗
            │      ╚═════════════
            │          (1-ε)·A     ←  Boden: verhindert zu starkes Update
```

Wenn r_t(θ) zu groß oder zu klein wird → Gradient wird abgeschnitten → kein übermäßiger Update.

### PPO-Algorithmus

```
Für jede Iteration:
  1. Sammle T Zeitschritte mit aktueller Policy π_θ_old
  2. Berechne Advantages A_t (mit GAE)
  3. Optimiere L^CLIP über K Epochen mit Mini-Batches
  4. Setze θ_old ← θ
```

### Warum PPO der Standard wurde

| Eigenschaft | TRPO | PPO |
|-------------|------|-----|
| Implementierung | Komplex | Einfach |
| Performance | Sehr gut | Sehr gut |
| Rechenaufwand | Hoch (Hesse-Matrix) | Niedrig |
| Hyperparameter | Viele | Wenige |
| Stabilität | Hoch | Hoch |

---

## DDPG, TD3 und SAC

### DDPG – Deep Deterministic Policy Gradient

Für **kontinuierliche Aktionsräume** mit deterministischer Policy:

```
$$μ_θ: s → a$$   (direkte Aktion, keine Verteilung)

Actor-Update: $$∇_θ J ≈ E[ ∇_a Q_φ(s,a)|_{a=μ_θ(s)} · ∇_θ μ_θ(s) ]$$

Exploration: Noise wird zur Aktion addiert (Ornstein-Uhlenbeck oder N(0,σ²))
```

**Architektur:**

- Actor: Policy-Netz μ_θ
- Critic: Q-Funktion Q_φ(s,a)
- Target Networks für beide (stabile Q-Targets)
- Replay Buffer (Off-Policy)

### TD3 – Twin Delayed Deep Deterministic

TD3 behebt drei Probleme von DDPG:

| Problem | TD3-Lösung |
|---------|-----------|
| Q-Überschätzung | **Twin Critics:** Minimum von zwei Q-Netzen |
| Instabilität | **Delayed Actor Update:** Critic öfter updaten als Actor |
| Overfit auf Q-Fehler | **Target Policy Noise:** Noise auf Zielaktionen |

### SAC – Soft Actor-Critic

SAC fügt **Entropy-Regularisierung** hinzu:

```
Ziel: $$π* = argmax E[Σ γ^t (r_t + α·H(π(·|s_t)))]$$

Dabei: $$H(π(·|s)) = -E[log π(a|s)]$$  (Entropie der Policy)
```

**Warum Entropie?**

- Fördert Exploration
- Vermeidet frühzeitiges Kollabieren auf deterministische Policy
- Robuster gegenüber Hyperparameter-Wahl

| Algorithmus | Aktionsraum | Policy | Besonderheit |
|-------------|-------------|--------|-------------|
| DDPG | Kontinuierlich | Deterministisch | Einfach, aber instabil |
| TD3 | Kontinuierlich | Deterministisch | Stabil, 3 Fixes |
| SAC | Kontinuierlich | Stochastisch | Entropy-Regularisierung, SOTA |

---

### Häufige Missverständnisse

> ❌ **„PPO ist exakt so sicher wie TRPO"**  
> ✅ PPO ist eine Approximation von TRPO. Der Clip-Mechanismus ist eine heuristische Annäherung an die Trust-Region-Bedingung, aber keine exakte Garantie.

> ❌ **„Entropy Bonus macht SAC langsamer (mehr Exploration)"**  
> ✅ Entropy-Regularisierung verbessert oft die Dateneffizienz, weil die Policy nicht vorzeitig auf suboptimale Aktionen konvergiert.

> ❌ **„DDPG ist besser als PPO, weil es off-policy ist"**  
> ✅ Off-Policy bedeutet höhere Dateneffizienz, aber auch größere Instabilität. Für viele Standardprobleme ist PPO die robustere Wahl.

---

### Mini-Quiz 5

**Frage 1:** Was begrenzt TRPO und wie unterscheidet sich PPO davon?  
> *Antwort:* TRPO begrenzt die KL-Divergenz zwischen alter und neuer Policy als harte Nebenbedingung. PPO approximiert das durch einen geclippen Objective-Term, der einfacher zu implementieren ist.

**Frage 2:** Wofür benötigt DDPG einen Replay Buffer?  
> *Antwort:* DDPG ist Off-Policy und kann Erfahrungen wiederverwenden. Der Replay Buffer speichert vergangene (s,a,r,s')-Tupel und stellt Mini-Batches für das Training bereit.

**Frage 3:** Wie verhindert TD3 die Q-Überschätzung?  
> *Antwort:* TD3 trainiert zwei separate Q-Netze und nimmt das Minimum der beiden als Zielwert. Das verhindert systematische Überschätzung, die bei DDPG auftritt.

---

---

# Einheit 11 · Stabilitätsmechanismen & Regularisierung

## Überblick der Stabilitätswerkzeuge

Training mit Policy Gradient kann instabil sein. Diese Techniken helfen:

| Technik | Problem | Lösung |
|---------|---------|--------|
| KL-Divergenz-Kontrolle | Zu große Updates | Begrenzt Änderung der Policy |
| Entropy Bonus | Verfrühte Konvergenz | Hält Policy explorativ |
| Gradient Clipping | Explodierende Gradienten | Begrenzt Gradientenlänge |
| Normalisierung | Schiefe Verteilungen | Stabilisiert Input/Output |
| Importance Sampling | Off-Policy-Bias | Korrigiert Verteilungsshift |
| Replay Buffer | Datenefizienz | Wiederverwendung alter Daten |

---

## KL-Divergenz als Sicherheitsgürtel

Die KL-Divergenz misst, wie weit sich zwei Wahrscheinlichkeitsverteilungen unterscheiden:

$$KL(π_old || π_new) = E_{a~π_old}[ log(π_old(a|s) / π_new(a|s)) ]$$

- KL = 0: Identische Policies
- KL > 0: Policies unterscheiden sich
- KL → ∞: vollständig verschiedene Policies

Verwendet in:

- TRPO (als Nebenbedingung)
- PPO (als Monitoring-Metrik, optional als zusätzlicher Loss-Term)

---

## Entropy Bonus

$$L_total = L_policy - β · H(π_θ(·|s))$$

Dabei: $$H(π) = -Σ_a π(a|s) · log π(a|s)$$

- **Hohe Entropie:** Policy ist explorativ (flache Verteilung)
- **Niedrige Entropie:** Policy ist deterministisch (scharfe Verteilung)
- Der Faktor β balanciert Exploitation vs. Exploration

### Visualisierungsidee: Entropieverlauf

```
Entropie
  2.0 ┃ ────────────
      ┃         ╲
  1.0 ┃          ╲────────────────
      ┃               ╲
  0.0 ┃                ╲──────── (Policy kollabiert auf deterministische Aktion ← Problem!)
      ┗━━━━━━━━━━━━━━━━━━━━━━━━━━▶ Training-Schritte
```

Wenn Entropie frühzeitig auf 0 fällt → Exploration stoppt → lokales Optimum.

---

## Gradient Clipping

```python
# Verhindert explodierende Gradienten
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
```

Alle Gradienten werden skaliert, sodass ihre L2-Norm nicht max_norm überschreitet.

---

## Normalisierung

### Observation Normalisierung

$$s_norm = (s - μ_s) / σ_s$$   (laufende Statistiken)

### Reward Normalisierung / Skalierung

$$r_norm = r / σ_returns$$   (laufende Standardabweichung der Returns)

### Advantage Normalisierung

$$A_norm = (A - mean(A)) / (std(A) + ε)$$

Verhindert, dass ein Batch mit hohen oder niedrigen Returns extreme Updates erzeugt.

---

## Importance Sampling (Off-Policy)

Wenn Daten von einer alten Policy π_old stammen, aber die aktuelle Policy π_θ gelernt werden soll:

$$E_{a~π_θ}[f(a)] ≈ E_{a~π_old}[π_θ(a|s)/π_old(a|s) · f(a)]$$

Importance Weight: $$ρ = π_θ(a|s) / π_old(a|s)$$

Ohne Korrektur würde das Training auf der falschen Verteilung optimieren.

---

## Generalized Advantage Estimation (GAE)

GAE interpoliert zwischen Monte-Carlo (niedrig-Bias, hohe Varianz) und TD(0) (hoch-Bias, niedrig-Varianz):

$$Â_t^GAE(γ,λ) = Σ_{k=0}^{∞} (γλ)^k · δ_{t+k}$$

Dabei: $$δ_t = r_t + γ·V(s_{t+1}) - V(s_t)$$   (TD-Fehler)

| λ = 0 | TD(0) | Hoch-Bias, niedrig-Varianz |
| λ = 1 | Monte-Carlo | Niedrig-Bias, hoch-Varianz |
| λ = 0.95 | Standard PPO | Balance |

---

### Häufige Missverständnisse

> ❌ **„Gradient Clipping verlangsamt das Training"**  
> ✅ Im Gegenteil – ohne Clipping können explodierende Gradienten das Training zum Absturz bringen. Clipping macht das Training robuster und oft schneller insgesamt.

> ❌ **„Reward-Normalisierung ändert die optimale Policy"**  
> ✅ Lineare Skalierung und Zentrierung verändern nicht, welche Policy optimal ist (da die Ordnung erhalten bleibt). Sie verbessern aber die numerische Stabilität.

---

### Mini-Quiz 6

**Frage 1:** Was passiert, wenn β (Entropy-Koeffizient) zu groß ist?  
> *Antwort:* Die Policy wird zu explorativ und konvergiert nicht mehr. Der Exploration-Bonus dominiert den Reward-Signal und das Training lernt keine sinnvolle Policy.

**Frage 2:** Warum ist Advantage-Normalisierung pro Batch sinnvoll?  
> *Antwort:* Sie stellt sicher, dass der Durchschnitt des Advantage nahe null liegt und die Varianz kontrolliert ist, unabhängig vom absoluten Reward-Level des Batches.

**Frage 3:** Was ist der Unterschied zwischen λ=0 und λ=1 in GAE?  
> *Antwort:* λ=0 entspricht einem reinen One-Step-TD-Fehler (hoher Bias, niedrige Varianz), λ=1 entspricht Monte-Carlo-Returns (kein Bias, hohe Varianz). λ ≈ 0.95 ist der übliche Kompromiss.

---

---

# Einheit 12 · Implementierungsdetails

## Policy-Netzwerkdesign

### Gaussian Policy (kontinuierliche Aktionen)

```python
class GaussianPolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        self.mean_head = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # gelernt oder fest

    def forward(self, state):
        x = self.shared(state)
        mean = self.mean_head(x)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()  # reparameterization trick
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob
```

### Tanh-Squashing (für begrenzte Aktionen)

```python
# Aktionen in [-1, 1] begrenzen
raw_action = dist.rsample()
action = torch.tanh(raw_action)

# Korrektur der Log-Likelihood (Änderung der Variablen)
log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(-1)
```

---

## Shared Backbone: Actor + Critic

```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        self.actor_head = nn.Linear(256, action_dim)
        self.critic_head = nn.Linear(256, 1)

    def forward(self, state):
        x = self.shared(state)
        logits = self.actor_head(x)   # für Softmax (diskret)
        value = self.critic_head(x)
        return logits, value
```

---

## GAE-Berechnung

```python
def compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
        next_value = values[t]
    return torch.tensor(advantages)
```

---

## PPO-Trainingsloop (vereinfacht)

```python
def ppo_update(actor_critic, optimizer, states, actions, old_log_probs,
               advantages, returns, clip_eps=0.2, epochs=10):
    for _ in range(epochs):
        logits, values = actor_critic(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)

        # Policy-Loss (Clipped)
        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value-Loss
        value_loss = (returns - values.squeeze()).pow(2).mean()

        # Entropy-Bonus
        entropy = dist.entropy().mean()

        # Gesamtverlust
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(actor_critic.parameters(), 0.5)
        optimizer.step()
```

---

## Typische Hyperparameter

| Hyperparameter | PPO Standard | Wirkung |
|----------------|-------------|---------|
| Lernrate | 3e-4 | Schrittgröße der Optimierung |
| γ (Diskontfaktor) | 0.99 | Gewichtung zukünftiger Rewards |
| λ (GAE) | 0.95 | Bias-Varianz-Trade-off im Advantage |
| ε (Clip) | 0.2 | Maximale Policy-Änderung pro Update |
| Batch-Größe | 2048–4096 | Stabilität des Gradienten |
| Mini-Batch-Größe | 64–256 | Stochastischer Gradient |
| Epochen pro Update | 10 | Datenwiederverwendung |
| Entropy-Koeffizient β | 0.01 | Explorations-Anreiz |
| Value-Loss-Gewicht | 0.5 | Balance Actor-/Critic-Loss |
| Gradient-Clipping | 0.5 | Stabilitäts-Sicherung |

---

## Logging & Debugging

Wichtige Metriken zum Tracken:

```
Training-Metriken:
├── Episode Return (mean, std)
├── Policy-Entropy H(π)           ← Exploration-Indikator
├── KL-Divergenz KL(π_old||π_new) ← Update-Größe
├── Value-Loss                    ← Critic-Qualität
├── Policy-Loss                   ← Actor-Learning-Signal
├── Clip-Fraction                 ← Anteil geclippter Updates
└── Gradientennorm                ← Instabilitäts-Indikator
```

---

---

# Einheit 13 · Typische Fehlerbilder & Debugging

## Checkliste: Wenn der Agent nicht lernt

```
Schritt 1: Reward-Signal prüfen
  [ ] Erhält der Agent überhaupt Rewards?
  [ ] Ist reward shaping korrekt?
  [ ] Sind Rewards normalisiert?

Schritt 2: Explorations-Problem?
  [ ] Sinkt Entropie zu schnell?
  [ ] Ist Entropy Bonus aktiv?

Schritt 3: Technische Probleme?
  [ ] Loss-Kurven stabil?
  [ ] Gradientennorm explodiert?
  [ ] Learning Rate zu groß/klein?
```

---

## Fehler 1: Policy kollabiert auf deterministische Aktion

**Symptome:**

- Entropie fällt schnell auf fast 0
- Agent macht immer die gleiche Aktion
- Kein weiterer Lernfortschritt

**Ursache:**

- Entropy-Koeffizient β zu klein oder 0
- Lernrate zu groß (frühzeitige Konvergenz)
- Reward-Signal zu klar (keine Exploration nötig anfangs)

**Diagnose:**

```
Entropie-Plot: ──────╲____ (kollabiert)     ← Problem
               ──────────~ (bleibt stabil)   ← OK
```

**Lösungen:**

- β erhöhen (z.B. 0.01 → 0.05)
- Lernrate reduzieren
- Entropy Clipping einführen

---

## Fehler 2: Explodierende Gradienten

**Symptome:**

- Loss springt zu NaN
- Gradientennorm > 100
- Training bricht zusammen

**Ursache:**

- Gradient Clipping fehlt oder max_norm zu groß
- Lernrate zu hoch
- Schiefe Advantage-Verteilung

**Lösungen:**

```python
# 1. Gradient Clipping aktivieren
torch.nn.utils.clip_grad_norm_(params, max_norm=0.5)

# 2. Advantage normalisieren
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

# 3. Lernrate halbieren
```

---

## Fehler 3: Reward-Plateaus

**Symptome:**

- Return steigt anfangs, stagniert dann
- Kein Durchbruch trotz langem Training

**Ursache:**

- Lokales Optimum durch zu geringe Exploration
- Reward-Shaping zu schwach (sparse rewards)
- Faulty Hyperparameter (γ zu klein)

**Diagnose:**

```
Return
200 ┃         ·····················
    ┃      ···
100 ┃   ···
    ┃···
  0 ┃
    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━▶ Episoden
         ↑ stagniert frühzeitig
```

**Lösungen:**

- Entropy Bonus erhöhen
- Reward-Shaping einführen (dichtere Rewards)
- γ erhöhen (längerfristiges Denken)
- Curriculum Learning (einfachere Tasks zuerst)

---

## Fehler 4: Schlechte Exploration

**Symptome:**

- Policy konvergiert schnell auf suboptimale Aktionen
- Performance schlechter als Zufallsagent in manchen Zuständen

**Ursache:**

- Entropy-Koeffizient β zu niedrig
- Aktionsraum schlecht skaliert
- Initialisierung ungünstig

---

## Fehler 5: Critic-Divergenz

**Symptome:**

- Value-Loss wächst unbegrenzt
- Advantage-Schätzungen werden extrem groß
- Training instabil

**Ursache:**

- Learning Rate des Critics zu groß
- Return-Skala zu groß (fehlende Normalisierung)
- Bootstrapping-Target zu wenig stabilisiert

**Lösungen:**

```python
# 1. Separate (niedrigere) Lernrate für Critic
optimizer = torch.optim.Adam([
    {'params': actor_params, 'lr': 3e-4},
    {'params': critic_params, 'lr': 1e-3}
])

# 2. Reward-Normalisierung
r_normalized = r / running_std_returns

# 3. Value-Loss-Clipping (PPO-Variante)
value_loss_clipped = (values_clipped - returns).pow(2)
```

---

## Diagnose-Dashboard

```
Korrekte Trainingskurven:
┌──────────────────┬────────────────────┐
│ Episode Return   │ Value Loss         │
│  200 ┃  /‾‾‾‾   │  0.5 ┃╲           │
│  100 ┃ /        │  0.2 ┃ ╲───────   │
│    0 ┃/         │  0.0 ┃            │
│      └─── t     │      └──── t      │
├──────────────────┼────────────────────┤
│ Policy Entropy   │ KL-Divergenz       │
│  2.0 ┃───────── │ 0.02 ┃─~──~──     │
│  1.0 ┃     ─── │ 0.01 ┃            │
│  0.0 ┃         │ 0.00 ┃            │
│      └─── t     │      └──── t      │
└──────────────────┴────────────────────┘
```

---

### Mini-Quiz 7

**Frage 1:** Der Value-Loss explodiert. Was sind die ersten zwei Dinge, die du prüfst?  
> *Antwort:* 1. Reward-Normalisierung: Sind die Returns in einem sinnvollen Bereich? 2. Lernrate des Critics: Ist sie zu groß?

**Frage 2:** Die Entropie ist nach 10.000 Schritten auf 0.01 gefallen. Ist das ein Problem?  
> *Antwort:* Es hängt von der Umgebung ab. Wenn der Agent noch nicht gut performt, ist es ein Problem (zu frühe Konvergenz). Lösung: β erhöhen.

**Frage 3:** Der Agent lernt initial gut, stagniert dann bei 50% der optimalen Performance. Welche Hypothese prüfst du zuerst?  
> *Antwort:* Explorations-Defizit. Prüfe die Entropiekurve. Wenn sie kollabiert ist: β erhöhen und Training neu starten.

---

---

# Einheit 14 · Praxisblöcke, Zusammenfassung & Cheat-Sheet

## Praxisblock 1: REINFORCE minimal implementieren

**Aufgabe:** Implementiere REINFORCE für CartPole-v1.

```python
import gym, torch, torch.nn as nn
from torch.distributions import Categorical

env = gym.make("CartPole-v1")

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        return Categorical(logits=self.net(x))

policy = Policy()
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

for episode in range(1000):
    state, _ = env.reset()
    log_probs, rewards = [], []

    while True:
        dist = policy(torch.FloatTensor(state))
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        state, reward, done, truncated, _ = env.step(action.item())
        rewards.append(reward)
        if done or truncated: break

    # Returns berechnen
    G, returns = 0, []
    for r in reversed(rewards):
        G = r + 0.99 * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Baseline

    # Policy Update
    loss = -sum(lp * G for lp, G in zip(log_probs, returns))
    optimizer.zero_grad(); loss.backward(); optimizer.step()

    if episode % 50 == 0:
        print(f"Episode {episode}: Return = {sum(rewards):.0f}")
```

---

## Praxisblock 2: Actor-Critic mit Advantage

**Erweiterung:** Füge einen Critic hinzu, der V(s) lernt und den Advantage schätzt.

```python
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(4, 64), nn.ReLU())
        self.actor = nn.Linear(64, 2)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        h = self.shared(x)
        dist = Categorical(logits=self.actor(h))
        value = self.critic(h)
        return dist, value

# Im Training:
dist, value = model(state_tensor)
action = dist.sample()
log_prob = dist.log_prob(action)

# Nächster Schritt:
next_state, reward, done, *_ = env.step(action.item())
_, next_value = model(torch.FloatTensor(next_state))

# Advantage (TD-Fehler):
advantage = reward + 0.99 * next_value.detach() * (1 - done) - value

actor_loss  = -log_prob * advantage.detach()
critic_loss = advantage.pow(2)
entropy     = dist.entropy()

loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
```

---

## Praxisblock 3: PPO auf Pendulum (Continuous Control)

**Aufgabe:** Trainiere PPO auf `Pendulum-v1` (kontinuierlicher Aktionsraum: Drehmoment ∈ [−2, 2]).

Schlüsselpunkte gegenüber CartPole:

1. Policy gibt μ, σ aus (Gaussian Policy)
2. Tanh-Squashing für begrenzte Aktionen
3. GAE für Advantage-Schätzung
4. Mehrere Update-Epochen pro Rollout

---

## Praxisblock 4: Hyperparameter-Tuning-Challenge

Ändere jeweils **einen Parameter** und beobachte den Effekt:

| Parameter | Änderung | Erwarteter Effekt |
|-----------|----------|-------------------|
| ε (Clip) | 0.2 → 0.5 | Größere Updates, mögliche Instabilität |
| β (Entropy) | 0.01 → 0.1 | Mehr Exploration, langsamere Konvergenz |
| γ (Diskont) | 0.99 → 0.9 | Kurzfristiger Fokus, anders Policy |
| λ (GAE) | 0.95 → 0.5 | Mehr Bias, weniger Varianz |
| Epochen | 10 → 20 | Mehr Dateneffizienz, evtl. Overfitting |

---

## Entscheidungsbaum: Welche Methode wähle ich wann?

```
                    ┌──────────────┐
                    │ Wie ist der  │
                    │ Aktionsraum? │
                    └──────┬───────┘
                           │
          ┌────────────────┴──────────────────┐
          ▼                                   ▼
    Diskret (z.B. Atari)           Kontinuierlich (z.B. Roboter)
          │                                   │
          ▼                                   ▼
    Datenmenge groß?           Dateneffizienz wichtig?
     │           │                  │              │
    Ja          Nein               Ja             Nein
     │           │                  │              │
     ▼           ▼                  ▼              ▼
    PPO       REINFORCE            SAC         PPO / TD3
  (Standard)  (Lernen)          (Off-Policy)  (Standard)

Sicherheitskritisch?
→ TRPO / CPO (Constrained Policy Optimization)

Sehr wenig Daten?
→ Model-Based RL (nicht Policy Gradient)
```

---

## Policy Gradient in 5 Sätzen

1. Wir parametrisieren die Policy direkt als neuronales Netz $$π_θ$$ und optimieren θ mit Gradientenaufstieg.
2. Der Policy-Gradient-Theorem gibt uns den Gradienten ohne Ableitung durch die Umgebung: $$∇J(θ) = E[∇ log π_θ(a|s) · Q(s,a)]$$
3. REINFORCE verwendet Monte-Carlo-Returns – unbiased, aber mit hoher Varianz.
4. Actor-Critic reduziert Varianz durch einen Critic V(s) und ermöglicht Online-Updates via Advantage $$A = r + γV(s') - V(s)$$
5. PPO ist der heutige Standard: Es sichert stabile Updates durch Clipping des Probability-Ratios und ist einfach zu implementieren.

---

## Vergleichstabelle: REINFORCE vs. A2C vs. PPO vs. SAC

| Eigenschaft | REINFORCE | A2C | PPO | SAC |
|-------------|-----------|-----|-----|-----|
| Policy-Typ | Stochastisch | Stochastisch | Stochastisch | Stochastisch |
| Update | Episode | Schrittweise | Batch | Off-Policy |
| Varianz | Hoch | Mittel | Niedrig | Niedrig |
| Bias | Kein | Gering | Gering | Gering |
| Dateneffizienz | Sehr gering | Mittel | Mittel | Hoch |
| Stabilität | Gering | Mittel | Hoch | Hoch |
| Aktionsraum | Diskret/Kont. | Diskret/Kont. | Diskret/Kont. | Kontinuierlich |
| Komplexität | Sehr einfach | Einfach | Moderat | Komplex |
| Stan. Env. | CartPole | Atari | MuJoCo | MuJoCo |

---

## Cheat-Sheet: Formeln, Losses, Tricks

### Kernformeln

Policy Gradient Theorem:
  $$∇_θ J(θ) = E[ ∇_θ log π_θ(a|s) · Q^π(s,a) ]$$

REINFORCE Update:
  $$θ ← θ + α · Σ_t ∇_θ log π_θ(a_t|s_t) · G_t$$

Advantage:
  $$A(s,a) = Q(s,a) - V(s)  ≈  r + γV(s') - V(s)$$

GAE:
  $$Â_t = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}$$
  $$δ_t = r_t + γV(s_{t+1}) - V(s_t)$$

PPO Clipped Objective:
  $$L^CLIP = E[ min( r_t·A_t, clip(r_t, 1-ε, 1+ε)·A_t ) ]$$
  $$r_t = π_θ(a|s) / π_θ_old(a|s)$$

SAC Objective:
  $$J(π) = E[Σ γ^t (r_t + α·H(π(·|s_t)))]$$

### Loss-Komponenten (PPO)

$$L_total = L_policy + c₁·L_value - c₂·H(π)$$

$$L_policy = -L^CLIP$$
$$L_value  = (V_θ(s) - V_target)²$$
$$H(π)     = -E[log π(a|s)]$$    ← Entropie-Bonus

Typische Gewichte: c₁ = 0.5, c₂ = 0.01

### Wichtigste Tricks

```
1. Advantage normalisieren:    $$A_norm = (A - μ) / (σ + ε)$$
2. Gradient Clipping:          $$‖∇θ‖ ≤ 0.5$$
3. Observation normalized:     $$s_norm = (s - μ_s) / σ_s$$
4. Tanh-Squashing:             $$a = tanh(raw_a)$$
5. Separate Lernraten:         $$lr_actor ≠ lr_critic$$
6. Entropie überwachen:        H(π) sollte langsam fallen
7. KL-Divergenz überwachen:    KL > 0.02 → zu großes Update
```

---

## Lernkontrolle: Abschluss-Quiz

**Frage 1:** Welches Problem löst Policy Gradient, das DQN hat?  
> *Antwort:* Kontinuierliche Aktionsräume, differenzierbare End-to-End-Optimierung der Policy, native stochastische Policies.

**Frage 2:** Was ist der Log-Likelihood-Trick und wozu wird er benötigt?  
> *Antwort:* Er erlaubt es, `∇P(x;θ)` als `P(x;θ) · ∇ log P(x;θ)` umzuschreiben, sodass der Gradient als Erwartungswert formuliert werden kann und die Umgebungsdynamik herausfällt.

**Frage 3:** Warum hat REINFORCE hohe Varianz und Actor-Critic niedrigere?  
> *Antwort:* REINFORCE verwendet vollständige Monte-Carlo-Returns (Summe vieler zufälliger Rewards). Actor-Critic bootstrappt mit V(s') – ein Ein-Schritt-Fehler statt einer langen Summe.

**Frage 4:** Was klemmt PPO genau, und warum?  
> *Antwort:* Das Probability-Ratio r_t = π_new/π_old wird auf [1-ε, 1+ε] geclippt. Damit wird verhindert, dass ein Update die Policy zu weit vom alten Verhalten entfernt – ähnlich wie ein Sicherheitsgürtel für den Optimierungsschritt.

**Frage 5:** Was hat SAC, das PPO nicht hat?  
> *Antwort:* Entropy-Regularisierung als expliziter Bestandteil des Zielfunktionals, Off-Policy-Training mit Replay Buffer, und automatische Temperaturanpassung α.

**Frage 6:** Du siehst in deinem Training: KL-Divergenz = 0.15 (erwartet: ~0.01). Was tust du?  
> *Antwort:* Das Update war zu groß. Lösungen: ε-Clip-Wert verkleinern, Lernrate reduzieren, Epochi pro Update reduzieren.

**Frage 7:** Describe den Vorteil von GAE gegenüber reinem TD(0) oder reinem Monte-Carlo.  
> *Antwort:* GAE interpoliert mit λ zwischen beiden. λ nahe 0 gibt TD(0) (niedrige Varianz, hoher Bias), λ=1 gibt Monte-Carlo (kein Bias, hohe Varianz). λ≈0.95 liefert in der Praxis eine bessere Balance.

---

## Ressourcen & Weiterführendes

| Thema | Empfehlung |
|-------|-----------|
| Policy Gradient Theorem | Sutton & Barto, Kapitel 13 |
| REINFORCE Original | Williams (1992) |
| Actor-Critic Überblick | Mnih et al. (2016), A3C |
| PPO Paper | Schulman et al. (2017) |
| TRPO Paper | Schulman et al. (2015) |
| SAC Paper | Haarnoja et al. (2018) |
| Implementierung | Stable Baselines3, CleanRL |
| Umgebungen | OpenAI Gym, MuJoCo, dm_control |

---

*Ende Lektion 4 – Policy Gradient*  
*Nächste Lektion: Model-Based Reinforcement Learning*
