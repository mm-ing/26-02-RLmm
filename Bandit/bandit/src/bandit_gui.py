import tkinter as tk
from tkinter import Label, Entry, Button, StringVar, IntVar, OptionMenu, Spinbox, Frame, LabelFrame, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from bandit_logic import OpenArmedBandit, Agent

# One colour per method series kept in the plot
METHOD_COLORS = {
    "Epsilon Greedy": "steelblue",
    "Thompson Sampling": "darkorange",
}


class BanditGUI:
    def __init__(self, master):
        self.master = master
        master.title("Multi-Armed Bandit Experiment")

        # --- tk variables ---
        self.agent_loops_var = IntVar(value=100)
        self.agent_memory_var = IntVar(value=0)
        self.epsilon_var = StringVar(value="0.1")
        self.decay_var = StringVar(value="1.0")
        self.method_var = StringVar(value="Epsilon Greedy")

        # --- bandits (fixed across sessions) ---
        self.bandits = [
            OpenArmedBandit(0.2),
            OpenArmedBandit(0.4),
            OpenArmedBandit(0.8),
        ]

        # --- per-method reward history for the plot ---
        # {method_name: [reward, reward, ...]}
        self.method_rewards: dict[str, list] = {}
        # rewards for the *current* agent session
        self.current_rewards: list[int] = []

        # guard: suppress method-change side-effects during reset
        self._resetting: bool = False

        # --- agent ---
        self.agent = self._make_agent()

        # --- build UI ---
        self._create_widgets()

        # watch method changes -> recreate agent
        self.method_var.trace_add("write", self._on_method_change)

        self._update_display()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _make_agent(self) -> Agent:
        return Agent(
            n_arms=3,
            epsilon=float(self.epsilon_var.get()),
            decay=float(self.decay_var.get()),
            memory=self.agent_memory_var.get(),
            method=self.method_var.get(),
        )

    def _on_method_change(self, *_):
        """Keep current plot series, recreate agent for the new method."""
        if self._resetting:
            return
        from bandit_logic import EpsilonGreedyPolicy
        old_method = (
            "Epsilon Greedy"
            if isinstance(self.agent.policy, EpsilonGreedyPolicy)
            else "Thompson Sampling"
        )
        if self.current_rewards:
            self.method_rewards.setdefault(old_method, []).extend(self.current_rewards)
        self.current_rewards = []
        self.agent = self._make_agent()
        self._update_display()

    def _record(self, reward: int):
        self.current_rewards.append(reward)

    # ------------------------------------------------------------------
    # widget construction
    # ------------------------------------------------------------------

    def _create_widgets(self):
        pad = {"padx": 6, "pady": 3}

        # ---- Controls frame ----
        ctrl = LabelFrame(self.master, text="Controls", padx=6, pady=6)
        ctrl.grid(row=0, column=0, sticky="nsew", **pad)

        Label(ctrl, text="Agent loops (n):").grid(row=0, column=0, sticky="w")
        Entry(ctrl, textvariable=self.agent_loops_var, width=8).grid(row=0, column=1, sticky="w")

        Label(ctrl, text="Agent memory (last n pulls):").grid(row=1, column=0, sticky="w")
        Entry(ctrl, textvariable=self.agent_memory_var, width=8).grid(row=1, column=1, sticky="w")

        Label(ctrl, text="Epsilon:").grid(row=2, column=0, sticky="w")
        Spinbox(ctrl, from_=0.0, to=1.0, increment=0.01,
                textvariable=self.epsilon_var, width=8).grid(row=2, column=1, sticky="w")

        Label(ctrl, text="Decay:").grid(row=3, column=0, sticky="w")
        Spinbox(ctrl, from_=0.0, to=1.0, increment=0.01,
                textvariable=self.decay_var, width=8).grid(row=3, column=1, sticky="w")

        Label(ctrl, text="Method:").grid(row=4, column=0, sticky="w")
        OptionMenu(ctrl, self.method_var, "Epsilon Greedy", "Thompson Sampling").grid(
            row=4, column=1, sticky="w")
        Button(ctrl, text="Save Plot", bg="#a8d8a8", activebackground="#5cb85c",
               command=self._save_plot).grid(row=4, column=2, padx=(8, 0), sticky="w")

        # manual pull buttons
        btn_frame = Frame(ctrl)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=(8, 2))
        for i in range(3):
            Button(btn_frame, text=f"Pull Bandit {i+1}",
                   command=lambda idx=i: self._pull_bandit(idx)).pack(side="left", padx=2)

        # agent action buttons
        agent_frame = Frame(ctrl)
        agent_frame.grid(row=6, column=0, columnspan=2, pady=(4, 2), sticky="w")
        Button(agent_frame, text="Agent: single step",
               command=self._agent_single_step).pack(side="left", padx=2)
        Button(agent_frame, text="Agent: run n loops",
               command=self._run_agent_loops).pack(side="left", padx=2)

        # reset / save — placed on master grid after canvas so always visible
        # (added at end of _create_widgets after canvas)

        # ---- Current state frame ----
        state = LabelFrame(self.master, text="Current State", padx=6, pady=6)
        state.grid(row=0, column=1, sticky="nsew", **pad)

        headers = ["Bandit", "True p", "Pulls", "Rewards", "Rate"]
        for col, h in enumerate(headers):
            Label(state, text=h, font=("TkDefaultFont", 9, "bold")).grid(
                row=0, column=col, padx=4)

        self._state_labels = []
        for i in range(3):
            row_labels = []
            Label(state, text=f"Bandit {i+1}").grid(row=i+1, column=0, padx=4, sticky="w")
            Label(state, text=f"{self.bandits[i].reward_prob:.2f}").grid(
                row=i+1, column=1, padx=4)
            for col in range(2, 5):
                lbl = Label(state, text="0", width=7)
                lbl.grid(row=i+1, column=col, padx=4)
                row_labels.append(lbl)
            self._state_labels.append(row_labels)  # [pulls_lbl, rewards_lbl, rate_lbl]

        self._total_label = Label(state, text="Total pulls: 0",
                                  font=("TkDefaultFont", 9, "bold"))
        self._total_label.grid(row=4, column=0, columnspan=5, pady=(6, 0), sticky="w")

        # ---- Plot frame ----
        self._figure = plt.Figure(figsize=(7, 4), dpi=100)
        self._ax = self._figure.add_subplot(111)
        self._canvas = FigureCanvasTkAgg(self._figure, master=self.master)
        self._canvas.get_tk_widget().grid(row=1, column=0, columnspan=2, **pad)

        # ---- Action bar below plot — always visible ----
        action_bar = Frame(self.master)
        action_bar.grid(row=2, column=0, columnspan=2, pady=(2, 6), sticky="ew", padx=6)
        Button(action_bar, text="Reset", bg="#f28b82", activebackground="#ea4335",
               command=self._reset).pack(side="left", padx=4)
        Button(action_bar, text="Save Plot", bg="#a8d8a8", activebackground="#5cb85c",
               command=self._save_plot).pack(side="left", padx=4)

        # ensure window grows to fit all rows and columns
        self.master.columnconfigure(0, weight=1)
        self.master.columnconfigure(1, weight=1)
        self.master.rowconfigure(0, weight=0)
        self.master.rowconfigure(1, weight=1)
        self.master.rowconfigure(2, weight=0)
        self.master.resizable(True, True)
        self.master.update_idletasks()
        self.master.minsize(self.master.winfo_reqwidth(), self.master.winfo_reqheight())

    # ------------------------------------------------------------------
    # actions
    # ------------------------------------------------------------------

    def _pull_bandit(self, arm_index: int):
        reward = self.bandits[arm_index].pull()
        self.agent.update(arm_index, reward)
        self._record(reward)
        self._update_display()

    def _agent_single_step(self):
        arm_index = self.agent.select_action()
        reward = self.bandits[arm_index].pull()
        self.agent.update(arm_index, reward)
        self._record(reward)
        self._update_display()

    def _run_agent_loops(self):
        n = self.agent_loops_var.get()
        rewards = self.agent.run(self.bandits, n)
        self.current_rewards.extend(rewards)
        self._update_display()

    def _reset(self):
        """Reset agent, current rewards and plot — start fresh."""
        self._resetting = True
        self.method_rewards = {}
        self.current_rewards = []
        self.agent = self._make_agent()
        self._resetting = False
        self._update_display()

    def _save_plot(self):
        """Open a save-file dialog and write the current plot to an image file."""
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG image", "*.png"),
                ("JPEG image", "*.jpg"),
                ("SVG vector", "*.svg"),
                ("PDF document", "*.pdf"),
            ],
            title="Save plot as",
        )
        if path:
            self._figure.savefig(path, bbox_inches="tight")

    # ------------------------------------------------------------------
    # display
    # ------------------------------------------------------------------

    def _update_display(self):
        self._update_state_panel()
        self._update_plot()

    def _update_state_panel(self):
        stats = self.agent.get_stats()
        for i in range(3):
            pulls = stats["pulls"][i]
            rewards = stats["successes"][i]
            rate = stats["success_rates"][i]
            self._state_labels[i][0].config(text=str(pulls))
            self._state_labels[i][1].config(text=str(rewards))
            self._state_labels[i][2].config(text=f"{rate:.3f}")
        self._total_label.config(text=f"Total pulls: {sum(stats['pulls'])}")

    def _update_plot(self):
        self._ax.clear()

        # previously completed method series
        for method, rewards in self.method_rewards.items():
            if rewards:
                cumsum = []
                s = 0
                for r in rewards:
                    s += r
                    cumsum.append(s)
                color = METHOD_COLORS.get(method, "gray")
                self._ax.plot(cumsum, label=f"{method} (prev)",
                              color=color, alpha=0.4, linestyle="--")

        # current session
        if self.current_rewards:
            cumsum = []
            s = 0
            for r in self.current_rewards:
                s += r
                cumsum.append(s)
            current_method = self.method_var.get()
            color = METHOD_COLORS.get(current_method, "gray")
            self._ax.plot(cumsum, label=current_method, color=color)

        self._ax.set_title("Cumulative Reward Over Time")
        self._ax.set_xlabel("Pulls")
        self._ax.set_ylabel("Cumulative Reward")
        if self._ax.lines:
            self._ax.legend()
        self._canvas.draw()


if __name__ == "__main__":
    import tkinter as tk
    root = tk.Tk()
    BanditGUI(root)
    root.mainloop()
