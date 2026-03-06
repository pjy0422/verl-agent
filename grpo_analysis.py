from fpdf import FPDF
import os

class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(40, 60, 100)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, "Multiturn GRPO vs Regular GRPO - Training Logic Analysis", fill=True, ln=True, align="C")
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_fill_color(220, 230, 245)
        self.set_text_color(20, 40, 80)
        self.cell(0, 9, title, fill=True, ln=True)
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def subsection_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(40, 80, 140)
        self.cell(0, 7, title, ln=True)
        self.set_text_color(0, 0, 0)

    def body_text(self, text, indent=0):
        self.set_font("Helvetica", "", 10)
        self.set_x(self.l_margin + indent)
        self.multi_cell(0, 6, text)
        self.ln(1)

    def code_block(self, code):
        self.set_font("Courier", "", 8.5)
        self.set_fill_color(245, 245, 245)
        self.set_draw_color(180, 180, 180)
        lines = code.split("\n")
        for line in lines:
            self.set_x(self.l_margin + 4)
            self.cell(0, 5.5, line, fill=True, border=0, ln=True)
        self.ln(2)
        self.set_font("Helvetica", "", 10)

    def table_header(self, cols, widths):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(60, 100, 160)
        self.set_text_color(255, 255, 255)
        for col, w in zip(cols, widths):
            self.cell(w, 7, col, border=1, fill=True, align="C")
        self.ln()
        self.set_text_color(0, 0, 0)

    def table_row(self, cells, widths, fill=False):
        self.set_font("Helvetica", "", 9)
        if fill:
            self.set_fill_color(235, 242, 255)
        else:
            self.set_fill_color(255, 255, 255)
        for cell, w in zip(cells, widths):
            self.multi_cell(w, 6, cell, border=1, fill=True, align="L", max_line_height=6)
            # reset position after multi_cell
        self.ln()


pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# ── TITLE ──────────────────────────────────────────────────────────────
pdf.set_font("Helvetica", "B", 16)
pdf.set_text_color(20, 40, 80)
pdf.cell(0, 12, "Multiturn GRPO vs Regular GRPO", ln=True, align="C")
pdf.set_font("Helvetica", "", 11)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 7, "Training Logic Equivalence Analysis when max_turn = 1", ln=True, align="C")
pdf.ln(6)
pdf.set_text_color(0, 0, 0)

# ── CONCLUSION BOX ────────────────────────────────────────────────────
pdf.set_fill_color(210, 240, 210)
pdf.set_draw_color(60, 160, 60)
pdf.set_font("Helvetica", "B", 11)
pdf.cell(0, 8, "  Conclusion", fill=True, border="LT", ln=True)
pdf.set_font("Helvetica", "", 10)
pdf.set_fill_color(235, 255, 235)
pdf.multi_cell(0, 6,
    "  When max_turn=1 and lambda_div=0.0, the Multiturn GRPO training logic is "
    "functionally equivalent to Regular GRPO. The only remaining difference is "
    "the epsilon value used in std normalization (1e-6 vs 1e-8), which has "
    "negligible impact on training.",
    fill=True, border="LB")
pdf.ln(6)

# ── SECTION 1: Overview ───────────────────────────────────────────────
pdf.section_title("1. Overview")
pdf.body_text(
    "This analysis compares the training logic of the Multiturn GRPO trainer "
    "(configured via examples/multiturn_grpo_trainer/run_mtjailbreak.sh) with "
    "the Regular GRPO trainer (examples/grpo_trainer/) when max_turn is set to 1. "
    "Both trainers share the same RayPPOTrainer.fit() loop in "
    "verl/trainer/ppo/ray_trainer.py."
)

# ── SECTION 2: Rollout Loop ───────────────────────────────────────────
pdf.section_title("2. Rollout Loop  (Identical)")
pdf.body_text(
    "Both trainers use TrajectoryCollector.multi_turn_loop() from "
    "agent_system/multi_turn_rollout/rollout_loop.py (line 715), which internally "
    "calls vanilla_multi_turn_loop(). When env.max_steps=1, the loop executes "
    "exactly one iteration, making the environment interaction flow identical."
)
pdf.code_block(
    "# rollout_loop.py:398\n"
    "for _step in range(self.config.env.max_steps):  # range(1) when max_turn=1\n"
    "    batch = self.preprocess_batch(gen_batch, obs)\n"
    "    batch_output = actor_rollout_wg.generate_sequences(batch_input)\n"
    "    next_obs, rewards, dones, infos = envs.step(text_actions)\n"
    "    turn_rewards_all[i].append(float(rewards_np[i]))  # 1 entry"
)

# ── SECTION 3: Reward Source ──────────────────────────────────────────
pdf.section_title("3. Reward Source  (Identical)")
pdf.body_text(
    "EpisodeRewardManager (agent_system/reward_manager/episode.py) places "
    "episode_rewards as a scalar at the last valid response token:"
)
pdf.code_block(
    "# episode.py:72\n"
    "episode_rewards = data_item.non_tensor_batch['episode_rewards']\n"
    "score = episode_rewards  # normalize_by_length=False\n"
    "reward_tensor[i, valid_response_length - 1] = score"
)
pdf.body_text(
    "With max_turn=1:  episode_rewards = sum of 1 turn = turn_rewards[0]\n"
    "  Regular GRPO  : token_level_rewards.sum(dim=-1) = episode_rewards\n"
    "  Multiturn GRPO: turn_rewards = [single_turn_reward] = episode_rewards\n"
    "=> Both use the same reward value."
)

# ── SECTION 4: Advantage Computation ─────────────────────────────────
pdf.section_title("4. Advantage Computation  (Functionally Equivalent)")

pdf.subsection_title("4.1  Regular GRPO  --  compute_grpo_outcome_advantage()")
pdf.code_block(
    "# core_algos.py:119\n"
    "scores = token_level_rewards.sum(dim=-1)        # episode reward\n"
    "id2mean[idx] = torch.mean(torch.tensor(scores)) # group mean\n"
    "id2std[idx]  = torch.std(torch.tensor(scores))  # group std\n"
    "advantage    = (score - mean) / (std + 1e-6)    # epsilon = 1e-6\n"
    "output       = advantage.unsqueeze(-1) * response_mask"
)

pdf.subsection_title("4.2  Multiturn GRPO  --  compute_multiturn_grpo_advantage()  (max_turn=1)")
pdf.code_block(
    "# core_algos.py:185\n"
    "# Step 1: Diversity  (lambda_div=0.0 => r_hat = reward)\n"
    "r_hat = [[r_i + 0.0 * 1.0]]  =>  r_hat = [[r_i]]\n"
    "\n"
    "# Step 2: Group normalization (same as Regular GRPO scope)\n"
    "flat = [r_0, r_1, r_2, r_3]  (group_size=4)\n"
    "mean = flat.mean() ;  std = flat.std()\n"
    "A[i] = [(r_i - mean) / (std + 1e-8)]    # epsilon = 1e-8\n"
    "\n"
    "# Step 3: Temporal credit  (T=1 => A_tilde = A, no discounting)\n"
    "A_tilde[i] = [A[i][0]]\n"
    "\n"
    "# Step 4: Token mapping  (step_index=0 => uniform broadcast)\n"
    "token_advantages[i, :] = A_tilde[i][0]\n"
    "output = token_advantages * response_mask"
)

pdf.subsection_title("4.3  Why lambda_div does not matter")
pdf.body_text(
    "The diversity function currently always returns 1.0 (hardcoded), so:\n"
    "  r_hat[i] = r_i + lambda_div * 1.0\n"
    "  mean_r_hat = mean_r + lambda_div\n"
    "  advantage = (r_i + lambda_div - mean_r - lambda_div) / std\n"
    "             = (r_i - mean_r) / std    <-- lambda_div cancels out\n"
    "So the advantage is unchanged regardless of lambda_div."
)

# ── SECTION 5: Comparison Table ──────────────────────────────────────
pdf.section_title("5. Side-by-Side Comparison")

cols   = ["Component", "Regular GRPO", "Multiturn GRPO (max_turn=1)", "Same?"]
widths = [40, 48, 72, 18]

pdf.table_header(cols, widths)

rows = [
    ("Rollout loop",       "multi_turn_loop", "multi_turn_loop", "YES"),
    ("Loop iterations",    "max_steps",       "max_steps = 1",   "YES"),
    ("Env interaction",    "envs.step()",      "envs.step()",     "YES"),
    ("Reward source",      "episode_rewards", "turn_rewards[0]", "YES"),
    ("Group normalization","uid grouping",     "uid grouping",    "YES"),
    ("Norm scope",         "4 episodes",      "4 x 1 turn",      "YES"),
    ("Temporal discount",  "N/A",             "no-op (T=1)",     "YES"),
    ("Token mapping",      "uniform * mask",  "uniform * mask",  "YES"),
    ("Epsilon",            "1e-6",            "1e-8",            "minor"),
    ("loss_mask",          "not used",        "not used*",       "YES"),
    ("Extra metadata",     "none",            "turn_rewards etc","no effect"),
    ("Actor update",       "identical",       "identical",       "YES"),
]

for i, row in enumerate(rows):
    fill = (i % 2 == 0)
    # Use a simpler row approach
    pdf.set_font("Helvetica", "", 9)
    if fill:
        pdf.set_fill_color(235, 242, 255)
    else:
        pdf.set_fill_color(255, 255, 255)
    # Color the "Same?" column
    component, rg, mt, same = row
    same_color = (0, 140, 0) if same == "YES" else (180, 100, 0) if same == "minor" else (0, 0, 0)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(widths[0], 6, component, border=1, fill=fill)
    pdf.cell(widths[1], 6, rg,        border=1, fill=fill)
    pdf.cell(widths[2], 6, mt,        border=1, fill=fill)
    pdf.set_text_color(*same_color)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(widths[3], 6, same,      border=1, fill=fill, ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 9)

pdf.ln(3)
pdf.set_font("Helvetica", "I", 8.5)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 5, "* rollout.multi_turn.enable=False in mtjailbreak.yaml => loss_mask not generated", ln=True)
pdf.set_text_color(0, 0, 0)

# ── SECTION 6: Key File References ───────────────────────────────────
pdf.section_title("6. Key File References")

refs = [
    ("RayPPOTrainer.fit()",           "verl/trainer/ppo/ray_trainer.py", "1512"),
    ("compute_advantage()",            "verl/trainer/ppo/ray_trainer.py", "495"),
    ("compute_grpo_outcome_advantage()","verl/trainer/ppo/core_algos.py", "119"),
    ("compute_multiturn_grpo_advantage()","verl/trainer/ppo/core_algos.py","185"),
    ("TrajectoryCollector.multi_turn_loop()", "agent_system/multi_turn_rollout/rollout_loop.py","715"),
    ("vanilla_multi_turn_loop()",      "agent_system/multi_turn_rollout/rollout_loop.py","333"),
    ("EpisodeRewardManager",           "agent_system/reward_manager/episode.py","20"),
    ("mtjailbreak.yaml config",        "verl/trainer/config/mtjailbreak.yaml","1"),
    ("run_mtjailbreak.sh",             "examples/multiturn_grpo_trainer/run_mtjailbreak.sh","1"),
]

pdf.set_font("Helvetica", "", 9.5)
for name, path, line in refs:
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.cell(60, 6, name, border=0)
    pdf.set_font("Courier", "", 8.5)
    pdf.set_text_color(30, 80, 160)
    pdf.cell(0, 6, f"{path}:{line}", ln=True)
    pdf.set_text_color(0, 0, 0)

pdf.ln(4)

# ── SECTION 7: Config Diff ────────────────────────────────────────────
pdf.section_title("7. Configuration Differences (run_mtjailbreak.sh vs run_alfworld.sh)")
pdf.body_text("These are config differences that do NOT affect equivalence when max_turn=1:")

config_diffs = [
    ("algorithm.adv_estimator", "grpo", "multiturn_grpo",
     "Different code path but equivalent result at max_turn=1"),
    ("env.max_steps",           "50",   "1 (hypothetical)",
     "Loop runs once in both cases"),
    ("env.env_name",            "alfworld", "multiturn_conv",
     "Same env interface (reset/step)"),
    ("algorithm.lambda_div",    "N/A",  "0.0",
     "Diversity bonus = 0, cancels out anyway"),
    ("custom_cls (dataset)",    "default", "JailbreakDataset",
     "Data format only, not training logic"),
]

cols2   = ["Config Key", "Regular GRPO", "Multiturn GRPO", "Note"]
widths2 = [45, 28, 32, 73]
pdf.table_header(cols2, widths2)
for i, (k, v1, v2, note) in enumerate(config_diffs):
    fill = (i % 2 == 0)
    if fill:
        pdf.set_fill_color(235, 242, 255)
    else:
        pdf.set_fill_color(255, 255, 255)
    pdf.set_font("Courier", "", 8.5)
    pdf.cell(widths2[0], 6, k,    border=1, fill=fill)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(widths2[1], 6, v1,   border=1, fill=fill)
    pdf.cell(widths2[2], 6, v2,   border=1, fill=fill)
    pdf.cell(widths2[3], 6, note, border=1, fill=fill, ln=True)

pdf.ln(6)

# ── FINAL SUMMARY BOX ─────────────────────────────────────────────────
pdf.set_fill_color(255, 245, 210)
pdf.set_draw_color(200, 140, 0)
pdf.set_font("Helvetica", "B", 11)
pdf.cell(0, 8, "  Final Summary", fill=True, border="LT", ln=True)
pdf.set_font("Helvetica", "", 10)
pdf.set_fill_color(255, 252, 230)
pdf.multi_cell(0, 6,
    "  Given max_turn=1 and lambda_div=0.0:\n"
    "  1. Rollout:       Identical (same TrajectoryCollector, 1 env step)\n"
    "  2. Rewards:       Identical (env step reward = episode_rewards)\n"
    "  3. Group norm:    Identical (same group, same reward values)\n"
    "  4. Token mapping: Identical (uniform broadcast over response tokens)\n"
    "  5. Discounting:   No-op (T=1, no future steps to discount)\n"
    "  6. Diversity:     Cancels out (constant bonus, absorbed by mean subtraction)\n"
    "  7. Actor update:  Identical\n\n"
    "  Only difference: epsilon = 1e-6 (GRPO) vs 1e-8 (Multiturn GRPO)\n"
    "  => Negligible numerical difference, functionally equivalent.",
    fill=True, border="LB"
)

# ── SAVE ──────────────────────────────────────────────────────────────
out = "/home2/pjy0422/workspace/verl-agent/grpo_equivalence_analysis.pdf"
pdf.output(out)
print(f"PDF saved: {out}")
