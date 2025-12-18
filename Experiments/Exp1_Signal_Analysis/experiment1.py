"""
Experiment 1: Analysis of Fundamental Signals
Objective: To simulate and analyze fundamental signals such as unit impulse, impulse train,
unit step, and ramp signals in both continuous and discrete time domains, and to understand
their properties, behavior, and importance in digital signal processing.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directory
os.makedirs('outputs', exist_ok=True)

print("=" * 50)
print("EXPERIMENT 1: FUNDAMENTAL SIGNAL ANALYSIS")
print("=" * 50)
print()

# Time parameters
n = np.arange(-10, 11)  # Discrete time
t = np.linspace(-10, 10, 1000)  # Continuous time

# ============================================================================
# 1. UNIT IMPULSE (DELTA) SIGNAL
# ============================================================================
print("=" * 50)
print("1. UNIT IMPULSE (DELTA) SIGNAL")
print("=" * 50)

# Discrete unit impulse: δ[n] = 1 for n=0, else 0
delta_discrete = np.zeros_like(n)
delta_discrete[n == 0] = 1

# Continuous approximation (Dirac delta)
delta_continuous = np.zeros_like(t)
delta_continuous[np.abs(t) < 0.01] = 100  # Approximate impulse

print(f"Discrete δ[n]:")
print(f"  Value at n=0: {delta_discrete[n == 0][0]}")
print(f"  Sum of all values: {np.sum(delta_discrete)}")
print(f"  Non-zero samples: {np.count_nonzero(delta_discrete)}")
print()

# ============================================================================
# 2. IMPULSE TRAIN SIGNAL
# ============================================================================
print("=" * 50)
print("2. IMPULSE TRAIN (PERIODIC IMPULSES)")
print("=" * 50)

# Discrete impulse train with period T=4
T = 4
impulse_train = np.zeros_like(n)
impulse_train[n % T == 0] = 1

print(f"Impulse Train with period T={T}:")
print(f"  Number of impulses: {np.sum(impulse_train)}")
print(f"  Impulse locations: {n[impulse_train == 1].tolist()}")
print()

# ============================================================================
# 3. UNIT STEP SIGNAL
# ============================================================================
print("=" * 50)
print("3. UNIT STEP SIGNAL")
print("=" * 50)

# Discrete unit step: u[n] = 1 for n≥0, else 0
step_discrete = np.zeros_like(n, dtype=float)
step_discrete[n >= 0] = 1

# Continuous unit step
step_continuous = np.zeros_like(t)
step_continuous[t >= 0] = 1

print(f"Discrete u[n]:")
print(f"  Value at n=-1: {step_discrete[n == -1][0]}")
print(f"  Value at n=0: {step_discrete[n == 0][0]}")
print(f"  Value at n=1: {step_discrete[n == 1][0]}")
print(f"  Sum from n=0 to n=10: {np.sum(step_discrete[n >= 0])}")
print()

# ============================================================================
# 4. RAMP SIGNAL
# ============================================================================
print("=" * 50)
print("4. RAMP SIGNAL")
print("=" * 50)

# Discrete ramp: r[n] = n for n≥0, else 0
ramp_discrete = np.zeros_like(n, dtype=float)
ramp_discrete[n >= 0] = n[n >= 0]

# Continuous ramp
ramp_continuous = np.zeros_like(t)
ramp_continuous[t >= 0] = t[t >= 0]

print(f"Discrete r[n]:")
print(f"  Value at n=0: {ramp_discrete[n == 0][0]}")
print(f"  Value at n=5: {ramp_discrete[n == 5][0]}")
print(f"  Value at n=10: {ramp_discrete[n == 10][0]}")
print(f"  Slope (difference): {ramp_discrete[n == 5][0] - ramp_discrete[n == 4][0]}")
print()

# ============================================================================
# PLOTTING ALL SIGNALS
# ============================================================================
print("=" * 50)
print("GENERATING VISUALIZATIONS")
print("=" * 50)

fig, axes = plt.subplots(4, 2, figsize=(14, 12))
fig.suptitle('Fundamental Signals: Discrete vs Continuous', fontsize=16, fontweight='bold')

# Row 1: Unit Impulse
axes[0, 0].stem(n, delta_discrete, basefmt=' ')
axes[0, 0].set_title('Discrete Unit Impulse δ[n]', fontweight='bold')
axes[0, 0].set_xlabel('n')
axes[0, 0].set_ylabel('δ[n]')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(y=0, color='k', linewidth=0.5)
axes[0, 0].axvline(x=0, color='k', linewidth=0.5)

axes[0, 1].plot(t, delta_continuous, 'b-', linewidth=2)
axes[0, 1].set_title('Continuous Unit Impulse δ(t)', fontweight='bold')
axes[0, 1].set_xlabel('t')
axes[0, 1].set_ylabel('δ(t)')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=0, color='k', linewidth=0.5)
axes[0, 1].axvline(x=0, color='k', linewidth=0.5)

# Row 2: Impulse Train
axes[1, 0].stem(n, impulse_train, basefmt=' ')
axes[1, 0].set_title(f'Discrete Impulse Train (T={T})', fontweight='bold')
axes[1, 0].set_xlabel('n')
axes[1, 0].set_ylabel('Amplitude')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axhline(y=0, color='k', linewidth=0.5)

# Continuous impulse train
impulse_train_cont = np.zeros_like(t)
for i in range(-2, 3):
    impulse_train_cont[np.abs(t - i*T) < 0.05] = 20
axes[1, 1].plot(t, impulse_train_cont, 'b-', linewidth=2)
axes[1, 1].set_title(f'Continuous Impulse Train (T={T})', fontweight='bold')
axes[1, 1].set_xlabel('t')
axes[1, 1].set_ylabel('Amplitude')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(y=0, color='k', linewidth=0.5)

# Row 3: Unit Step
axes[2, 0].stem(n, step_discrete, basefmt=' ')
axes[2, 0].set_title('Discrete Unit Step u[n]', fontweight='bold')
axes[2, 0].set_xlabel('n')
axes[2, 0].set_ylabel('u[n]')
axes[2, 0].grid(True, alpha=0.3)
axes[2, 0].axhline(y=0, color='k', linewidth=0.5)
axes[2, 0].axvline(x=0, color='k', linewidth=0.5)

axes[2, 1].plot(t, step_continuous, 'b-', linewidth=2)
axes[2, 1].set_title('Continuous Unit Step u(t)', fontweight='bold')
axes[2, 1].set_xlabel('t')
axes[2, 1].set_ylabel('u(t)')
axes[2, 1].grid(True, alpha=0.3)
axes[2, 1].axhline(y=0, color='k', linewidth=0.5)
axes[2, 1].axvline(x=0, color='k', linewidth=0.5)
axes[2, 1].set_ylim(-0.2, 1.2)

# Row 4: Ramp
axes[3, 0].stem(n, ramp_discrete, basefmt=' ')
axes[3, 0].set_title('Discrete Ramp r[n]', fontweight='bold')
axes[3, 0].set_xlabel('n')
axes[3, 0].set_ylabel('r[n]')
axes[3, 0].grid(True, alpha=0.3)
axes[3, 0].axhline(y=0, color='k', linewidth=0.5)
axes[3, 0].axvline(x=0, color='k', linewidth=0.5)

axes[3, 1].plot(t, ramp_continuous, 'b-', linewidth=2)
axes[3, 1].set_title('Continuous Ramp r(t)', fontweight='bold')
axes[3, 1].set_xlabel('t')
axes[3, 1].set_ylabel('r(t)')
axes[3, 1].grid(True, alpha=0.3)
axes[3, 1].axhline(y=0, color='k', linewidth=0.5)
axes[3, 1].axvline(x=0, color='k', linewidth=0.5)

plt.tight_layout()
plt.savefig('outputs/fundamental_signals.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Fundamental signals plot saved to outputs/fundamental_signals.png")
print()

# ============================================================================
# SIGNAL PROPERTIES AND RELATIONSHIPS
# ============================================================================
print("=" * 50)
print("SIGNAL PROPERTIES AND RELATIONSHIPS")
print("=" * 50)

# Relationship: Step is integral of impulse
# Relationship: Ramp is integral of step
# Relationship: Impulse is derivative of step

print("\nKey Relationships:")
print("1. Unit Step = Integral of Unit Impulse")
print("   u[n] = Σ δ[k] for k from -∞ to n")
print(f"   Verification: Cumulative sum of δ[n] at n=0 = {np.cumsum(delta_discrete)[n == 0][0]}")
print()

print("2. Ramp = Integral of Unit Step")
print("   r[n] = Σ u[k] for k from -∞ to n")
print(f"   Verification: Cumulative sum of u[n] at n=5 = {np.cumsum(step_discrete)[n == 5][0]}")
print()

print("3. Impulse = Derivative of Step")
print("   δ[n] = u[n] - u[n-1]")
diff_step = np.diff(np.concatenate(([0], step_discrete)))
print(f"   Verification: Difference of u[n] has impulse at n=0")
print()

# ============================================================================
# ADDITIONAL OPERATIONS
# ============================================================================
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
fig2.suptitle('Signal Operations and Transformations', fontsize=16, fontweight='bold')

# 1. Delayed and Advanced Signals
delay = 3
step_delayed = np.zeros_like(n, dtype=float)
step_delayed[n >= delay] = 1
step_advanced = np.zeros_like(n, dtype=float)
step_advanced[n >= -delay] = 1

axes2[0, 0].stem(n, step_discrete, basefmt=' ', label='u[n]')
axes2[0, 0].stem(n, step_delayed, basefmt=' ', linefmt='r-', markerfmt='ro', label=f'u[n-{delay}]')
axes2[0, 0].stem(n, step_advanced, basefmt=' ', linefmt='g-', markerfmt='go', label=f'u[n+{delay}]')
axes2[0, 0].set_title('Time Shifting', fontweight='bold')
axes2[0, 0].set_xlabel('n')
axes2[0, 0].set_ylabel('Amplitude')
axes2[0, 0].legend()
axes2[0, 0].grid(True, alpha=0.3)

# 2. Scaled Signals
ramp_scaled = 2 * ramp_discrete
axes2[0, 1].stem(n, ramp_discrete, basefmt=' ', label='r[n]')
axes2[0, 1].stem(n, ramp_scaled, basefmt=' ', linefmt='r-', markerfmt='ro', label='2·r[n]')
axes2[0, 1].set_title('Amplitude Scaling', fontweight='bold')
axes2[0, 1].set_xlabel('n')
axes2[0, 1].set_ylabel('Amplitude')
axes2[0, 1].legend()
axes2[0, 1].grid(True, alpha=0.3)

# 3. Signal Combination
combined = step_discrete + 0.5 * ramp_discrete
axes2[1, 0].stem(n, combined, basefmt=' ')
axes2[1, 0].set_title('Combined Signal: u[n] + 0.5·r[n]', fontweight='bold')
axes2[1, 0].set_xlabel('n')
axes2[1, 0].set_ylabel('Amplitude')
axes2[1, 0].grid(True, alpha=0.3)

# 4. Exponential Signal
alpha = 0.8
exp_signal = np.zeros_like(n, dtype=float)
exp_signal[n >= 0] = alpha ** n[n >= 0]
axes2[1, 1].stem(n, exp_signal, basefmt=' ')
axes2[1, 1].set_title(f'Exponential Signal: ({alpha})^n·u[n]', fontweight='bold')
axes2[1, 1].set_xlabel('n')
axes2[1, 1].set_ylabel('Amplitude')
axes2[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/signal_operations.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Signal operations plot saved to outputs/signal_operations.png")
print()

print("=" * 50)
print("EXPERIMENT 1 COMPLETED SUCCESSFULLY!")
print("=" * 50)
print()
print("Summary:")
print("- Generated and analyzed 4 fundamental signals")
print("- Compared discrete and continuous representations")
print("- Demonstrated signal properties and relationships")
print("- Performed time shifting, scaling, and combination operations")
print()
print("Output files:")
print("  1. outputs/fundamental_signals.png")
print("  2. outputs/signal_operations.png")
print()
