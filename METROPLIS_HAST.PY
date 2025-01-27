import numpy as np
import matplotlib.pyplot as plt

# Funzione di distribuzione target (pi(x) ∝ e^(-x^2))
def target_distribution(x):
    return np.exp(-x**2)

# Funzione di proposta (random walk, distribuzione normale centrata in x)
def proposal_distribution(x, sigma=2):
    return np.random.normal(x, sigma)

# Algoritmo di Metropolis-Hastings
def metropolis_hastings(target_distribution, proposal_distribution, num_samples, initial_state=0, sigma=1.0):
    samples_acepted = []
    samples_rejected =[]
    current_state = initial_state
    
    for _ in range(num_samples):
        # Proponi un nuovo stato
        proposed_state = proposal_distribution(current_state, sigma)
        
        # Calcola la probabilità di accettazione
        acceptance_ratio = min(1, target_distribution(proposed_state) / target_distribution(current_state))
        
        # Accetta o rifiuta il nuovo stato
        if np.random.rand() < acceptance_ratio:
            current_state = proposed_state
            samples_acepted.append(current_state)
        else:
            samples_rejected.append(proposed_state)
        # Aggiungi il campione corrente alla lista
   
    
    return np.array(samples_acepted), np.array(samples_rejected)

# Parametri
num_samples = 10000
initial_state = 0  # Stato iniziale
sigma = 1.0  # Deviazione standard della proposta

# Genera campioni usando Metropolis-Hastings
samples_acepted,samples_rejected = metropolis_hastings(target_distribution, proposal_distribution, num_samples, initial_state, sigma)

# Plot dei risultati
x_vals = np.linspace(-5, 5, 1000)
y_vals = target_distribution(x_vals) / np.sum(target_distribution(x_vals))  # Normalizziamo per la densità

plt.figure(figsize=(10, 6))
plt.hist(samples_acepted, bins=50, density=True, alpha=0.6, color='g', label='Campioni da MH')
plt.plot(x_vals, y_vals*100, label='Distribuzione target', color='r', lw=2)
plt.legend()
plt.title('Esempio di campionamento con Metropolis-Hastings')
plt.xlabel('x')
plt.ylabel('Densità di probabilità')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(samples_rejected, 'rx', label='Rejected',alpha=0.5)
plt.plot(samples_acepted, 'b.', label='Accepted',alpha=0.5)
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("y")
plt.title("Figure 2: MCMC sampling for $\sigma$ with Metropolis-Hastings. First 50 samples are shown.")
plt.show()