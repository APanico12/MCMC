import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Funzioni per calcolare la distribuzione a posteriori per ogni parametro

# Verosimiglianza dei dati
def likelihood(y, theta, sigma2, R, T):
    return np.sum([(y[t, r] - theta[t]) ** 2 for t in range(T) for r in range(R)]) / (2 * sigma2)

# Calcolare la distribuzione a posteriori di sigma^2
def sample_sigma2(y, theta, alpha, beta, R, T):
    residual = np.sum([(y[t, r] - theta[t]) ** 2 for t in range(T) for r in range(R)])
    alpha_post = alpha + T * R / 2
    beta_post = beta + residual / 2
    return np.random.gamma(alpha_post, 1 / beta_post)

# Calcolare la distribuzione a posteriori di sigma^2_theta
def sample_sigma2_theta(theta, alpha_theta, beta_theta, T):
    residual = np.sum([(theta[t] - np.mean(theta)) ** 2 for t in range(T)])
    alpha_post_theta = alpha_theta + T / 2
    beta_post_theta = beta_theta + residual / 2
    return np.random.gamma(alpha_post_theta, 1 / beta_post_theta)

# Calcolare la distribuzione a posteriori di mu
def sample_mu(theta, sigma2_theta, mu0, tau2, T):
    numerator = sigma2_theta * mu0 + tau2 * np.sum(theta)
    denominator = sigma2_theta + T * tau2
    return np.random.normal(numerator / denominator, np.sqrt(sigma2_theta * tau2 / denominator))

# Calcolare la distribuzione a posteriori di theta_t
def sample_theta(y, mu, sigma2, sigma2_theta, R, t):
    posterior_mean = (np.sum(y[t, :] - mu) / sigma2 + mu / sigma2_theta) / (R / sigma2 + 1 / sigma2_theta)
    posterior_var = 1 / (R / sigma2 + 1 / sigma2_theta)
    return np.random.normal(posterior_mean, np.sqrt(posterior_var))

# Funzione di Gibbs sampling
def gibbs_sampling(y, T, R, mu0, tau2, alpha, beta, alpha_theta, beta_theta, num_iterations):
    # Inizializzazione
    theta = np.random.normal(0, 1, T)
    sigma2 = np.random.gamma(1, 1, 1)
    sigma2_theta = np.random.gamma(1, 1, 1)
    mu = np.random.normal(mu0, tau2)
    
    samples = {
        'theta': [],
        'sigma2': [],
        'sigma2_theta': [],
        'mu': []
    }
    
    for i in range(num_iterations):
        # Passo 1: campionare sigma^2
        sigma2 = sample_sigma2(y, theta, alpha, beta, R, T)
        
        # Passo 2: campionare sigma^2_theta
        sigma2_theta = sample_sigma2_theta(theta, alpha_theta, beta_theta, T)
        
        # Passo 3: campionare mu
        mu = sample_mu(theta, sigma2_theta, mu0, tau2, T)
        
        # Passo 4: campionare theta_t
        for t in range(T):
            theta[t] = sample_theta(y, mu, sigma2, sigma2_theta, R, t)
        
        # Memorizzare i campioni
        samples['theta'].append(theta.copy())
        samples['sigma2'].append(sigma2)
        samples['sigma2_theta'].append(sigma2_theta)
        samples['mu'].append(mu)
    
    return samples

# Creazione del DataFrame con i dati dei gruppi
data = {
    'Group1': [68, 42, 69, 64, 39, 66, 29],
    'Group2': [49, 52, 41, 56, 40, 43, 20],
    'Group3': [41, 40, 26, 33, 42, 27, 35],
    'Group4': [33, 27, 48, 54, 42, 56, 19],
    'Group5': [40, 45, 50, 41, 37, 34, 42],
    'Group6': [30, 42, 35, 44, 49, 25, 45]
}
df = pd.DataFrame(data)
y = df.iloc[:, :].values.T  # Trasposizione per ottenere gruppi come righe

# Parametri iniziali
T = y.shape[0]  # Numero di gruppi
R = y.shape[1]  # Numero di osservazioni per gruppo
mu0 = 0  # Media a priori
tau2 = 1000  # Varianza a priori per mu
alpha = 1  # Parametro a priori di sigma^2
beta = 1  # Parametro a priori di sigma^2
alpha_theta = 1  # Parametro a priori di sigma^2_theta
beta_theta = 1  # Parametro a priori di sigma^2_theta

# Esegui Gibbs sampling
num_iterations = 1000
samples = gibbs_sampling(y, T, R, mu0, tau2, alpha, beta, alpha_theta, beta_theta, num_iterations)

# Visualizzare i risultati
theta_samples = np.array(samples['theta'])
mu_samples = np.array(samples['mu'])
sigma2_samples = np.array(samples['sigma2'])
sigma2_theta_samples = np.array(samples['sigma2_theta'])

# Posteriori per theta
plt.figure(figsize=(10, 6))
for t in range(T):
    plt.subplot(3, 2, t+1)
    plt.plot(theta_samples[:, t])
    plt.title(f'Samples for θ_{t+1}')
plt.tight_layout()
plt.show()

# Posteriori per mu
plt.figure(figsize=(10, 6))
plt.plot(mu_samples)
plt.title('Samples for μ')
plt.show()

# Posteriori per sigma^2
plt.figure(figsize=(10, 6))
plt.plot(sigma2_samples)
plt.title('Samples for σ^2')
plt.show()

# Posteriori per sigma^2_theta
plt.figure(figsize=(10, 6))
plt.plot(sigma2_theta_samples)
plt.title('Samples for σ^2_theta')
plt.show()

# Grafico dei valori osservati per i gruppi
plt.figure(figsize=(10, 6))
for t in range(T):
    plt.scatter([t+1] * R, y[t, :], label=f'Group {t+1}')
plt.title('Observed values for each group')
plt.xlabel('Group')
plt.ylabel('y values')
plt.legend()
plt.show()
