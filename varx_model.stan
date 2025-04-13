data {
  int<lower=2> N;                // Number of time points
  int<lower=1> L;                // Number of predictors
  vector[N] Y;                   // Single response variable
  matrix[N, L] X_all;            // Predictor matrix
}

parameters {
  vector[L] B;                  // Coefficients for predictors
  real C_raw;                   // Non-centered intercept
  real<lower=0> sigma_C;        // Scale for intercept
  real<lower=0> sigma;          // Standard deviation
}

transformed parameters {
  real C = sigma_C * C_raw;
}

model {
  // Priors
  sigma_C ~ gamma(1, 1);
  C_raw ~ normal(0, 1);
  B ~ normal(0, 1);
  sigma ~ gamma(1, 1);

  // Likelihood
  for (t in 2:N) {
    Y[t] ~ normal(C + dot_product(B, X_all[t]), sigma);
  }
}

