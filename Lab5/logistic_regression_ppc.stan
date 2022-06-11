data {
    int<lower=1> N; // Number of observations
    int<lower=1> M; // Number of covariates
    matrix[N, M] X; // Covariate design matrix
    real sigma;     // Prior standard deviation
}

transformed data {
    vector[N] ones_N = rep_vector(1, N);
    vector[M] ones_M = rep_vector(1, M);
}

// Simulate emperical probabilities from the current value of parameters
generated quantities {
   vector[N] prob_pcc;
   real beta[M] = normal_rng(0, ones_M*sigma);  // Prior model
   real alpha = normal_rng(0, sigma);           // Prior model
   prob_pcc = inv_logit(X * to_vector(beta) + ones_N*alpha);
}