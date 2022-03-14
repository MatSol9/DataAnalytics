from cmdstanpy import CmdStanModel

gen_quant = CmdStanModel(stan_file="code_1.stan")
samples = gen_quant.sample(data={'M':10}, 
                            fixed_param=True, 
                            iter_sampling=1000, 
                            iter_warmup=0, 
                            chains = 1)

print("results:")
print(samples)