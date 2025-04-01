import csv
import dateutil.parser
import numpy as np
import dateutil
# import statistics
# import matplotlib.pyplot as plt
import scipy.stats as stats

if __name__ == "__main__":
  
  
  dates = []
  closes = []
  
  with open("../ass4/MSFT-1.csv", 'r') as file:
    reader = csv.reader(file)
    
    next(reader)
    
    for row in reader:
      dates.append(dateutil.parser.parse(row[0],dateutil.parser.parserinfo(dayfirst=False, yearfirst=False)).timestamp())
      closes.append(float(row[1]))
      
  dates = np.asarray(dates)
  closes = np.asarray(closes)
  r_i = np.log(closes[1:] / closes[:-1])
  
  
  # 3.a
  bounds = np.asarray([float('-inf'), -0.001, -0.0004, 0, 0.0004, 0.001, float('inf')])
  
  # observed counts
  expanded_r_i = np.repeat(np.expand_dims(r_i, axis = 1), bounds.shape[0] - 1, axis = 1)
  lower_bounds = expanded_r_i >= bounds[:-1]
  upper_bounds = expanded_r_i < bounds[1:]
  categorised_r_i = lower_bounds & upper_bounds
  observed_counts = np.count_nonzero(categorised_r_i, axis = 0)
  
  # expected counts
  cdfs = stats.norm.cdf(bounds, loc = 0, scale = 0.02)
  total_counts = r_i.shape[0]
  expected_counts = (cdfs[1:] - cdfs[:-1]) * total_counts
  
  # chi square test
  chi_sq_statistic = np.sum(((observed_counts - expected_counts) ** 2) / expected_counts)
  chi_sq_p_value = 1 - stats.chi2.cdf(chi_sq_statistic, df = observed_counts.shape[0] - 1)
  
  print(f"3.a observed counts: {observed_counts}\n    expected counts: {expected_counts}")
  print(f"    chi sq statistic: {chi_sq_statistic:.3f} with p value {chi_sq_p_value}")
  
  
  # 3.b
  r_i_mle_mean = np.mean(r_i)
  r_i_mle_std_dev = np.sqrt(np.var(r_i))
  print(f"\nr_i mle mean: {r_i_mle_mean:.6f}, r_i mle std dev: {r_i_mle_std_dev:.6f}")
  
  est_cdfs = stats.norm.cdf(bounds, loc = r_i_mle_mean, scale = r_i_mle_std_dev)
  est_expected_counts = (est_cdfs[1:] - est_cdfs[:-1]) * total_counts
  
  # chi square test
  est_chi_sq_statistic = np.sum(((observed_counts - est_expected_counts) ** 2) / est_expected_counts)
  est_chi_sq_p_value = 1 - stats.chi2.cdf(est_chi_sq_statistic, df = observed_counts.shape[0] - 1 - 2) # minus dof for est params
  
  print(f"3.a observed counts: {observed_counts}\nest expected counts: {est_expected_counts}")
  print(f"    chi sq statistic: {est_chi_sq_statistic:.3f} with p value {est_chi_sq_p_value}")