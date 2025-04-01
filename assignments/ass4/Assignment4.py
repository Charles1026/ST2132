import csv
import dateutil.parser
import numpy as np
import dateutil
import statistics
import matplotlib.pyplot as plt
import scipy.stats as stats

if __name__ == "__main__":
  
  
  dates = []
  closes = []
  
  with open("./MSFT-1.csv", 'r') as file:
    reader = csv.reader(file)
    
    next(reader)
    
    for row in reader:
      dates.append(dateutil.parser.parse(row[0],dateutil.parser.parserinfo(dayfirst=False, yearfirst=False)).timestamp())
      closes.append(float(row[1]))
      
  dates = np.asarray(dates)
  closes = np.asarray(closes)
  
  
  # 4.a
  r_i = np.log(closes[1:] / closes[:-1])
  num_r_i = r_i.shape[0]
  r_i_mean = np.mean(r_i)
  r_i_sample_var = np.var(r_i, ddof=1)
  
  print(f"4.a: r_i mean: {r_i_mean:.6f}, r_i sample variance: {r_i_sample_var:.6f}")
  
  
  # 4.b
  ordered_r_i = np.sort(r_i)
  quantiles = np.asarray(statistics.NormalDist(mu=r_i_mean, sigma=np.sqrt(r_i_sample_var)).quantiles(num_r_i + 1))
  
  plt.scatter(quantiles, ordered_r_i)
  
  # plot y = x line
  lims = [
    np.min([plt.gca().get_xlim(), plt.gca().get_ylim()]),  # min of both axes
    np.max([plt.gca().get_xlim(), plt.gca().get_ylim()]),  # max of both axes
  ]
  plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
  plt.title(r"Plot of $r_{(i)}$ vs $\pi_{\frac{i}{n+1}}$")
  plt.show()
  
  
  # 5.a
  n_plus = np.count_nonzero(r_i > 0)
  result = stats.binomtest(n_plus, num_r_i, 0.5, alternative="greater")
  print(f"5.a: Binomial Test: {result}")

  # 5.b
  z_value = (r_i_mean - 0) / np.sqrt(r_i_sample_var / num_r_i)
  print(f"5.b: z_value: {z_value:.3f} against {statistics.NormalDist().inv_cdf(0.95):.3f}")
  
  
  # 5.c
  non_zero_r_i = r_i[r_i != 0]
  r_i_minus_m_0 = r_i - 0
  abs_r_i_minus_m_0 = np.abs(r_i_minus_m_0)
  rank = stats.rankdata(abs_r_i_minus_m_0, method="average")
  signed_rank = np.sign(r_i_minus_m_0) * rank
  W = np.sum(signed_rank)
  
  p_value = 1 - statistics.NormalDist().cdf((W - 1) / np.sqrt(num_r_i * (num_r_i + 1) * (2 * num_r_i + 1) / 6))
  print(f"5.c: W: {W}, p_value: {p_value}")

  