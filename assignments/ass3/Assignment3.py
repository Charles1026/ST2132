import csv
import numpy as np
import scipy.stats

if __name__ == "__main__":
  # Q1
  q1sample = np.asarray([21, 22, 18, 29, 35, 20, 30, 32, 26, 23, 19, 29, 31, 28, 27])
  
  print(f"Q1a. mean {np.mean(q1sample)}, sample variance {np.var(q1sample, ddof=1):.2f}")
  
  
  
  # Q4
  pages = []
  clicks = []
  
  with open("./AB_2024.csv", 'r') as file:
    reader = csv.reader(file)
    
    next(reader)
    
    for row in reader:
      pages.append( 0 if row[2] == 'A' else 1)
      clicks.append(int(row[3]))
      
  pages = np.asarray(pages)
  clicks = np.asarray(clicks)
  
  a_count = np.count_nonzero(pages == 0)
  b_count = np.count_nonzero(pages == 1)
  
  a_prop = np.count_nonzero(clicks[pages == 0]) / a_count
  b_prop = np.count_nonzero(clicks[pages == 1]) / b_count
  
  print(f"Q4b. p_a: {a_prop}, p_b: {b_prop}")
      
  pooled_prop = np.count_nonzero(clicks) / clicks.shape[0]    
  standard_err = np.sqrt(pooled_prop * (1 - pooled_prop) * ((1 / a_count) + (1 / b_count)))
  z_score = (a_prop - b_prop) / standard_err
  
  print(f"Q4c. p_pooled: {pooled_prop:.3f}, standard_error: {standard_err:.3f}, z score: {z_score:.3f}")
  
  # Q5
  weathers = []
  traffics = []
  with open("../ass2/traffic-1.csv", 'r') as file:
    reader = csv.reader(file)
    
    next(reader)
    
    for row in reader:
      weather = row[5]
      
      if (weather == "Clear" or weather == "Rain"):
        weathers.append(0 if row[5] == "Clear" else 1)
        traffics.append(int(row[8]))
        
  weathers = np.asarray(weathers)
  traffics = np.asarray(traffics)
  
  clear_traffic = traffics[weathers == 0]
  rainy_traffic = traffics[weathers == 1]
  
  clear_mean = clear_traffic.mean()
  rainy_mean = rainy_traffic.mean()
  
  clear_var = np.sum((clear_traffic - clear_mean) ** 2)
  rainy_var = np.sum((rainy_traffic - rainy_mean) ** 2)
  
  clear_count = clear_traffic.shape[0]
  rainy_count = rainy_traffic.shape[0]
  
  clear_dof = clear_count - 1
  rainy_dof = rainy_count - 1
  
  pooled_dof = weathers.shape[0] - 2
  pooled_var = (clear_var + rainy_var) / pooled_dof
  pooled_t = (clear_mean - rainy_mean) / (np.sqrt(pooled_var * ((1 / clear_count) + (1 / rainy_count))))
  
  print(f"Q5b. pooled_var: {pooled_var:.3f}, pooled t: {pooled_t:.3f}, pooled d.o.f.: {pooled_dof}")
  
  welch_var = (clear_var / (clear_dof * clear_count)) + (rainy_var / (rainy_dof * rainy_count))
  welch_t = (clear_mean - rainy_mean) / np.sqrt(welch_var)
  welch_dof = np.floor((welch_var ** 2) / ((1 / clear_dof) * ((clear_var / (clear_dof * clear_count)) ** 2) + (1 / rainy_dof) * ((rainy_var / (rainy_dof * rainy_count)) ** 2)))
  
  print(f"Q5c. welch_var: {welch_var:.3f}, welch t: {welch_t:.3f}, welch d.o.f.: {welch_dof}")
  
  f_statistic = (clear_var / clear_dof) / (rainy_var / rainy_dof)
  critical_value = scipy.stats.f.ppf(0.05, dfn = clear_dof, dfd = rainy_dof)
  
  print(clear_dof, rainy_dof, (clear_var / clear_dof))
  print(f"Q5d. f_statistic: {f_statistic:.3f}, critical_value: {critical_value:.3f}")
  
  
  
  