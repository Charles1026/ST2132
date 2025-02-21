import csv
import numpy as np
import statistics as stats # Only available on 3.8 and above


if __name__ == "__main__":
  weather = []
  traffic = []
  
  with open("./traffic-1.csv", 'r') as file:
    reader = csv.reader(file)
    reader.__next__() # skip headers
    
    for row in reader:
      weather_main = row[5]
      traffic_volume = int(row[8])
      
      if (weather_main == "Clear" or weather_main == "Rain"):
        weather.append(0 if weather_main == "Clear" else 1)
        traffic.append(traffic_volume)
        
  # a
  weather = np.asarray(weather)
  traffic = np.asarray(traffic)
  x = traffic[weather == 0]
  y = traffic[weather == 1]
  x_mean = x.mean()
  y_mean = y.mean()

  print(f"Clear Weather Traffic Volume: {x_mean:.2f}\nRainy Weather Traffic Volume: {y_mean:.2f}")
  
  # b
  x_count = x.shape[0]
  y_count = y.shape[0]
  
  pooled_var = (np.square(x - x_mean).sum() + np.square(y - y_mean) .sum()) / (x_count + y_count - 2)
  pooled_half_width = 1.960 * np.sqrt(pooled_var) * np.sqrt((1/x_count) + (1/y_count))
  pooled_lower_bound = x_mean - y_mean - pooled_half_width
  pooled_upper_bound = x_mean - y_mean + pooled_half_width
  
  print(f"Pooled t-Interval [{pooled_lower_bound:.2f}, {pooled_upper_bound:.2f}]")
  
  # c
  x_sample_var = x.var(ddof = 1)
  y_sample_var = y.var(ddof = 1)
  
  welch_half_width = 1.960 * np.sqrt((x_sample_var/x_count) + (y_sample_var/y_count))
  welch_lower_bound = x_mean - y_mean - welch_half_width
  welch_upper_bound = x_mean - y_mean + welch_half_width
  
  print(f"Welch t-Interval [{welch_lower_bound:.2f}, {welch_upper_bound:.2f}]")