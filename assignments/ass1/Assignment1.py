import csv

if __name__ == "__main__":
  with open("./GMM.csv") as file:
    data = csv.reader(file, )
    
    # find proportion and means
    totalCount = 0
    counts = [0, 0]
    sums = [0.0, 0.0]
    
    next(data, None)
    for row in data:
      K = int(row[0])
      X = float(row[1])
      totalCount += 1
      counts[K] += 1
      sums[K] += X
      
    means = [sum / count for (sum, count) in zip(sums, counts)]
    
    file.seek(0) # reset file ptr
    
    # find var with mean est
    sqrSums = [0.0, 0.0]
    
    next(data, None)
    for row in data:
      K = int(row[0])
      X = float(row[1])
      sqrSums[K] += (X - means[K]) ** 2
      
    vars = [sqrSum / count for (sqrSum, count) in zip(sqrSums, counts)]
    
    print(f"Class 0 has count {counts[0]}, proportion: {counts[0] / totalCount:.4f}, mean: {means[0]:.4f}, var: {vars[0]:.4f}\nClass 1 has count {counts[1]}, proportion: {counts[1] / totalCount:.4f}, mean: {means[1]:.4f}, var: {vars[1]:.4f}")