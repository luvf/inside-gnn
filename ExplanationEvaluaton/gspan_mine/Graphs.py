import matplotlib.pyplot as plt
reading_time = [14.9, 13.54, 12.24, 11.23, 10.27]
mining_time = [29.39, 168.83, 3.76, 22.27, 7.05]
total_time = [44.29, 182.37, 16.0, 33.5, 17.32]

positive_labels = [2045, 4278, 5299, 5915, 21697]
total = [5610, 6590, 7531, 8150, 22389]

'''plt.plot(total, reading_time, 'r', label="read")
plt.plot(total, mining_time, "g", label="mine")
plt.plot(total, total_time, "b", label="total")
plt.legend()
plt.savefig("total")
plt.show()'''

plt.plot(positive_labels, reading_time, 'r', label="read")
plt.plot(positive_labels, mining_time, "g", label="mine")
plt.plot(positive_labels, total_time, "b", label="total")
plt.legend()
plt.savefig("positive")
plt.show()