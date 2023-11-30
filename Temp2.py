# %%
import numpy as np
import matplotlib.pyplot as plt

def linear_protocol(a, b, period, repetitions):
    if period % 2 != 0:
        raise ValueError('period must be even')
    peaks = a * period / 2 * np.arange(repetitions * 2 + 1) + b
    peaks = [peak * (-1)**i for i, peak in enumerate(peaks)]
    values = [np.linspace(peaks[i], peaks[i+1], period//2, endpoint=False) for i in range(len(peaks)-1)]
    return np.concatenate(values)

# 파라미터 설정
a = 1
b = 0
period = 100
repetitions = 5

# 함수 실행 및 결과 계산
values = linear_protocol(a, b, period, repetitions)

# 결과 플롯
plt.plot(values)
plt.xlabel('Time step')
plt.ylabel('Displacement')
plt.title('Linear loading protocol')
plt.grid(True)
plt.show()

# %%
